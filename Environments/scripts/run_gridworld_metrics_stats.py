# run_gridworld_metrics_stats.py
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from gridworld_env import GridWorld
from ai_agent_factory import build_gridworld_agent

# ---------- math utils ----------
def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x); ex = np.exp(x); z = ex.sum()
    return ex / (z if z > 0 else 1.0)

def kl_div(p, q, eps=1e-16):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def normalize(p, eps=1e-16):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None); s = p.sum()
    return p / (s if s > 0 else 1.0)

def moving_average(x, w):
    if w <= 1: return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / float(w)
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, ma])

# ---------- pymdp compatibility ----------
def infer_states_compat(agent, obs):
    try: agent.infer_states(obs)
    except Exception: agent.infer_states([obs])

def sample_action_compat(agent) -> int:
    a = agent.sample_action()
    if isinstance(a, (list, tuple, np.ndarray)): return int(a[0])
    return int(a)

def reset_agent_compat(agent):
    if hasattr(agent, "reset") and callable(agent.reset):
        try: agent.reset()
        except Exception: pass

def infer_policies_compat(agent, sophisticated: bool, controls=None):
    if controls and callable(controls.get("infer_policies", None)):
        return controls["infer_policies"]()
    if sophisticated:
        for kw in ({"mode":"sophisticated"}, {"method":"sophisticated"}):
            try: return agent.infer_policies(**kw)
            except TypeError: pass
    return agent.infer_policies()

def get_qs_compat(agent, expected_S: int | None = None):
    def coerce_1d(x):
        try:
            a = np.asarray(x, dtype=np.float64)
            if a.ndim > 1: a = np.squeeze(a)
            if a.ndim == 1 and a.size > 0: return a
        except Exception: pass
        return None
    candidates = []
    qs_attr = getattr(agent, "qs", None)
    if qs_attr is not None:
        if isinstance(qs_attr, (list, tuple)):
            for item in qs_attr:
                v = coerce_1d(item);  candidates.append(v) if v is not None else None
        elif isinstance(qs_attr, np.ndarray) and qs_attr.dtype == object:
            for item in qs_attr.ravel():
                v = coerce_1d(item);  candidates.append(v) if v is not None else None
        else:
            v = coerce_1d(qs_attr);   candidates.append(v) if v is not None else None
    if not candidates:
        for name in ("q_s", "qs_current", "qs_prev"):
            v = coerce_1d(getattr(agent, name, None))
            if v is not None: candidates.append(v)
    if not candidates: return None
    if expected_S is not None:
        for v in candidates:
            if v.shape[0] == expected_S: return normalize(v)
    return normalize(max(candidates, key=lambda a: a.shape[0]))

def get_models(agent, model_dict):
    A = model_dict.get("A", getattr(agent, "A", None))
    if isinstance(A, (list, tuple)): A = A[0]
    A = np.asarray(A, dtype=np.float64)

    B = model_dict.get("B", getattr(agent, "B", None))
    if isinstance(B, (list, tuple)): B = B[0]
    B = np.asarray(B, dtype=np.float64)  # (S,S,U)

    C = model_dict.get("C", getattr(agent, "C", None))
    if isinstance(C, (list, tuple)): C = C[0]
    C = np.asarray(C, dtype=np.float64)

    D = model_dict.get("D", getattr(agent, "D", None))
    if isinstance(D, (list, tuple)): D = D[0]
    D = np.asarray(D, dtype=np.float64)

    return A, B, C, D

# ---------- noise makers ----------
def make_noisy_A_identity(S: int, eps: float):
    A = (1.0 - eps) * np.eye(S) + eps * (1.0 / S) * np.ones((S, S))
    A = A / np.clip(A.sum(axis=0, keepdims=True), 1e-12, None)
    return A

def make_noisy_B(B_base: np.ndarray, eps: float) -> np.ndarray:
    B_base = np.asarray(B_base, dtype=np.float64)
    S, S2, U = B_base.shape
    assert S == S2 and 0.0 <= eps < 1.0
    Bn = (1.0 - eps) * B_base + eps * (1.0 / S) * np.ones_like(B_base)
    colsum = Bn.sum(axis=0, keepdims=True)
    Bn = Bn / np.clip(colsum, 1e-12, None)
    return Bn

# ---------- per-step metrics ----------
def step_metrics(qs, prior_s, A, C, obs_idx):
    qs = normalize(qs); prior_s = normalize(prior_s)
    complexity = kl_div(qs, prior_s)
    lik_col = np.clip(A[obs_idx, :], 1e-16, 1.0)
    accuracy  = - float(np.sum(qs * np.log(lik_col)))
    q_o = normalize(A @ qs)
    pC = softmax(C)
    extrinsic = - float(np.sum(q_o * np.log(np.clip(pC, 1e-16, 1.0))))
    epistemic = 0.0
    for o in range(A.shape[0]):
        post_o = normalize(A[o, :] * qs)
        epistemic += q_o[o] * kl_div(post_o, qs)
    return complexity, accuracy, extrinsic, float(epistemic)

# ---------- worker ----------
def _worker_run_chunk(
    seed: int, episodes: int,
    rows: int, cols: int,
    reward_pos: tuple[int,int], punish_pos: tuple[int,int], start_pos: tuple[int,int] | None,
    step_cost: float, reward: float, punish: float, max_steps: int, slip_prob: float,
    policy_len: int, gamma: float, act_sel: str, c_reward: float, c_punish: float,
    sophisticated: bool, a_noise: float, b_noise: float,
):
    import numpy as np
    from gridworld_env import GridWorld
    from ai_agent_factory import build_gridworld_agent

    rng = np.random.default_rng(seed)

    env = GridWorld(
        n_rows=rows, n_cols=cols,
        reward_pos=reward_pos, punish_pos=punish_pos, start_pos=start_pos,
        step_cost=step_cost, reward=reward, punish=punish,
        max_steps=max_steps, slip_prob=slip_prob, render_mode=None
    )

    out_ret = np.zeros(episodes); out_steps = np.zeros(episodes, dtype=int); out_outcomes = []
    m_complex = np.zeros(episodes); m_acc = np.zeros(episodes); m_extr = np.zeros(episodes); m_epi = np.zeros(episodes)

    for ep in range(episodes):
        # build fresh agent each episode to avoid history carry-over
        factory_out = build_gridworld_agent(
            n_rows=rows, n_cols=cols,
            reward_pos=reward_pos, punish_pos=punish_pos, start_pos=start_pos,
            c_reward=c_reward, c_punish=c_punish,
            policy_len=policy_len, gamma=gamma, action_selection=act_sel,
            sophisticated=sophisticated,
        )
        if isinstance(factory_out, tuple) and len(factory_out) == 3:
            agent, model, controls = factory_out
        else:
            agent, model = factory_out; controls = None

        A, B, C, D = get_models(agent, model)

        # apply A/B noise (agent-side)
        if a_noise > 0.0:
            S = A.shape[1]
            A = make_noisy_A_identity(S, a_noise)
            try: agent.A = A
            except Exception: pass
            model["A"] = A
        if b_noise > 0.0:
            B = make_noisy_B(B, b_noise)
            try: agent.B = [B]
            except Exception: pass
            model["B"] = [B]

        # episode loop
        reset_agent_compat(agent)
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total_r, steps = 0.0, 0
        done = False

        # for metrics averaging
        sum_complex = sum_acc = sum_extr = sum_epi = 0.0
        prior_s = normalize(D.copy())

        while not done:
            infer_states_compat(agent, obs)
            qs = get_qs_compat(agent, expected_S=A.shape[1])
            if qs is None:
                S = A.shape[1]
                qs = np.zeros(S); qs[int(obs)] = 1.0

            cplx, acc, extr, epi = step_metrics(qs, prior_s, A, C, int(obs))
            sum_complex += cplx; sum_acc += acc; sum_extr += extr; sum_epi += epi

            infer_policies_compat(agent, sophisticated, controls)
            action = sample_action_compat(agent)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r; steps += 1
            done = terminated or truncated

            # predictive prior for next step (agent's B)
            prior_s = normalize(B[:, :, action] @ qs)

        out_ret[ep]   = total_r
        out_steps[ep] = steps
        out_outcomes.append("reward" if env.pos == env.reward_pos else ("punish" if env.pos == env.punish_pos else "timeout"))
        # per-episode means
        denom = max(steps, 1)
        m_complex[ep] = sum_complex / denom
        m_acc[ep]     = sum_acc / denom
        m_extr[ep]    = sum_extr / denom
        m_epi[ep]     = sum_epi / denom

    return out_ret, out_steps, out_outcomes, m_complex, m_acc, m_extr, m_epi

# ---------- helpers ----------
def parse_pos(s):
    if s.strip().lower() == "random": return None
    r, c = s.split(","); return (int(r), int(c))

def summarize(returns, steps, outcomes):
    cnt = Counter(outcomes)
    return {
        "success_rate": cnt.get("reward", 0) / len(outcomes),
        "punish_rate":  cnt.get("punish", 0) / len(outcomes),
        "timeout_rate": cnt.get("timeout", 0) / len(outcomes),
        "avg_return":   float(np.mean(returns)),
        "avg_steps":    float(np.mean(steps)),
        "counts":       cnt,
    }

def smart_bins(x, max_bins=40):
    """Return safe bin edges for histograms (handles constant arrays)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1  # matplotlib will show empty hist gracefully

    xmin, xmax = np.min(x), np.max(x)
    if np.isclose(xmin, xmax):
        # Constant array: make a single narrow bin around the value
        width = 1.0 if xmax == 0 else abs(xmax) * 0.1
        if width == 0:
            width = 1.0
        return np.array([xmax - width, xmax + width])

    # Use 'auto' then cap the number of bins
    edges = np.histogram_bin_edges(x, bins='auto')
    if edges.size - 1 > max_bins:
        edges = np.histogram_bin_edges(x, bins=max_bins)
    return edges

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Batch stats with A- and B-noise: returns, steps, outcomes, and FE terms.")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--reward-pos", type=str, default="4,6")
    ap.add_argument("--punish-pos", type=str, default="0,6")
    ap.add_argument("--start-pos", type=str, default="0,0")
    ap.add_argument("--step-cost", type=float, default=0.0)
    ap.add_argument("--reward", type=float, default=1.0)
    ap.add_argument("--punish", type=float, default=-1.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--slip-prob", type=float, default=0.0)

    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    # agent hyperparams
    ap.add_argument("--policy-len", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=16.0)
    ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic","deterministic"])
    ap.add_argument("--c-reward", type=float, default=3.0)
    ap.add_argument("--c-punish", type=float, default=-3.0)
    ap.add_argument("--sophisticated", action="store_true")

    # model noise
    ap.add_argument("--a-noise", type=float, default=0.0, help="ε in A'=(1-ε)I+ε*1/O")
    ap.add_argument("--b-noise", type=float, default=0.0, help="ε in B'=(1-ε)B+ε*1/S")

    # plots
    ap.add_argument("--ma-window", type=int, default=50)
    ap.add_argument("--savefig", type=str, default="")
    args = ap.parse_args()

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos  = parse_pos(args.start_pos)

    # split workload
    base = args.episodes // args.workers
    rem  = args.episodes %  args.workers
    sizes = [base + (1 if i < rem else 0) for i in range(args.workers)]
    seeds = [int(args.seed + i * 1000003) for i in range(args.workers)]

    returns = np.zeros(args.episodes)
    steps   = np.zeros(args.episodes, dtype=int)
    outcomes = []
    m_complex = np.zeros(args.episodes)
    m_acc     = np.zeros(args.episodes)
    m_extr    = np.zeros(args.episodes)
    m_epi     = np.zeros(args.episodes)

    idx = 0
    if args.workers <= 1:
        (r, s, o, mc, ma, me, mi) = _worker_run_chunk(
            seeds[0], sizes[0],
            args.rows, args.cols,
            reward_pos, punish_pos, start_pos,
            args.step_cost, args.reward, args.punish, args.max_steps, args.slip_prob,
            args.policy_len, args.gamma, args.act_sel, args.c_reward, args.c_punish,
            args.sophisticated, args.a_noise, args.b_noise,
        )
        returns[:] = r; steps[:] = s; outcomes = list(o)
        m_complex[:] = mc; m_acc[:] = ma; m_extr[:] = me; m_epi[:] = mi
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n_eps, sseed in zip(sizes, seeds):
                fut = ex.submit(
                    _worker_run_chunk,
                    sseed, n_eps,
                    args.rows, args.cols,
                    reward_pos, punish_pos, start_pos,
                    args.step_cost, args.reward, args.punish, args.max_steps, args.slip_prob,
                    args.policy_len, args.gamma, args.act_sel, args.c_reward, args.c_punish,
                    args.sophisticated, args.a_noise, args.b_noise,
                )
                futures.append((idx, n_eps, fut)); idx += n_eps
            for start, n_eps, fut in futures:
                (r, s, o, mc, ma, me, mi) = fut.result()
                end = start + n_eps
                returns[start:end] = r; steps[start:end] = s
                m_complex[start:end] = mc; m_acc[start:end] = ma
                m_extr[start:end]    = me; m_epi[start:end] = mi
                outcomes.extend(o)

    # summaries
    sums = summarize(returns, steps, outcomes)
    print("\n=== Summary ===")
    for k, v in sums.items():
        if k != "counts": print(f"{k:>13}: {v}")
    print("counts:", dict(sums["counts"]))
    print(f"\nPer-episode means (overall): "
          f"Complexity={np.mean(m_complex):.3f}, Accuracy={np.mean(m_acc):.3f}, "
          f"Extrinsic={np.mean(m_extr):.3f}, Epistemic={np.mean(m_epi):.3f}")

    # plots
    fig = plt.figure(figsize=(12, 10))

    # 1) Returns with MA
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(returns, alpha=0.35, lw=0.8, label="Return")
    ax1.plot(moving_average(returns, args.ma_window), lw=2, label=f"Return (MA={args.ma_window})")
    ax1.set_title("Episode Returns"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3); ax1.legend()

    # 2) Steps (MA)
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(moving_average(steps.astype(float), args.ma_window), lw=2, label="Steps (MA)")
    ax2.set_title("Steps per Episode (MA)"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3); ax2.legend()

    # 3) Outcomes
    ax3 = plt.subplot(3, 2, 3)
    labels = ["reward", "punish", "timeout"]
    counts = np.array([sums["counts"].get(k, 0) for k in labels])
    x = np.arange(len(labels)); w = 0.6
    ax3.bar(x, counts, width=w)
    ax3.set_xticks(x); ax3.set_xticklabels(labels); ax3.set_title("Outcome Counts"); ax3.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax3.text(i, v, str(int(v)), ha="center", va="bottom")





    # 4) Histograms of per-episode mean FE terms
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(m_complex, bins=smart_bins(m_complex, max_bins=40), alpha=0.8)
    ax4.set_title("Complexity (mean/ep)")
    ax4.set_xlabel("Value"); ax4.set_ylabel("Episodes")

    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(m_acc, bins=smart_bins(m_acc, max_bins=40), alpha=0.8)
    ax5.set_title("Accuracy (mean/ep)")
    ax5.set_xlabel("Value"); ax5.set_ylabel("Episodes")

    ax6 = plt.subplot(3, 2, 6)
    ax6.hist(m_extr, bins=smart_bins(m_extr, max_bins=40), alpha=0.6, label="Extrinsic")
    ax6.hist(m_epi,  bins=smart_bins(m_epi,  max_bins=40), alpha=0.6, label="Epistemic")
    ax6.set_title("Extrinsic vs Epistemic (mean/ep)")
    ax6.set_xlabel("Value"); ax6.set_ylabel("Episodes"); ax6.legend()









    # # 4) Histograms of per-episode mean FE terms
    # ax4 = plt.subplot(3, 2, 4)
    # ax4.hist(m_complex, bins=40, alpha=0.8); ax4.set_title("Complexity (mean/ep)")
    # ax4.set_xlabel("Value"); ax4.set_ylabel("Episodes")
    #
    # ax5 = plt.subplot(3, 2, 5)
    # ax5.hist(m_acc, bins=40, alpha=0.8); ax5.set_title("Accuracy (mean/ep)")
    # ax5.set_xlabel("Value"); ax5.set_ylabel("Episodes")
    #
    # ax6 = plt.subplot(3, 2, 6)
    # # overlay Extrinsic & Epistemic hist for quick compare
    # ax6.hist(m_extr, bins=40, alpha=0.6, label="Extrinsic")
    # ax6.hist(m_epi,  bins=40, alpha=0.6, label="Epistemic")
    # ax6.set_title("Extrinsic vs Epistemic (mean/ep)")
    # ax6.set_xlabel("Value"); ax6.set_ylabel("Episodes"); ax6.legend()

    fig.suptitle(
        f"GridWorld {args.rows}x{args.cols} — Episodes={args.episodes}, workers={args.workers}, slip={args.slip_prob}\n"
        f"A-noise ε={args.a_noise}, B-noise ε={args.b_noise}, Sophisticated={args.sophisticated} | "
        f"Avg FE terms: Cplx={np.mean(m_complex):.3f}, Acc={np.mean(m_acc):.3f}, Extr={np.mean(m_extr):.3f}, Epi={np.mean(m_epi):.3f}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    if args.savefig:
        plt.savefig(args.savefig, dpi=150)
        print(f"Saved figure to: {args.savefig}")

    plt.show()

if __name__ == "__main__":
    main()
