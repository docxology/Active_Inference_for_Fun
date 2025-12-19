# run_gridworld_obs_noise.py
import argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from gridworld_env import GridWorld
from ai_agent_factory import build_gridworld_agent


# ---------- Small compatibility helpers ----------
def infer_states_compat(agent, obs):
    try:
        agent.infer_states(obs)
    except Exception:
        agent.infer_states([obs])

def sample_action_compat(agent) -> int:
    a = agent.sample_action()
    if isinstance(a, (list, tuple, np.ndarray)):
        return int(a[0])
    return int(a)

def reset_agent_compat(agent):
    if hasattr(agent, "reset") and callable(agent.reset):
        try: agent.reset()
        except Exception: pass

def infer_policies_compat(agent, sophisticated: bool, controls=None):
    if controls and callable(controls.get("infer_policies", None)):
        return controls["infer_policies"]()
    if sophisticated:
        for kw in ({"mode": "sophisticated"}, {"method": "sophisticated"}):
            try:
                return agent.infer_policies(**kw)
            except TypeError:
                pass
    return agent.infer_policies()

def get_map_state_idx_compat(agent):
    """
    Try to read posterior over states (single factor) and return MAP index.
    Returns None if unavailable (older pymdp).
    """
    q = getattr(agent, "qs", None)
    if q is None:
        return None
    if isinstance(q, (list, tuple)):
        q0 = q[0]
    else:
        q0 = q
    try:
        return int(np.argmax(q0))
    except Exception:
        return None


# ---------- Make a noisy A from an identity base ----------
def make_noisy_A(S: int, eps: float):
    """
    A_noisy = (1-eps)*I + eps*(1/S) * 1, column-normalized.
    Shape: (O,S) with O=S here (fully observable base case).
    """
    assert 0.0 <= eps < 1.0, "noise eps must be in [0,1)"
    A = (1.0 - eps) * np.eye(S, dtype=np.float64) + eps * (1.0 / S) * np.ones((S, S), dtype=np.float64)
    # Normalize columns (robustness)
    colsum = A.sum(axis=0, keepdims=True)
    A = A / np.clip(colsum, 1e-12, None)
    return A


# ---------- Worker (parallel-safe) ----------
def _worker_run_chunk(
    seed: int,
    episodes: int,
    rows: int, cols: int,
    reward_pos: tuple[int,int], punish_pos: tuple[int,int], start_pos: tuple[int,int] | None,
    step_cost: float, reward: float, punish: float, max_steps: int, slip_prob: float,
    policy_len: int, gamma: float, act_sel: str, c_reward: float, c_punish: float,
    sophisticated: bool, a_noise: float,
):
    import numpy as np
    from gridworld_env import GridWorld
    from ai_agent_factory import build_gridworld_agent

    # Local copies of compat helpers (avoid capturing outer objects)
    def infer_states_compat_local(agent, obs):
        try: agent.infer_states(obs)
        except Exception: agent.infer_states([obs])

    def sample_action_compat_local(agent) -> int:
        a = agent.sample_action()
        return int(a[0] if isinstance(a, (list, tuple, np.ndarray)) else a)

    def reset_agent_compat_local(agent):
        if hasattr(agent, "reset") and callable(agent.reset):
            try: agent.reset()
            except Exception: pass

    def infer_policies_compat_local(agent, sophisticated: bool, controls=None):
        if controls and callable(controls.get("infer_policies", None)):
            return controls["infer_policies"]()
        if sophisticated:
            for kw in ({"mode": "sophisticated"}, {"method": "sophisticated"}):
                try:
                    return agent.infer_policies(**kw)
                except TypeError:
                    pass
        return agent.infer_policies()

    def get_map_state_idx_local(agent):
        q = getattr(agent, "qs", None)
        if q is None: return None
        if isinstance(q, (list, tuple)): q0 = q[0]
        else: q0 = q
        try: return int(np.argmax(q0))
        except Exception: return None

    rng = np.random.default_rng(seed)

    # Env
    env = GridWorld(
        n_rows=rows, n_cols=cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        step_cost=step_cost, reward=reward, punish=punish,
        max_steps=max_steps, slip_prob=slip_prob,
        render_mode=None
    )

    # Build agent (classic or sophisticated-aware)
    factory_out = build_gridworld_agent(
        n_rows=rows, n_cols=cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        c_reward=c_reward, c_punish=c_punish,
        policy_len=policy_len, gamma=gamma, action_selection=act_sel,
        sophisticated=sophisticated,
    )
    if isinstance(factory_out, tuple) and len(factory_out) == 3:
        agent, model, controls = factory_out
    else:
        agent, model = factory_out
        controls = None

    # Overwrite A with a noisy version (same shape + normalization)
    S = rows * cols
    A_noisy = make_noisy_A(S, a_noise)
    # Try to set on agent and in model (some versions check consistency)
    try: agent.A = A_noisy
    except Exception: pass
    try: model["A"] = A_noisy
    except Exception: pass

    def run_episode_aif_local():
        reset_agent_compat_local(agent)
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total_r, steps, outcome = 0.0, 0, "timeout"
        mismatch_steps = 0
        while True:
            # Inference + policy
            infer_states_compat_local(agent, obs)

            # belief mismatch metric (MAP vs true state)
            map_idx = get_map_state_idx_local(agent)
            true_idx = obs  # observation is position index in this env
            if map_idx is not None and map_idx != int(true_idx):
                mismatch_steps += 1

            infer_policies_compat_local(agent, sophisticated, controls)
            action = sample_action_compat_local(agent)

            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r; steps += 1
            if terminated:
                outcome = "reward" if env.pos == env.reward_pos else ("punish" if env.pos == env.punish_pos else "terminal")
                break
            if truncated:
                outcome = "timeout"
                break
        # ratio in [0,1]
        mismatch_ratio = (mismatch_steps / steps) if steps > 0 else 0.0
        return total_r, steps, outcome, mismatch_ratio

    def run_episode_random_local():
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total_r, steps, outcome = 0.0, 0, "timeout"
        while True:
            action = int(rng.integers(0, env.action_space.n))
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r; steps += 1
            if terminated:
                outcome = "reward" if env.pos == env.reward_pos else ("punish" if env.pos == env.punish_pos else "terminal")
                break
            if truncated:
                outcome = "timeout"
                break
        return total_r, steps, outcome

    returns_aif  = np.zeros(episodes)
    steps_aif    = np.zeros(episodes, dtype=int)
    outcomes_aif = []
    belief_err_aif = np.zeros(episodes)

    returns_rand = np.zeros(episodes)
    steps_rand   = np.zeros(episodes, dtype=int)
    outcomes_rand = []

    for i in range(episodes):
        # Random baseline
        tr, st, oc = run_episode_random_local()
        returns_rand[i] = tr; steps_rand[i] = st; outcomes_rand.append(oc)

        # AIF with noisy A
        tr, st, oc, mr = run_episode_aif_local()
        returns_aif[i] = tr; steps_aif[i] = st; outcomes_aif.append(oc); belief_err_aif[i] = mr

    return (
        returns_rand, steps_rand, outcomes_rand,
        returns_aif,  steps_aif,  outcomes_aif, belief_err_aif
    )


# ---------- Utilities ----------
def moving_average(x, w):
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / float(w)
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, ma])


# ---------- CLI & Orchestration ----------
def parse_pos(s):
    if s.strip().lower() == "random":
        return None
    r, c = s.split(",")
    return (int(r), int(c))

def main():
    ap = argparse.ArgumentParser(description="GridWorld with observation noise in A; parallel episodes + plots.")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--reward-pos", type=str, default="4,6")
    ap.add_argument("--punish-pos", type=str, default="0,6")
    ap.add_argument("--start-pos", type=str, default="0,0", help="'random' for random starts")
    ap.add_argument("--step-cost", type=float, default=0.0)
    ap.add_argument("--reward", type=float, default=1.0)
    ap.add_argument("--punish", type=float, default=-1.0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--slip-prob", type=float, default=0.0)

    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    # Agent
    ap.add_argument("--policy-len", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=16.0)
    ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic", "deterministic"])
    ap.add_argument("--c-reward", type=float, default=3.0)
    ap.add_argument("--c-punish", type=float, default=-3.0)
    ap.add_argument("--sophisticated", action="store_true", help="request sophisticated inference if supported")

    # Noise in A
    ap.add_argument("--a-noise", type=float, default=0.15,
                    help="ε in A_noisy = (1-ε)I + ε*1/O (0 ≤ ε < 1)")

    # Plots
    ap.add_argument("--ma-window", type=int, default=50)
    ap.add_argument("--savefig", type=str, default="")
    args = ap.parse_args()

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos  = parse_pos(args.start_pos)

    # Split work
    base = args.episodes // args.workers
    rem  = args.episodes %  args.workers
    sizes = [base + (1 if i < rem else 0) for i in range(args.workers)]
    seeds = [int(args.seed + i * 1000003) for i in range(args.workers)]

    returns_rand = np.zeros(args.episodes)
    steps_rand   = np.zeros(args.episodes, dtype=int)
    outcomes_rand = []
    returns_aif  = np.zeros(args.episodes)
    steps_aif    = np.zeros(args.episodes, dtype=int)
    outcomes_aif = []
    belief_err_aif = np.zeros(args.episodes)

    idx = 0
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n_eps, s in zip(sizes, seeds):
            fut = ex.submit(
                _worker_run_chunk,
                s, n_eps,
                args.rows, args.cols,
                reward_pos, punish_pos, start_pos,
                args.step_cost, args.reward, args.punish, args.max_steps, args.slip_prob,
                args.policy_len, args.gamma, args.act_sel, args.c_reward, args.c_punish,
                args.sophisticated, args.a_noise,
            )
            futures.append((idx, n_eps, fut))
            idx += n_eps

        for start, n_eps, fut in futures:
            (rR, sR, oR, rA, sA, oA, bA) = fut.result()
            end = start + n_eps
            returns_rand[start:end] = rR
            steps_rand[start:end]   = sR
            outcomes_rand.extend(oR)

            returns_aif[start:end]  = rA
            steps_aif[start:end]    = sA
            outcomes_aif.extend(oA)
            belief_err_aif[start:end] = bA

    # ---------- Summaries ----------
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

    sum_r = summarize(returns_rand, steps_rand, outcomes_rand)
    sum_a = summarize(returns_aif, steps_aif, outcomes_aif)
    avg_belief_err = float(np.mean(belief_err_aif))

    print("\n=== Summary (Random) ===")
    for k, v in sum_r.items():
        if k != "counts": print(f"{k:>13}: {v}")
    print("counts:", dict(sum_r["counts"]))

    print("\n=== Summary (AIF, noisy A) ===")
    for k, v in sum_a.items():
        if k != "counts": print(f"{k:>13}: {v}")
    print("counts:", dict(sum_a["counts"]))
    print(f" belief_error_ratio (mean over episodes): {avg_belief_err:.3f}\n")

    # ---------- Plots ----------
    fig = plt.figure(figsize=(12, 9))

    # 1) Returns with MA
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(returns_rand, alpha=0.35, lw=0.8, label="Random")
    ax1.plot(returns_aif,  alpha=0.35, lw=0.8, label=f"AIF (ε={args.a_noise})")
    ax1.plot(moving_average(returns_rand, args.ma_window), lw=2, label=f"Random (MA={args.ma_window})")
    ax1.plot(moving_average(returns_aif,  args.ma_window), lw=2, label=f"AIF (MA={args.ma_window})")
    ax1.set_title("Episode Returns"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3); ax1.legend()

    # 2) Steps per episode (MA)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(moving_average(steps_rand.astype(float), args.ma_window), lw=2, label="Random (MA)")
    ax2.plot(moving_average(steps_aif.astype(float),  args.ma_window), lw=2, label=f"AIF (MA, ε={args.a_noise})")
    ax2.set_title("Steps per Episode (MA)"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3); ax2.legend()

    # 3) Outcomes (side-by-side bars)
    ax3 = plt.subplot(2, 2, 3)
    labels = ["reward", "punish", "timeout"]
    r_counts = np.array([sum_r["counts"].get(k, 0) for k in labels])
    a_counts = np.array([sum_a["counts"].get(k, 0) for k in labels])
    x = np.arange(len(labels)); w = 0.4
    ax3.bar(x - w/2, r_counts, width=w, label="Random")
    ax3.bar(x + w/2, a_counts, width=w, label=f"AIF (ε={args.a_noise})")
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_title("Episode Outcomes"); ax3.set_ylabel("Count"); ax3.legend()

    for i, (rv, av) in enumerate(zip(r_counts, a_counts)):
        ax3.text(i - w/2, rv, str(int(rv)), ha="center", va="bottom")
        ax3.text(i + w/2, av, str(int(av)), ha="center", va="bottom")

    # 4) Belief error ratio distribution
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(belief_err_aif, bins=30, alpha=0.85)
    ax4.set_title(f"AIF Belief Error Ratio (ε={args.a_noise})")
    ax4.set_xlabel("Fraction of steps (MAP state ≠ true)"); ax4.set_ylabel("Episodes")
    ax4.axvline(np.mean(belief_err_aif), linestyle="--", label=f"mean={np.mean(belief_err_aif):.3f}")
    ax4.legend()

    fig.suptitle(
        f"GridWorld {args.rows}x{args.cols} — Episodes={args.episodes}, workers={args.workers}, slip={args.slip_prob}, "
        f"Sophisticated={args.sophisticated}\n"
        f"A-noise ε={args.a_noise} | AIF: success={sum_a['success_rate']:.3f}, avgR={sum_a['avg_return']:.3f}, "
        f"avgSteps={sum_a['avg_steps']:.1f}, beliefErr={avg_belief_err:.3f} | "
        f"Random: success={sum_r['success_rate']:.3f}, avgR={sum_r['avg_return']:.3f}, avgSteps={sum_r['avg_steps']:.1f}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    if args.savefig:
        plt.savefig(args.savefig, dpi=150)
        print(f"Saved figure to: {args.savefig}")

    plt.show()


if __name__ == "__main__":
    main()
