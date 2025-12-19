# run_gridworld_live_metrics.py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gridworld_env import GridWorld
from ai_agent_factory import build_gridworld_agent


# --------- small math utils ---------
def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    z = ex.sum()
    return ex / (z if z > 0 else 1.0)

def kl_div(p, q, eps=1e-16):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def normalize(p, eps=1e-16):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum()
    return p / (s if s > 0 else 1.0)


# --------- pymdp compatibility helpers ---------
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
    # factory-provided wrapper (best)
    if controls and callable(controls.get("infer_policies", None)):
        return controls["infer_policies"]()
    # common kwargs across pymdp variants
    if sophisticated:
        for kw in ({"mode": "sophisticated"}, {"method": "sophisticated"}):
            try:
                return agent.infer_policies(**kw)
            except TypeError:
                pass
    return agent.infer_policies()

def get_qs_compat(agent, expected_S: int | None = None):
    """
    Return a 1-D numpy array for the posterior over states (single factor).
    Works across pymdp variants where agent.qs can be:
      - a numpy array
      - a list/tuple of arrays
      - an object array holding arrays
    If expected_S is given, prefer vectors with that length.
    """
    def coerce_1d(x):
        try:
            a = np.asarray(x, dtype=np.float64)
            if a.ndim > 1:
                a = np.squeeze(a)
            if a.ndim == 1 and a.size > 0:
                return a
        except Exception:
            pass
        return None

    candidates = []

    qs_attr = getattr(agent, "qs", None)
    if qs_attr is not None:
        if isinstance(qs_attr, (list, tuple)):
            for item in qs_attr:
                v = coerce_1d(item)
                if v is not None:
                    candidates.append(v)
        elif isinstance(qs_attr, np.ndarray) and qs_attr.dtype == object:
            for item in qs_attr.ravel():
                v = coerce_1d(item)
                if v is not None:
                    candidates.append(v)
        else:
            v = coerce_1d(qs_attr)
            if v is not None:
                candidates.append(v)

    # Fallback probes (some versions use these names)
    if not candidates:
        for name in ("q_s", "qs_current", "qs_prev"):
            v = coerce_1d(getattr(agent, name, None))
            if v is not None:
                candidates.append(v)

    if not candidates:
        return None

    if expected_S is not None:
        for v in candidates:
            if v.shape[0] == expected_S:
                return normalize(v)

    # Otherwise pick the longest 1-D vector
    best = max(candidates, key=lambda a: a.shape[0])
    return normalize(best)

def get_models(agent, model_dict):
    # A
    A = model_dict.get("A", None)
    if A is None and hasattr(agent, "A"):
        A = getattr(agent, "A")
    if isinstance(A, (list, tuple)):
        A = A[0]
    A = np.asarray(A, dtype=np.float64)

    # B
    B = model_dict.get("B", None)
    if B is None and hasattr(agent, "B"):
        B = getattr(agent, "B")
    if isinstance(B, (list, tuple)):
        B = B[0]
    B = np.asarray(B, dtype=np.float64)  # (S,S,U)

    # C (we’ll convert to probs via softmax for plotting extrinsic)
    C = model_dict.get("C", None)
    if C is None and hasattr(agent, "C"):
        C = getattr(agent, "C")
    if isinstance(C, (list, tuple)):
        C = C[0]
    C = np.asarray(C, dtype=np.float64)

    # D
    D = model_dict.get("D", None)
    if D is None and hasattr(agent, "D"):
        D = getattr(agent, "D")
    if isinstance(D, (list, tuple)):
        D = D[0]
    D = np.asarray(D, dtype=np.float64)

    return A, B, C, D

def hard_reset_agent(agent, D0: np.ndarray):
    D0 = np.asarray(D0, dtype=np.float64)
    D0 = D0 / max(D0.sum(), 1e-12)
    # reset() if available
    if hasattr(agent, "reset") and callable(agent.reset):
        try:
            agent.reset()
        except Exception:
            pass
    # force prior and posterior beliefs
    try:
        agent.D = D0.copy()
    except Exception:
        pass
    if hasattr(agent, "qs"):
        try:
            if isinstance(agent.qs, (list, tuple)):
                agent.qs[0] = D0.copy()
            else:
                agent.qs = D0.copy()
        except Exception:
            pass

# --------- core metrics from belief and models ---------
def step_metrics(qs, prior_s, A, C, obs_idx):
    """
    Returns:
      complexity, accuracy, extrinsic, epistemic
    """
    qs = normalize(qs)
    prior_s = normalize(prior_s)

    # Variational Free Energy pieces (at current step, given observation)
    # Complexity: KL(q(s) || prior(s))
    complexity = kl_div(qs, prior_s)

    # Accuracy: - E_q(s)[ ln p(o_t | s) ]
    lik_col = A[obs_idx, :]  # shape (S,)
    lik_col = np.clip(lik_col, 1e-16, 1.0)
    accuracy = - float(np.sum(qs * np.log(lik_col)))

    # Expected Free Energy (1-step lookahead approximations)
    # Predictive outcomes from current belief
    q_o = A @ qs  # shape (O=S,)
    q_o = normalize(q_o)

    # Extrinsic: E_{q(o)}[ - ln p(o|C) ]
    pC = softmax(C)                 # convert C “utilities” to probabilities
    extrinsic = - float(np.sum(q_o * np.log(np.clip(pC, 1e-16, 1.0))))

    # Epistemic: E_{q(o)}[ KL(q(s|o)||q(s)) ], with q(s|o) ∝ A[o,:] * q(s)
    epistemic = 0.0
    for o in range(A.shape[0]):
        post_o = normalize(A[o, :] * qs)
        epistemic += q_o[o] * kl_div(post_o, qs)

    return complexity, accuracy, extrinsic, float(epistemic)


# --------- rendering helpers (single figure: grid + bars) ---------
def init_figure(n_rows, n_cols, cell=32):
    fig = plt.figure(figsize=(10, 5))
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_bars = fig.add_subplot(1, 2, 2)

    ax_grid.set_title("GridWorld")
    ax_grid.set_axis_off()

    bars = ax_bars.bar(
        ["Complexity", "Accuracy", "Extrinsic", "Epistemic"],
        [0, 0, 0, 0]
    )
    ax_bars.set_ylim(0, 5)  # auto-adjust later
    ax_bars.set_ylabel("Value")
    ax_bars.set_title("Free Energy Terms (step-by-step)")
    ax_bars.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    plt.ion(); plt.show(block=False)
    return fig, ax_grid, ax_bars, bars

def update_grid(ax_grid, env_img):
    if not hasattr(update_grid, "_im") or update_grid._im is None:
        update_grid._im = ax_grid.imshow(env_img, interpolation="nearest")
    else:
        update_grid._im.set_data(env_img)

def update_bars(ax_bars, bars, values):
    vmax = max(1e-6, float(np.max(values)))
    # give some headroom
    target_ylim = max(1.0, vmax * 1.2)
    ymin, ymax = ax_bars.get_ylim()
    if target_ylim > ymax or target_ylim < 0.5 * ymax:
        ax_bars.set_ylim(0, target_ylim)

    for b, v in zip(bars, values):
        b.set_height(float(v))

# --------- plot episode metrics history ---------
def plot_episode_metrics(history: dict, episode_idx: int, title_suffix: str = ""):
    """Plot temporal evolution of the 4 metrics for one episode."""
    t = np.arange(1, len(history["complexity"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(t, history["complexity"], label="Complexity (KL(q||prior))", linewidth=2)
    plt.plot(t, history["accuracy"],   label="Accuracy (-E_q ln p(o|s))", linewidth=2)
    plt.plot(t, history["extrinsic"],  label="Extrinsic (utility)", linewidth=2)
    plt.plot(t, history["epistemic"],  label="Epistemic (state info gain)", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Per-step Free Energy Terms — Episode {episode_idx}{(' — ' + title_suffix) if title_suffix else ''}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)  # non-blocking so live window can keep running

# --------- run one live episode and plot terms ---------
def run_live_episode_once(env, agent, A, B, C, D, fig, ax_grid, ax_bars, bars,
                          fps=12.0, sophisticated=False, controls=None,
                          mode_label="AIF"):
    prior_s = normalize(D.copy())
    obs, info = env.reset()
    reset_agent_compat(agent)
    total_r, steps, done = 0.0, 0, False

    # --- per-step metric histories ---
    hist_complexity = []
    hist_accuracy   = []
    hist_extrinsic  = []
    hist_epistemic  = []

    while not done:
        infer_states_compat(agent, obs)
        qs = get_qs_compat(agent, expected_S=A.shape[1])
        if qs is None:
            S = A.shape[1]
            qs = np.zeros(S); qs[int(obs)] = 1.0

        complexity, accuracy, extrinsic, epistemic = step_metrics(qs, prior_s, A, C, int(obs))
        # record step
        hist_complexity.append(complexity)
        hist_accuracy.append(accuracy)
        hist_extrinsic.append(extrinsic)
        hist_epistemic.append(epistemic)

        # live visuals
        img = env.render()  # rgb array
        update_grid(ax_grid, img)
        update_bars(ax_bars, bars, [complexity, accuracy, extrinsic, epistemic])
        ax_bars.set_title(f"Free Energy Terms — step {steps+1} | {mode_label}")
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        time.sleep(1.0 / fps)

        # act
        infer_policies_compat(agent, sophisticated, controls)
        action = sample_action_compat(agent)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r; steps += 1
        done = terminated or truncated

        # predictive prior for next step
        prior_s = normalize(B[:, :, action] @ qs)

    # pack histories
    history = {
        "complexity": np.asarray(hist_complexity, dtype=float),
        "accuracy":   np.asarray(hist_accuracy,   dtype=float),
        "extrinsic":  np.asarray(hist_extrinsic,  dtype=float),
        "epistemic":  np.asarray(hist_epistemic,  dtype=float),
    }
    return total_r, steps, history


# --------- make noisy B helper ---------

def make_noisy_B(B_base: np.ndarray, eps: float) -> np.ndarray:
    """
    Column-stochastic noise on B (agent's transition model).
    B has shape (S, S, U) with columns indexed by previous state.
    We mix each (·, s_prev, u) column with a uniform distribution over next-states.

        B_noisy[:, s_prev, u] = (1 - eps) * B_base[:, s_prev, u] + eps * (1/S)

    Then re-normalize each column to sum to 1.
    """
    B_base = np.asarray(B_base, dtype=np.float64)
    assert B_base.ndim == 3, "B must be (S, S, U)"
    S, S2, U = B_base.shape
    assert S == S2, "B must be square over states"
    assert 0.0 <= eps < 1.0, "eps must be in [0,1)"

    Bn = (1.0 - eps) * B_base + eps * (1.0 / S) * np.ones_like(B_base)
    # normalize each column (over s_next) for every (s_prev, u)
    colsum = Bn.sum(axis=0, keepdims=True)  # shape (1, S, U)
    Bn = Bn / np.clip(colsum, 1e-12, None)
    return Bn


def main():
    ap = argparse.ArgumentParser(description="Live Grid + dynamic bars for VFE & EFE terms (step-by-step).")
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

    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--episodes", type=int, default=3, help="number of live episodes to run")
    ap.add_argument("--pause-after-ep", type=float, default=0.4, help="seconds to pause between episodes")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--policy-len", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=16.0)
    ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic", "deterministic"])
    ap.add_argument("--c-reward", type=float, default=3.0)
    ap.add_argument("--c-punish", type=float, default=-3.0)
    ap.add_argument("--sophisticated", action="store_true", help="request sophisticated inference if supported")
    ap.add_argument("--a-noise", type=float, default=0.0, help="ε in A_noisy=(1-ε)I+ε*1/O; 0 disables")
    ap.add_argument("--b-noise", type=float, default=0.0, help="ε for B mixing inside the agent: B'=(1-ε)B + ε*1/S (0 ≤ ε < 1)")
    args = ap.parse_args()

    def parse_pos(s):
        if s.strip().lower() == "random": return None
        r, c = s.split(","); return (int(r), int(c))

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos  = parse_pos(args.start_pos)

    env = GridWorld(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        step_cost=args.step_cost, reward=args.reward, punish=args.punish,
        max_steps=args.max_steps, slip_prob=args.slip_prob,
        render_mode="rgb_array"  # important: we embed the RGB array into our figure
    )
    env.reset(seed=args.seed)

    factory_out = build_gridworld_agent(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        c_reward=args.c_reward, c_punish=args.c_punish,
        policy_len=args.policy_len, gamma=args.gamma,
        action_selection=args.act_sel,
        sophisticated=args.sophisticated,
    )
    if isinstance(factory_out, tuple) and len(factory_out) == 3:
        agent, model, controls = factory_out
    else:
        agent, model = factory_out
        controls = None

    # Keep a pristine copy when you build the agent:
    A, B, C, D = get_models(agent, model)
    D0 = D.copy()

    # For each episode:
    hard_reset_agent(agent, D0)
    prior_s = D0.copy()
    obs, info = env.reset(seed=args.seed)   # Ensure D0 matches your start scheme:
                                  # - one-hot at (start_pos) if fixed
                                  # - uniform over states (or allowed starts) if random

    # A-noise
    if args.a_noise > 0.0:
        S = A.shape[1]
        A = (1.0 - args.a_noise) * np.eye(S) + args.a_noise * (1.0 / S) * np.ones((S, S))
        A = A / np.clip(A.sum(axis=0, keepdims=True), 1e-12, None)
        try: agent.A = A
        except Exception: pass
        model["A"] = A

    # B-noise (agent-only model mismatch)
    if args.b_noise > 0.0:
        B = make_noisy_B(B, args.b_noise)
        try: agent.B = [ B ]
        except Exception: pass
        model["B"] = [ B ]

    # one persistent figure for all episodes
    fig, ax_grid, ax_bars, bars = init_figure(env.n_rows, env.n_cols)

    try:
        totals = []
        for ep in range(1, args.episodes + 1):
            # zero bars at episode start (visual clarity)
            update_bars(ax_bars, bars, [0, 0, 0, 0])
            ax_bars.set_title(f"Free Energy Terms — episode {ep}/{args.episodes}")
            fig.canvas.draw_idle(); fig.canvas.flush_events()

            total_r, steps, hist = run_live_episode_once(
                env, agent, A, B, C, D, fig, ax_grid, ax_bars, bars,
                fps=args.fps, sophisticated=args.sophisticated, controls=controls,
                mode_label=f"AIF (live metrics) — ep {ep}"
            )
            totals.append((total_r, steps))
            print(f"[Episode {ep}] return={total_r:.2f}, steps={steps}")

            # NEW: plot temporal evolution for this episode
            plot_episode_metrics(hist, ep, title_suffix=f"A-noise ε={args.a_noise}, soph={args.sophisticated}")

            time.sleep(args.pause_after_ep)

            factory_out = build_gridworld_agent(
                n_rows=args.rows, n_cols=args.cols,
                reward_pos=reward_pos, punish_pos=punish_pos,
                start_pos=start_pos,
                c_reward=args.c_reward, c_punish=args.c_punish,
                policy_len=args.policy_len, gamma=args.gamma,
                action_selection=args.act_sel,
                sophisticated=args.sophisticated,
            )
            if isinstance(factory_out, tuple) and len(factory_out) == 3:
                agent, model, controls = factory_out
            else:
                agent, model = factory_out
                controls = None

            # Keep a pristine copy when you build the agent:
            A, B, C, D = get_models(agent, model)
            D0 = D.copy()

            # For each episode:
            hard_reset_agent(agent, D0)
            prior_s = D0.copy()
            obs, info = env.reset(seed=args.seed)   # Ensure D0 matches your start scheme:
                                          # - one-hot at (start_pos) if fixed
                                          # - uniform over states (or allowed starts) if random

            # A-noise
            if args.a_noise > 0.0:
                S = A.shape[1]
                A = (1.0 - args.a_noise) * np.eye(S) + args.a_noise * (1.0 / S) * np.ones((S, S))
                A = A / np.clip(A.sum(axis=0, keepdims=True), 1e-12, None)
                try: agent.A = A
                except Exception: pass
                model["A"] = A

            # B-noise (agent-only model mismatch)
            if args.b_noise > 0.0:
                B = make_noisy_B(B, args.b_noise)
                try: agent.B = [ B ]
                except Exception: pass
                model["B"] = [ B ]

        print("All episodes complete:", totals)

        # keep window open
        while plt.fignum_exists(plt.gcf().number):
            plt.pause(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
