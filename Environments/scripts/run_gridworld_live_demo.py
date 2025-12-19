# run_gridworld_live_demo.py
import time
import argparse
import numpy as np

from gridworld_env import GridWorld
from ai_agent_factory import build_gridworld_agent


# --- small compat helpers (pymdp versions differ slightly) ---
def infer_states_compat(agent, obs):
    try:
        agent.infer_states(obs)      # some versions accept scalar
    except Exception:
        agent.infer_states([obs])    # others expect list per modality

def sample_action_compat(agent) -> int:
    a = agent.sample_action()
    if isinstance(a, (list, tuple, np.ndarray)):
        return int(a[0])
    return int(a)

def reset_agent_compat(agent):
    if hasattr(agent, "reset") and callable(agent.reset):
        try:
            agent.reset()
        except Exception:
            pass

def set_overlay(env, text: str):
    """Write a small overlay title onto the env's matplotlib figure, if available."""
    try:
        import matplotlib.pyplot as plt
        if getattr(env, "render_mode", None) == "human" and hasattr(env, "_render_ax") and env._render_ax is not None:
            env._render_ax.set_title(text)
            if hasattr(env, "_render_fig") and env._render_fig is not None:
                env._render_fig.canvas.draw_idle()
                env._render_fig.canvas.flush_events()
    except Exception:
        pass


def run_one_episode_random(env: GridWorld, rng: np.random.Generator, fps: float, episode_idx: int):
    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    total_r = 0.0
    steps = 0
    while not done:
        action = int(rng.integers(0, env.action_space.n))
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r; steps += 1
        env.render()  # dynamic update (do NOT print)
        set_overlay(env, f"RANDOM | ep {episode_idx} | step {steps} | R={total_r:.2f}")
        time.sleep(1.0 / fps)
        done = terminated or truncated
    return total_r, steps


def run_one_episode_aif(env: GridWorld, agent, rng: np.random.Generator, fps: float, episode_idx: int):
    reset_agent_compat(agent)
    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    total_r = 0.0
    steps = 0
    while not done:
        infer_states_compat(agent, obs)
        agent.infer_policies()
        action = sample_action_compat(agent)
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r; steps += 1
        env.render()
        set_overlay(env, f"AIF | ep {episode_idx} | step {steps} | R={total_r:.2f}")
        time.sleep(1.0 / fps)
        done = terminated or truncated
    return total_r, steps


def parse_pos(s: str):
    if s.strip().lower() == "random":
        return None
    r, c = s.split(",")
    return (int(r), int(c))


def main():
    ap = argparse.ArgumentParser(description="Live GridWorld demo: random episodes then AIF episodes (single window).")
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
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--episodes-random", type=int, default=3, help="how many random episodes to show")
    ap.add_argument("--episodes-aif", type=int, default=3, help="how many AIF episodes to show")
    ap.add_argument("--fps", type=float, default=12.0, help="animation speed (frames per second)")

    # AIF hyperparams
    ap.add_argument("--policy-len", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=16.0)
    ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic", "deterministic"])
    ap.add_argument("--c-reward", type=float, default=3.0)
    ap.add_argument("--c-punish", type=float, default=-3.0)
    args = ap.parse_args()

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos = parse_pos(args.start_pos)

    rng = np.random.default_rng(args.seed)

    # Build env with dynamic (human) renderer
    env = GridWorld(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        step_cost=args.step_cost, reward=args.reward, punish=args.punish,
        max_steps=args.max_steps, slip_prob=args.slip_prob,
        render_mode="human"
    )

    # Build AIF agent
    agent, _, _ = build_gridworld_agent(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos,
        c_reward=args.c_reward, c_punish=args.c_punish,
        policy_len=args.policy_len, gamma=args.gamma,
        action_selection=args.act_sel,
    )

    try:
        # --- Random episodes ---
        for ep in range(1, args.episodes_random + 1):
            R, steps = run_one_episode_random(env, rng, args.fps, ep)
            print(f"[RANDOM] Episode {ep}: return={R:.2f}, steps={steps}")
            time.sleep(0.4)  # brief pause so you can see terminal state

        # --- AIF episodes ---
        for ep in range(1, args.episodes_aif + 1):
            R, steps = run_one_episode_aif(env, agent, rng, args.fps, ep)
            print(f"[AIF] Episode {ep}: return={R:.2f}, steps={steps}")
            time.sleep(0.4)

        # leave window open until closed by user (optional)
        set_overlay(env, "Demo complete — close the window to exit.")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
