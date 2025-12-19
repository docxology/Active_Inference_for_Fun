# run_nav3_live_demo.py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from oriented_trimodal_grid import OrientedTriModalGrid, ORIENTS
from ai_agent_factory_nav3 import build_trimodal_nav_agent

# -------------- compat helpers ----------------
def infer_states_compat(agent, obs_triplet):
    """Pass 3-modality observation to pymdp agent (handles API variants)."""
    try:
        agent.infer_states(list(obs_triplet))
    except Exception:
        agent.infer_states([obs_triplet[0], obs_triplet[1], obs_triplet[2]])

def sample_action_compat(agent) -> int:
    a = agent.sample_action()
    return int(a[0] if isinstance(a, (list, tuple, np.ndarray)) else a)

def infer_policies_compat(agent, controls, sophisticated: bool):
    if sophisticated and controls and callable(controls.get("infer_policies", None)):
        return controls["infer_policies"]()
    # fallback
    try:
        return agent.infer_policies(mode="sophisticated" if sophisticated else "classic")
    except TypeError:
        try:
            return agent.infer_policies(method="sophisticated" if sophisticated else "classic")
        except TypeError:
            return agent.infer_policies()

# -------------- action decoding ----------------
ACTIONS = ["forward", "turn_left", "turn_right"]

# -------------- live figure ----------------
def init_figure():
    fig = plt.figure(figsize=(10, 6))
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_grid.set_axis_off()
    ax_hud  = fig.add_subplot(1, 2, 2)
    ax_hud.set_axis_off()
    txt = ax_hud.text(
        0.02, 0.98,
        "HUD",
        va="top", ha="left", fontsize=12, family="monospace"
    )
    plt.ion(); plt.show(block=False)
    return fig, ax_grid, ax_hud, txt

def update_grid(ax_grid, frame):
    if not hasattr(update_grid, "_im") or update_grid._im is None:
        update_grid._im = ax_grid.imshow(frame, interpolation="nearest")
    else:
        update_grid._im.set_data(frame)

def update_hud(txt_obj, step, cum_return, orient_idx, obs, last_action):
    m1, m2, m3 = obs
    # map M2 class ids to strings
    m2_map = {0: "EDGE", 1: "RED", 2: "GREEN"}
    m3_map = {0: "EMPTY", 1: "EDGE", 2: "RED", 3: "GREEN"}
    content = (
        f"Step         : {step}\n"
        f"Return (Σr)  : {cum_return:.3f}\n"
        f"Orientation  : {ORIENTS[orient_idx]}\n"
        f"Obs M1 dist  : {m1}\n"
        f"Obs M2 class : {m2_map.get(int(m2), str(m2))}\n"
        f"Obs M3 here  : {m3_map.get(int(m3), str(m3))}\n"
        f"Last action  : {last_action}"
    )
    txt_obj.set_text(content)

# -------------- main episode loop ----------------
def run_live_episode(env, agent, controls, fps: float, sophisticated: bool, hud_txt, ax_grid):
    obs, _ = env.reset()
    total_r, steps = 0.0, 0
    done = False
    last_action_name = "—"
    # Draw first frame
    frame = env.render()  # rgb_array
    update_grid(ax_grid, frame)
    update_hud(hud_txt, steps, total_r, env.ori, obs, last_action_name)
    plt.gcf().canvas.draw_idle(); plt.gcf().canvas.flush_events()

    while not done:
        # 1) infer states from tri-modal obs
        infer_states_compat(agent, obs)
        # 2) plan policies (sophisticated if available)
        infer_policies_compat(agent, controls, sophisticated)
        # 3) sample and step
        action = sample_action_compat(agent)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += r; steps += 1
        done = terminated or truncated
        last_action_name = ACTIONS[action]

        # 4) render + HUD
        frame = env.render()
        update_grid(ax_grid, frame)
        update_hud(hud_txt, steps, total_r, env.ori, obs, last_action_name)
        plt.gcf().canvas.draw_idle(); plt.gcf().canvas.flush_events()
        time.sleep(max(1e-3, 1.0 / fps))

    return total_r, steps

# -------------- CLI ----------------
def parse_pos(s: str):
    s = s.strip().lower()
    if s == "random":
        return None
    r, c = s.split(",")
    return (int(r), int(c))

def main():
    ap = argparse.ArgumentParser(description="Live demo for the tri-modal oriented agent.")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--reward-pos", type=str, default="4,6")
    ap.add_argument("--punish-pos", type=str, default="0,6")
    ap.add_argument("--start-pos", type=str, default="0,0")
    ap.add_argument("--start-ori", type=str, default="E", choices=["N","E","S","W"])
    ap.add_argument("--step-cost", type=float, default=0.0)
    ap.add_argument("--reward", type=float, default=1.0)
    ap.add_argument("--punish", type=float, default=-1.0)
    ap.add_argument("--max-steps", type=int, default=200)

    # agent settings
    ap.add_argument("--policy-len", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=16.0)
    ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic","deterministic"])
    ap.add_argument("--c-green", type=float, default=3.0)
    ap.add_argument("--c-red", type=float, default=-3.0)
    ap.add_argument("--sophisticated", action="store_true")

    # model noise (inside agent)
    ap.add_argument("--a-noise", type=float, default=0.0, help="obs noise for A in factory (0..1)")
    ap.add_argument("--b-noise", type=float, default=0.0, help="dynamics model noise for B in factory (0..1)")

    # display
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reward_pos = parse_pos(args.reward_pos)
    punish_pos = parse_pos(args.punish_pos)
    start_pos  = parse_pos(args.start_pos)

    # --- Environment (rgb_array for fast blitting into our figure) ---
    env = OrientedTriModalGrid(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos, start_ori=args.start_ori,
        step_cost=args.step_cost, reward=args.reward, punish=args.punish,
        max_steps=args.max_steps, render_mode="rgb_array"
    )
    env.reset(seed=args.seed)

    # --- Agent (build with noise in-factory) ---
    agent, model, controls = build_trimodal_nav_agent(
        n_rows=args.rows, n_cols=args.cols,
        reward_pos=reward_pos, punish_pos=punish_pos,
        start_pos=start_pos, start_ori=args.start_ori,
        a_obs_noise=args.a_noise, b_model_noise=args.b_noise,
        policy_len=args.policy_len, gamma=args.gamma,
        action_selection=args.act_sel,
        c_green=args.c_green, c_red=args.c_red,
        sophisticated=args.sophisticated,
    )

    # --- Figure ---
    fig, ax_grid, ax_hud, hud_txt = init_figure()

    try:
        for ep in range(1, args.episodes + 1):
            # reset agent beliefs between episodes if available
            if hasattr(agent, "reset") and callable(agent.reset):
                try: agent.reset()
                except Exception: pass

            total_r, steps = run_live_episode(
                env, agent, controls, args.fps, args.sophisticated, hud_txt, ax_grid
            )
            print(f"[Episode {ep}] return={total_r:.2f}, steps={steps}")
            time.sleep(0.4)
    finally:
        # keep window open until user closes it
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
        env.close()

if __name__ == "__main__":
    main()


































































# import argparse
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, RegularPolygon
#
# from ai_agent_factory_nav3 import (
#     build_trimodal_nav_agent,
#     ORIENTS, ORI2IDX,
#     CLASS_EMPTY, CLASS_EDGE, CLASS_RED, CLASS_GREEN,
# )
#
# # ---------- world helpers (mirror the factory's geometry) ----------
# def step_forward(r, c, ori, n_rows, n_cols):
#     if ori == ORI2IDX["N"]:
#         return max(0, r - 1), c
#     if ori == ORI2IDX["S"]:
#         return min(n_rows - 1, r + 1), c
#     if ori == ORI2IDX["E"]:
#         return r, min(n_cols - 1, c + 1)
#     return r, max(0, c - 1)  # W
#
# def class_of_cell(r, c, reward_pos, punish_pos, n_rows, n_cols):
#     if (r, c) == reward_pos: return CLASS_GREEN
#     if (r, c) == punish_pos: return CLASS_RED
#     if r == 0 or r == n_rows-1 or c == 0 or c == n_cols-1: return CLASS_EDGE
#     return CLASS_EMPTY
#
# def raycast_terminal(r, c, ori, reward_pos, punish_pos, n_rows, n_cols):
#     """Return (distance, terminal_class) along look direction.
#        Terminals are EDGE, RED, GREEN. EMPTY squares are skipped."""
#     if r == 0 and ori == ORI2IDX["N"]:   return 0, CLASS_EDGE
#     if r == n_rows-1 and ori == ORI2IDX["S"]: return 0, CLASS_EDGE
#     if c == 0 and ori == ORI2IDX["W"]:   return 0, CLASS_EDGE
#     if c == n_cols-1 and ori == ORI2IDX["E"]: return 0, CLASS_EDGE
#
#     dr, dc = 0, 0
#     if ori == ORI2IDX["N"]: dr, dc = -1, 0
#     elif ori == ORI2IDX["S"]: dr, dc =  1, 0
#     elif ori == ORI2IDX["E"]: dr, dc =  0, 1
#     else: dr, dc = 0, -1
#
#     rr, cc, dist = r, c, 0
#     while True:
#         rr += dr; cc += dc; dist += 1
#         if rr < 0 or rr >= n_rows or cc < 0 or cc >= n_cols:
#             return dist - 1, CLASS_EDGE  # safety
#         if (rr, cc) == reward_pos: return dist, CLASS_GREEN
#         if (rr, cc) == punish_pos: return dist, CLASS_RED
#         nxt_r, nxt_c = rr + dr, cc + dc
#         if nxt_r < 0 or nxt_r >= n_rows or nxt_c < 0 or nxt_c >= n_cols:
#             return dist, CLASS_EDGE
#         # continue over EMPTY
#
# # map terminal class to M2 indices (EDGE, RED, GREEN) = (0,1,2)
# def terminal_to_m2_idx(term_class):
#     if term_class == CLASS_EDGE: return 0
#     if term_class == CLASS_RED:  return 1
#     return 2  # GREEN
#
# # ---------- plotting ----------
# def init_fig(n_rows, n_cols):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_aspect('equal')
#     ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
#     ax.invert_yaxis()
#     ax.set_xticks(range(n_cols+1)); ax.set_yticks(range(n_rows+1))
#     ax.grid(True, which='both', alpha=0.25)
#     ax.set_title("nav3 live demo — pos×ori agent")
#     return fig, ax
#
# # def draw_grid(ax, n_rows, n_cols, reward_pos, punish_pos):
# #     # clear existing rectangles except the agent patch (which we track separately)
# #     ax.patches.clear()
# #     # empty cells background
# #     for r in range(n_rows):
# #         for c in range(n_cols):
# #             face = (0.95, 0.95, 0.95)
# #             edge = (0.8, 0.8, 0.8)
# #             # edges slightly darker
# #             if r == 0 or r == n_rows-1 or c == 0 or c == n_cols-1:
# #                 face = (0.90, 0.90, 0.90)
# #             rect = Rectangle((c, r), 1, 1, facecolor=face, edgecolor=edge, linewidth=1.0)
# #             ax.add_patch(rect)
# #     # reward
# #     rr, rc = reward_pos
# #     ax.add_patch(Rectangle((rc, rr), 1, 1, facecolor=(0.6, 0.93, 0.6), edgecolor='g', linewidth=2.0))
# #     # punish
# #     pr, pc = punish_pos
# #     ax.add_patch(Rectangle((pc, pr), 1, 1, facecolor=(0.98, 0.7, 0.7), edgecolor='r', linewidth=2.0))
#
#
#
# def draw_grid(ax, n_rows, n_cols, reward_pos, punish_pos):
#     """Redraw cell backgrounds + special squares, preserving gridlines.
#     Works on Matplotlib versions where ax.patches has no .clear()."""
#     # Remove all existing patch artists except the agent patch (tracked by place_agent)
#     agent_patch = getattr(place_agent, "_patch", None)
#     for p in list(ax.patches):
#         if p is agent_patch:
#             continue
#         try:
#             p.remove()
#         except Exception:
#             pass
#
#     # empty cells background
#     for r in range(n_rows):
#         for c in range(n_cols):
#             face = (0.95, 0.95, 0.95)
#             edge = (0.8, 0.8, 0.8)
#             if r == 0 or r == n_rows-1 or c == 0 or c == n_cols-1:
#                 face = (0.90, 0.90, 0.90)
#             rect = Rectangle((c, r), 1, 1, facecolor=face, edgecolor=edge, linewidth=1.0)
#             ax.add_patch(rect)
#
#     # reward (green)
#     rr, rc = reward_pos
#     ax.add_patch(Rectangle((rc, rr), 1, 1, facecolor=(0.6, 0.93, 0.6), edgecolor='g', linewidth=2.0))
#
#     # punish (red)
#     pr, pc = punish_pos
#     ax.add_patch(Rectangle((pc, pr), 1, 1, facecolor=(0.98, 0.7, 0.7), edgecolor='r', linewidth=2.0))
#
# def place_agent(ax, r, c, ori, size=0.45):
#     # Remove prior agent if any
#     if hasattr(place_agent, "_patch") and place_agent._patch in ax.patches:
#         try: ax.patches.remove(place_agent._patch)
#         except Exception: pass
#     # Triangle centered in the cell
#     cx, cy = c + 0.5, r + 0.5
#     # RegularPolygon triangle with orientation in radians:
#     # Matplotlib rotation is in radians; pointing up means 90deg (pi/2).
#     # We map N/E/S/W to angles: N=pi/2, E=0, S=3pi/2, W=pi
#     ang = { "N": np.pi/2, "E": 0.0, "S": 3*np.pi/2, "W": np.pi }[ORIENTS[ori]]
#     tri = RegularPolygon((cx, cy), numVertices=3, radius=size, orientation=ang,
#                          facecolor=(0.5, 0.5, 0.5), edgecolor='k', linewidth=1.5)
#     ax.add_patch(tri)
#     place_agent._patch = tri
#
# def annotate(ax, text):
#     if hasattr(annotate, "_txt") and annotate._txt in ax.texts:
#         try: annotate._txt.set_text(text); return
#         except Exception: pass
#     annotate._txt = ax.text(0.02, 0.98, text, va='top', ha='left',
#                             transform=ax.transAxes, fontsize=10,
#                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8))
#
# # ---------- main loop ----------
# def main():
#     ap = argparse.ArgumentParser(description="Live demo for tri-modal nav agent (pos×ori factor, 3 actions).")
#     ap.add_argument("--rows", type=int, default=5)
#     ap.add_argument("--cols", type=int, default=7)
#     ap.add_argument("--reward-pos", type=str, default="4,6")
#     ap.add_argument("--punish-pos", type=str, default="0,6")
#     ap.add_argument("--start-pos", type=str, default="0,0")
#     ap.add_argument("--start-ori", type=str, default="E", choices=ORIENTS)
#     ap.add_argument("--max-steps", type=int, default=200)
#     ap.add_argument("--fps", type=float, default=10.0)
#     ap.add_argument("--seed", type=int, default=0)
#
#     # agent prefs / planning
#     ap.add_argument("--policy-len", type=int, default=6)
#     ap.add_argument("--gamma", type=float, default=16.0)
#     ap.add_argument("--act-sel", type=str, default="stochastic", choices=["stochastic","deterministic"])
#     ap.add_argument("--c-green", type=float, default=3.0)
#     ap.add_argument("--c-red", type=float, default=-3.0)
#     ap.add_argument("--sophisticated", action="store_true")
#
#     # agent model noise (A and B) – world stays noise-free here
#     ap.add_argument("--a-noise", type=float, default=0.0, help="ε in A'=(1-ε)I+ε*1/O per modality (factory applies)")
#     ap.add_argument("--b-noise", type=float, default=0.0, help="ε in B'=(1-ε)B+ε*1/S inside the agent")
#     args = ap.parse_args()
#
#     def parse_pos(s):
#         r, c = s.split(","); return (int(r), int(c))
#     reward_pos = parse_pos(args.reward_pos)
#     punish_pos = parse_pos(args.punish_pos)
#     start_pos  = parse_pos(args.start_pos)
#
#     rng = np.random.default_rng(args.seed)
#
#     agent, model, controls = build_trimodal_nav_agent(
#         n_rows=args.rows, n_cols=args.cols,
#         reward_pos=reward_pos, punish_pos=punish_pos,
#         start_pos=start_pos, start_ori=args.start_ori,
#         a_obs_noise=args.a_noise,
#         b_model_noise=args.b_noise,
#         policy_len=args.policy_len, gamma=args.gamma,
#         action_selection=args.act_sel,
#         c_green=args.c_green, c_red=args.c_red,
#         sophisticated=args.sophisticated
#     )
#
#     # World state (true)
#     r, c = start_pos
#     ori = ORI2IDX[args.start_ori]
#
#     # Render setup
#     fig, ax = init_fig(args.rows, args.cols)
#     draw_grid(ax, args.rows, args.cols, reward_pos, punish_pos)
#     place_agent(ax, r, c, ori)
#     plt.ion(); plt.show(block=False)
#
#     # Loop
#     total_r = 0.0
#     for step in range(1, args.max_steps + 1):
#         # Build true observations (M1 distance, M2 terminal class, M3 current class)
#         dist, terminal = raycast_terminal(r, c, ori, reward_pos, punish_pos, args.rows, args.cols)
#         m1 = dist
#         m2 = terminal_to_m2_idx(terminal)
#         m3 = class_of_cell(r, c, reward_pos, punish_pos, args.rows, args.cols)
#
#         # Update beliefs
#         try:
#             agent.infer_states([m1, m2, m3])  # multi-modality list
#         except Exception:
#             # Some pymdp variants accept scalar per modality too; but list is standard
#             agent.infer_states([m1, m2, m3])
#
#         # Choose action (with sophisticated inference if available)
#         try:
#             if args.sophisticated:
#                 controls["infer_policies"]()
#             else:
#                 agent.infer_policies()
#         except Exception:
#             agent.infer_policies()
#         a = agent.sample_action()
#         a = int(a[0] if isinstance(a, (list, tuple, np.ndarray)) else a)  # 0=fwd, 1=left, 2=right
#
#         # Apply to world (deterministic)
#         if a == 0:  # forward
#             r, c = step_forward(r, c, ori, args.rows, args.cols)
#         elif a == 1:  # left
#             ori = (ori - 1) % 4
#         else:  # right
#             ori = (ori + 1) % 4
#
#         # Reward structure: +1 on GREEN, -1 on RED, else 0
#         cls = class_of_cell(r, c, reward_pos, punish_pos, args.rows, args.cols)
#         rew = 1.0 if cls == CLASS_GREEN else (-1.0 if cls == CLASS_RED else 0.0)
#         total_r += rew
#
#         # Render update
#         draw_grid(ax, args.rows, args.cols, reward_pos, punish_pos)
#         place_agent(ax, r, c, ori)
#         annotate(ax, f"step {step}\nori: {ORIENTS[ori]}\nobs: (dist={m1}, term={['EDGE','RED','GREEN'][m2]}, here={['EMPTY','EDGE','RED','GREEN'][m3]})\nreturn: {total_r:.1f}")
#         fig.canvas.draw_idle(); fig.canvas.flush_events()
#         time.sleep(max(0.0, 1.0 / args.fps))
#
#         # Termination on GREEN/RED
#         if cls in (CLASS_GREEN, CLASS_RED):
#             break
#
#     print(f"Episode finished in {step} steps; total return={total_r:.2f}")
#     # Keep window open
#     while plt.fignum_exists(fig.number):
#         plt.pause(0.1)
#
# if __name__ == "__main__":
#     main()
