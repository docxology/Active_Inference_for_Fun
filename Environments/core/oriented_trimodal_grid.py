"""
Oriented Tri-Modal Grid Environment for Active Inference.

A grid world environment with agent orientation and three observation modalities:
- M1: Distance to first terminal cell along look direction
- M2: Terminal class along look direction (EDGE/RED/GREEN)
- M3: Current cell class (EMPTY/EDGE/RED/GREEN)

The agent can move forward or rotate in place. Empty cells are transparent
for distance calculations. Termination occurs on entering reward/punish cells.

Example:
    >>> from active_inference.environments.core import OrientedTriModalGrid
    >>>
    >>> env = OrientedTriModalGrid(
    ...     n_rows=6, n_cols=6,
    ...     reward_pos=(5, 5), punish_pos=(0, 5),
    ...     start_pos=(0, 0), start_ori="N"
    ... )
    >>>
    >>> obs, info = env.reset()
    >>> # obs is (distance, terminal_class, current_class)
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

# Cell classes
CLASS_EMPTY, CLASS_EDGE, CLASS_RED, CLASS_GREEN = 0, 1, 2, 3

# Terminal class mappings for modality 2
M2_EDGE, M2_RED, M2_GREEN = 0, 1, 2

# Orientations
ORIENTS = ["N", "E", "S", "W"]
ORI2IDX = {o: i for i, o in enumerate(ORIENTS)}

# Actions
FWD, TURN_L, TURN_R = 0, 1, 2


class OrientedTriModalGrid(gym.Env):
    """
    Grid world with agent orientation and tri-modal observations.

    The agent has position and orientation in a grid world. Observations include:
    - Distance to first terminal cell along current look direction
    - Type of terminal cell along look direction (EDGE/RED/GREEN)
    - Type of current cell (EMPTY/EDGE/RED/GREEN)

    Actions are: forward movement, left turn, right turn. Empty cells are
    transparent for distance calculations. Episode terminates on entering
    reward/punish cells or reaching max_steps.

    Args:
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        reward_pos: (row, col) position of reward terminal
        punish_pos: (row, col) position of punish terminal
        start_pos: Starting (row, col) position
        start_ori: Starting orientation ("N", "E", "S", "W")
        step_cost: Reward penalty per step
        reward: Terminal reward for reaching reward_pos
        punish: Terminal reward for reaching punish_pos
        max_steps: Maximum episode length
        render_mode: Rendering mode ("human" or "rgb_array")
        cell_px: Pixel size per cell for rendering

    Attributes:
        action_space: Discrete(3) - actions 0=FWD, 1=TURN_L, 2=TURN_R
        observation_space: Tuple of three Discrete spaces for tri-modal observations
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        reward_pos: Tuple[int, int],
        punish_pos: Tuple[int, int],
        start_pos: Optional[Tuple[int, int]] = (0, 0),
        start_ori: str = "N",
        step_cost: float = 0.0,
        reward: float = 1.0,
        punish: float = -1.0,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        cell_px: int = 28,
    ):
        super().__init__()

        if n_rows < 2 or n_cols < 2:
            raise ValueError("Grid must be at least 2×2 for meaningful dynamics")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.reward_pos = tuple(reward_pos)
        self.punish_pos = tuple(punish_pos)

        # Validate positions are different and within bounds
        if self.reward_pos == self.punish_pos:
            raise ValueError("Reward and punish positions must be different")
        if not (0 <= self.reward_pos[0] < n_rows and 0 <= self.reward_pos[1] < n_cols):
            raise ValueError(f"Reward position {self.reward_pos} out of bounds for {n_rows}×{n_cols} grid")
        if not (0 <= self.punish_pos[0] < n_rows and 0 <= self.punish_pos[1] < n_cols):
            raise ValueError(f"Punish position {self.punish_pos} out of bounds for {n_rows}×{n_cols} grid")

        self.start_pos = None if start_pos is None else tuple(start_pos)
        if self.start_pos and not (0 <= self.start_pos[0] < n_rows and 0 <= self.start_pos[1] < n_cols):
            raise ValueError(f"Start position {self.start_pos} out of bounds for {n_rows}×{n_cols} grid")

        if start_ori.upper() not in ORI2IDX:
            raise ValueError(f"Invalid start orientation '{start_ori}', must be one of {ORIENTS}")

        self.start_ori = start_ori.upper()
        self.step_cost = float(step_cost)
        self.reward_val = float(reward)
        self.punish_val = float(punish)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.cell_px = int(cell_px)

        # Spaces
        max_range = max(n_rows, n_cols) - 1
        self.observation_space = spaces.Tuple((
            spaces.Discrete(max_range + 1, start=0),   # M1: distance to terminal
            spaces.Discrete(3),                        # M2: terminal class (EDGE/RED/GREEN)
            spaces.Discrete(4),                        # M3: current class (EMPTY/EDGE/RED/GREEN)
        ))
        self.action_space = spaces.Discrete(3)

        # State
        self.pos = (0, 0)
        self.ori = ORI2IDX["N"]
        self.steps = 0

        # Rendering cache
        self._fig = None
        self._ax = None
        self._im = None
        self._bg = None

        # Precompute static board background
        self._board_rgb = None
        self._rebuild_board_rgb()

        logger.info(
            f"Created {n_rows}×{n_cols} OrientedTriModalGrid: "
            f"reward={self.reward_pos}, punish={self.punish_pos}, "
            f"start_pos={self.start_pos}, start_ori={self.start_ori}"
        )

    # -------- Gym API --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if self.start_pos is None:
            # Random start anywhere
            r = self.np_random.integers(0, self.n_rows)
            c = self.np_random.integers(0, self.n_cols)
            self.pos = (int(r), int(c))
            logger.debug(f"Reset to random position: {self.pos}")
        else:
            self.pos = tuple(self.start_pos)
            logger.debug(f"Reset to configured position: {self.pos}")

        self.ori = ORI2IDX.get(self.start_ori, 0)
        self.steps = 0
        obs = self._get_obs()
        info = {}

        logger.info(f"Environment reset: pos={self.pos}, ori={ORIENTS[self.ori]}")
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}, must be in {self.action_space}")

        action = int(action)
        self.steps += 1

        old_pos = self.pos
        old_ori = self.ori

        if action == FWD:
            self.pos = self._forward(self.pos, self.ori)
            action_name = "forward"
        elif action == TURN_L:
            self.ori = (self.ori - 1) % 4
            action_name = "turn_left"
        elif action == TURN_R:
            self.ori = (self.ori + 1) % 4
            action_name = "turn_right"
        else:
            raise ValueError(f"Invalid action {action}")

        r = self.step_cost
        terminated = False

        if self.pos == self.reward_pos:
            r += self.reward_val
            terminated = True
            logger.debug(f"Reached reward at {self.pos}, reward: {r}")
        elif self.pos == self.punish_pos:
            r += self.punish_val
            terminated = True
            logger.debug(f"Reached punish at {self.pos}, reward: {r}")

        truncated = self.steps >= self.max_steps
        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        logger.debug(
            f"Step {self.steps}: {old_pos}/{ORIENTS[old_ori]} -> "
            f"{self.pos}/{ORIENTS[self.ori]} ({action_name}), reward: {r}"
        )
        return obs, float(r), bool(terminated), bool(truncated), info

    # -------- Observations --------
    def _get_obs(self):
        dist, term_class = self._raycast_terminal(self.pos, self.ori)
        curr_class = self._class_of(self.pos)

        # Map internal class codes -> M2 indices expected by the agent:
        # M2: {0: EDGE, 1: RED, 2: GREEN}
        m2_map = {CLASS_EDGE: 0, CLASS_RED: 1, CLASS_GREEN: 2}
        term_idx = m2_map.get(int(term_class), 0)  # default to EDGE if ever unknown

        return (int(dist), int(term_idx), int(curr_class))

    # -------- Geometry / helpers --------
    def _forward(self, pos, ori):
        r, c = pos
        if ori == ORI2IDX["N"]:
            r = max(0, r - 1)
        elif ori == ORI2IDX["S"]:
            r = min(self.n_rows - 1, r + 1)
        elif ori == ORI2IDX["E"]:
            c = min(self.n_cols - 1, c + 1)
        else:  # W
            c = max(0, c - 1)
        return (r, c)

    def _class_of(self, pos):
        r, c = pos
        if pos == self.reward_pos:
            return CLASS_GREEN
        if pos == self.punish_pos:
            return CLASS_RED
        if r == 0 or r == self.n_rows - 1 or c == 0 or c == self.n_cols - 1:
            return CLASS_EDGE
        return CLASS_EMPTY

    def _raycast_terminal(self, pos, ori):
        """Return (distance, terminal_class) along look direction.
        If standing on edge and facing outward, distance=0 and class=EDGE.
        Only terminals are EDGE/RED/GREEN; EMPTY cells are transparent.
        """
        r, c = pos
        # outward on edge => 0, EDGE
        if r == 0 and ori == ORI2IDX["N"]:
            return 0, CLASS_EDGE
        if r == self.n_rows - 1 and ori == ORI2IDX["S"]:
            return 0, CLASS_EDGE
        if c == 0 and ori == ORI2IDX["W"]:
            return 0, CLASS_EDGE
        if c == self.n_cols - 1 and ori == ORI2IDX["E"]:
            return 0, CLASS_EDGE

        # marching deltas
        dr, dc = 0, 0
        if ori == ORI2IDX["N"]:   dr, dc = -1, 0
        elif ori == ORI2IDX["S"]: dr, dc = +1, 0
        elif ori == ORI2IDX["E"]: dr, dc = 0, +1
        else:                     dr, dc = 0, -1

        rr, cc = r, c
        dist = 0
        while True:
            rr += dr; cc += dc; dist += 1
            # reward/punish take precedence if seen
            if (rr, cc) == self.reward_pos:
                return dist, CLASS_GREEN
            if (rr, cc) == self.punish_pos:
                return dist, CLASS_RED

            # about to leave grid => EDGE is terminal
            nxt_r, nxt_c = rr + dr, cc + dc
            if nxt_r < 0 or nxt_r >= self.n_rows or nxt_c < 0 or nxt_c >= self.n_cols:
                return dist, CLASS_EDGE
            # else continue (ignore EMPTY)

    # -------- Rendering --------
    def _rebuild_board_rgb(self):
        """Static background: empty/edge/red/green cells (no agent)."""
        H = self.n_rows * self.cell_px
        W = self.n_cols * self.cell_px
        img = np.ones((H, W, 3), dtype=np.uint8) * 240  # light background

        # Colors
        col_empty = np.array([240, 240, 240], dtype=np.uint8)
        col_edge  = np.array([210, 210, 210], dtype=np.uint8)
        col_red   = np.array([220, 60, 60], dtype=np.uint8)
        col_green = np.array([60, 200, 90], dtype=np.uint8)
        col_grid  = np.array([180, 180, 180], dtype=np.uint8)

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                y0 = r * self.cell_px
                x0 = c * self.cell_px
                y1 = y0 + self.cell_px
                x1 = x0 + self.cell_px

                klass = CLASS_EMPTY
                if (r, c) == self.reward_pos:
                    klass = CLASS_GREEN
                elif (r, c) == self.punish_pos:
                    klass = CLASS_RED
                elif r == 0 or r == self.n_rows - 1 or c == 0 or c == self.n_cols - 1:
                    klass = CLASS_EDGE

                if klass == CLASS_EMPTY:
                    img[y0:y1, x0:x1] = col_empty
                elif klass == CLASS_EDGE:
                    img[y0:y1, x0:x1] = col_edge
                elif klass == CLASS_RED:
                    img[y0:y1, x0:x1] = col_red
                else:
                    img[y0:y1, x0:x1] = col_green

                # grid line
                img[y0:y0+1, x0:x1] = col_grid
                img[y0:y1, x0:x0+1] = col_grid

        self._board_rgb = img

    def _draw_agent(self, base_img: np.ndarray) -> np.ndarray:
        """Overlay the agent as a grey square with a small orientation marker."""
        img = base_img.copy()
        r, c = self.pos
        y0 = r * self.cell_px
        x0 = c * self.cell_px
        y1 = y0 + self.cell_px
        x1 = x0 + self.cell_px

        # agent body
        body = np.array([120, 120, 120], dtype=np.uint8)  # grey
        pad = max(2, self.cell_px // 6)
        img[y0+pad:y1-pad, x0+pad:x1-pad] = body

        # orientation marker (a thinner rectangle near the facing edge)
        mark = np.array([30, 30, 30], dtype=np.uint8)
        t = max(2, self.cell_px // 10)
        if self.ori == ORI2IDX["N"]:
            img[y0+pad:y0+pad+t, x0+pad:x1-pad] = mark
        elif self.ori == ORI2IDX["S"]:
            img[y1-pad-t:y1-pad, x0+pad:x1-pad] = mark
        elif self.ori == ORI2IDX["E"]:
            img[y0+pad:y1-pad, x1-pad-t:x1-pad] = mark
        else:  # W
            img[y0+pad:y1-pad, x0+pad:x0+pad+t] = mark

        return img

    def render(self):
        frame = self._draw_agent(self._board_rgb)
        if self.render_mode == "rgb_array":
            return frame

        # human: persistent window via matplotlib
        import matplotlib.pyplot as plt
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._im = self._ax.imshow(frame, interpolation="nearest")
            self._ax.set_axis_off()
            self._fig.canvas.manager.set_window_title("OrientedTriModalGrid")
            plt.ion(); plt.show(block=False)
        else:
            self._im.set_data(frame)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        return None

    def close(self):
        # best-effort matplotlib teardown
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
            self._fig = self._ax = self._im = None
        super().close()
