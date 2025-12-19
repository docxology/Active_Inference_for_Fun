"""
GridWorld Environment for Active Inference.

A simple N×M grid world environment with discrete actions and observations.
The environment contains two terminal cells: one rewarding and one punishing.
All other cells are neutral with optional step costs.

Actions (discrete):
    0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT

Observations:
    Single discrete index [0, N×M-1] representing the agent's cell position.

Terminal states:
    - reward_pos: terminates with positive reward
    - punish_pos: terminates with negative reward

Features:
    - Deterministic dynamics by default
    - Optional action stochasticity (slip_prob)
    - Optional per-step costs
    - Multiple rendering modes (text, graphical, RGB array)

Example:
    >>> import gymnasium as gym
    >>> from active_inference.environments.core import GridWorld
    >>>
    >>> env = GridWorld(
    ...     n_rows=4, n_cols=5,
    ...     reward_pos=(3, 4), punish_pos=(0, 4),
    ...     start_pos=(0, 0), step_cost=0.0,
    ...     reward=1.0, punish=-1.0, max_steps=100
    ... )
    >>>
    >>> obs, info = env.reset(seed=42)
    >>> done = False
    >>> total = 0.0
    >>> while not done:
    ...     action = env.action_space.sample()
    ...     obs, r, terminated, truncated, info = env.step(action)
    ...     total += r
    ...     done = terminated or truncated
    >>> print(f"Return: {total}")
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

Action = int
Pos = Tuple[int, int]


class GridWorld(gym.Env):
    """
    Grid world environment with rewarding and punishing terminal states.

    The agent navigates a grid using discrete actions. Reaching the reward
    position terminates the episode with positive reward. Reaching the punish
    position terminates with negative reward. All other cells are neutral.

    Args:
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        reward_pos: (row, col) position of rewarding terminal cell
        punish_pos: (row, col) position of punishing terminal cell
        start_pos: Starting (row, col) position. If None, random non-terminal cell at reset
        step_cost: Reward penalty per step (negative values encourage shorter paths)
        reward: Terminal reward for reaching reward_pos
        punish: Terminal reward for reaching punish_pos (typically negative)
        max_steps: Maximum episode length. If None, defaults to 5×(N×M)
        slip_prob: Probability of random action instead of chosen action
        render_mode: Rendering mode - "ansi" for text, "human" for GUI, "rgb_array" for image

    Attributes:
        action_space: Discrete(4) - actions 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        observation_space: Discrete(N×M) - cell indices from 0 to N×M-1
    """

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        reward_pos: Pos = (4, 4),
        punish_pos: Pos = (0, 4),
        start_pos: Optional[Pos] = (0, 0),
        step_cost: float = 0.0,
        reward: float = 1.0,
        punish: float = -1.0,
        max_steps: Optional[int] = None,
        slip_prob: float = 0.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if n_rows < 2 or n_cols < 2:
            raise ValueError("Grid must be at least 2×2 for meaningful dynamics")

        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.reward_pos = self._validate_pos(reward_pos)
        self.punish_pos = self._validate_pos(punish_pos)

        # Validate positions are different
        if self.reward_pos == self.punish_pos:
            raise ValueError("Reward and punish positions must be different")

        self.start_pos_cfg = None if start_pos is None else self._validate_pos(start_pos)
        self.step_cost = float(step_cost)
        self.r_reward = float(reward)
        self.r_punish = float(punish)
        self.max_steps = max_steps if max_steps is not None else 5 * (self.n_rows * self.n_cols)
        self.slip_prob = float(slip_prob)
        self.render_mode = render_mode

        # Spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)

        # State
        self.pos: Pos = (0, 0)
        self._steps = 0
        self._rng = np.random.default_rng()

        # Precompute for speed
        self._reward_idx = self._pos_to_idx(self.reward_pos)
        self._punish_idx = self._pos_to_idx(self.punish_pos)

        logger.info(
            f"Created {n_rows}×{n_cols} GridWorld: "
            f"reward={self.reward_pos}, punish={self.punish_pos}, "
            f"start={self.start_pos_cfg}, slip_prob={slip_prob}"
        )

    # ------------------------- Gymnasium core API ------------------------- #

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick start position
        if self.start_pos_cfg is not None:
            self.pos = self.start_pos_cfg
            logger.debug(f"Reset to configured start position: {self.pos}")
        else:
            # Uniform over non-terminal positions
            all_idxs = [i for i in range(self.n_rows * self.n_cols) if i not in (self._reward_idx, self._punish_idx)]
            start_idx = self._rng.choice(all_idxs)
            self.pos = self._idx_to_pos(int(start_idx))
            logger.debug(f"Reset to random start position: {self.pos}")

        self._steps = 0
        obs = self._pos_to_idx(self.pos)
        info = {"pos": self.pos}

        logger.info(f"Environment reset to position {self.pos}")
        return obs, info

    def step(self, action: Action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}, must be in {self.action_space}")

        self._steps += 1
        original_action = action

        # Slip/stochastic action override
        if self.slip_prob > 0.0 and self._rng.random() < self.slip_prob:
            action = int(self._rng.integers(0, 4))
            logger.debug(f"Action slipped from {original_action} to {action}")

        # Execute action
        r, c = self.pos
        if action == 0:  # UP
            r = max(0, r - 1)
        elif action == 1:  # RIGHT
            c = min(self.n_cols - 1, c + 1)
        elif action == 2:  # DOWN
            r = min(self.n_rows - 1, r + 1)
        elif action == 3:  # LEFT
            c = max(0, c - 1)

        old_pos = self.pos
        self.pos = (r, c)
        idx = self._pos_to_idx(self.pos)

        # Base step reward
        reward = self.step_cost
        terminated = False

        # Check terminal conditions
        if idx == self._reward_idx:
            reward += self.r_reward
            terminated = True
            logger.debug(f"Reached reward at {self.pos}, reward: {reward}")
        elif idx == self._punish_idx:
            reward += self.r_punish
            terminated = True
            logger.debug(f"Reached punish at {self.pos}, reward: {reward}")

        truncated = self._steps >= self.max_steps
        info = {"pos": self.pos}

        logger.debug(f"Step {self._steps}: {old_pos} -> {self.pos} (action {original_action}), reward: {reward}")
        return idx, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """
        Dynamic renderer:
          - 'human'     -> single persistent window; updates in-place (returns None)
          - 'rgb_array' -> returns HxWx3 uint8 image (no window)
          - otherwise   -> ANSI string board
        """
        import numpy as np

        cell = 32
        H, W = self.n_rows * cell, self.n_cols * cell

        # Build image
        img = np.full((self.n_rows, self.n_cols, 3), 255, dtype=np.uint8)
        rr, rc = self.reward_pos
        pr, pc = self.punish_pos
        ar, ac = self.pos
        img[rr, rc] = (0, 200, 0)      # green
        img[pr, pc] = (200, 0, 0)      # red
        img[ar, ac] = (128, 128, 128)  # gray

        # Upscale + grid lines
        img = np.kron(img, np.ones((cell, cell, 1), dtype=np.uint8))
        img[::cell, :, :] = 0
        img[:, ::cell, :] = 0

        if self.render_mode == "rgb_array":
            return img

        elif self.render_mode == "human":
            try:
                import matplotlib
                import matplotlib.pyplot as plt
            except Exception:
                return img  # fallback if matplotlib is unavailable

            # Init (once) or recreate if window was closed
            need_init = (
                not hasattr(self, "_render_fig") or
                self._render_fig is None or
                not plt.fignum_exists(self._render_fig.number)
            )
            if need_init:
                plt.ion()
                self._render_fig, self._render_ax = plt.subplots()
                self._render_ax.set_axis_off()
                self._render_im = self._render_ax.imshow(img, interpolation="nearest", animated=True)
                self._render_fig.canvas.manager.set_window_title("GridWorld")
                self._render_fig.tight_layout()
                self._render_fig.canvas.draw()
                # Cache background for blitting
                self._bg = self._render_fig.canvas.copy_from_bbox(self._render_ax.bbox)
                plt.show(block=False)
                return None

            # Fast in-place update with blitting
            self._render_im.set_data(img)
            canvas = self._render_fig.canvas
            ax = self._render_ax

            # If figure was resized, refresh background
            if not hasattr(self, "_bg") or self._bg is None:
                canvas.draw()
                self._bg = canvas.copy_from_bbox(ax.bbox)

            try:
                canvas.restore_region(self._bg)
            except Exception:
                # If restore fails (e.g., window resize), redraw background
                canvas.draw()
                self._bg = canvas.copy_from_bbox(ax.bbox)
                canvas.restore_region(self._bg)

            ax.draw_artist(self._render_im)
            canvas.blit(ax.bbox)
            canvas.flush_events()
            return None

        # ANSI fallback
        rows = []
        for r in range(self.n_rows):
            line = []
            for c in range(self.n_cols):
                if (r, c) == self.pos:          line.append("A")
                elif (r, c) == self.reward_pos: line.append("G")
                elif (r, c) == self.punish_pos: line.append("X")
                else:                            line.append(".")
            rows.append(" ".join(line))
        return "\n".join(rows)







    def close(self):
        pass

    # ----------------------------- Helpers ------------------------------ #

    def _validate_pos(self, pos: Pos) -> Pos:
        r, c = int(pos[0]), int(pos[1])
        assert 0 <= r < self.n_rows and 0 <= c < self.n_cols, (
            f"Position {pos} is out of bounds for grid {self.n_rows}x{self.n_cols}"
        )
        return (r, c)

    def _pos_to_idx(self, pos: Pos) -> int:
        r, c = pos
        return r * self.n_cols + c

    def _idx_to_pos(self, idx: int) -> Pos:
        return (idx // self.n_cols, idx % self.n_cols)


# Optional: lightweight registration helper for gymnasium.make
try:
    from gymnasium.envs.registration import register

    register(
        id="GridWorld-AIF-v0",
        entry_point="gridworld_env:GridWorld",
        kwargs={},
        max_episode_steps=None,
    )
except Exception:
    # Safe to ignore if registration is called multiple times or in notebooks
    pass

