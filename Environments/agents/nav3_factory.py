"""
Active Inference agent factory for oriented tri-modal grid environments.

This module provides functions to construct Active Inference agents
for environments with position, orientation, and multi-modal observations.
"""

import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np

try:
    from pymdp.agent import Agent
except ImportError:
    raise ImportError("pymdp package required for Active Inference agents")

logger = logging.getLogger(__name__)

# Orientation constants
ORIENTS = ["N", "E", "S", "W"]
ORI2IDX = {o: i for i, o in enumerate(ORIENTS)}

# Cell class constants
CLASS_EMPTY, CLASS_EDGE, CLASS_RED, CLASS_GREEN = 0, 1, 2, 3

# Terminal class mappings for modality 2
M2_EDGE, M2_RED, M2_GREEN = 0, 1, 2

# Action constants
FWD = 0
TURN_L = 1
TURN_R = 2


def _rc_to_lin(r:int, c:int, n_cols:int) -> int:
    return r * n_cols + c

def _lin_to_rc(idx:int, n_cols:int) -> Tuple[int,int]:
    return idx // n_cols, idx % n_cols

def _state_index(r:int, c:int, ori:int, n_rows:int, n_cols:int) -> int:
    return ( _rc_to_lin(r,c,n_cols) * 4 ) + ori

def _from_state_index(s:int, n_rows:int, n_cols:int) -> Tuple[int,int,int]:
    pos_lin, ori = divmod(s, 4)
    r, c = _lin_to_rc(pos_lin, n_cols)
    return r, c, ori

def _turn_left(ori:int) -> int:
    return (ori - 1) % 4

def _turn_right(ori:int) -> int:
    return (ori + 1) % 4

def _step_forward(r:int, c:int, ori:int, n_rows:int, n_cols:int) -> Tuple[int,int]:
    if ori == ORI2IDX["N"]:
        nr, nc = max(0, r-1), c
    elif ori == ORI2IDX["S"]:
        nr, nc = min(n_rows-1, r+1), c
    elif ori == ORI2IDX["E"]:
        nr, nc = r, min(n_cols-1, c+1)
    else:  # W
        nr, nc = r, max(0, c-1)
    return nr, nc

def _class_of_cell(r:int, c:int, reward_pos:Tuple[int,int], punish_pos:Tuple[int,int],
                   n_rows:int, n_cols:int) -> int:
    if (r, c) == reward_pos:
        return CLASS_GREEN
    if (r, c) == punish_pos:
        return CLASS_RED
    # edge?
    if r == 0 or r == n_rows-1 or c == 0 or c == n_cols-1:
        return CLASS_EDGE
    return CLASS_EMPTY

def _raycast_terminal(r:int, c:int, ori:int, reward_pos, punish_pos, n_rows, n_cols) -> Tuple[int,int]:
    """
    March in 'ori' from (r,c) until you hit first non-empty terminal in {EDGE, RED, GREEN}.
    Return (distance, terminal_class). Distance is number of steps to that terminal,
    0 means terminal is current cell's boundary (standing at edge and facing outward).
    """
    # Step until either we hit reward/punish or the boundary (edge).
    dr, dc = 0, 0
    if ori == ORI2IDX["N"]:
        dr, dc = -1, 0
    elif ori == ORI2IDX["S"]:
        dr, dc = +1, 0
    elif ori == ORI2IDX["E"]:
        dr, dc = 0, +1
    else:
        dr, dc = 0, -1

    rr, cc = r, c
    dist = 0

    # If facing outward on an edge square, distance=0 and terminal is EDGE.
    if r == 0 and ori == ORI2IDX["N"]:   return 0, CLASS_EDGE
    if r == n_rows-1 and ori == ORI2IDX["S"]: return 0, CLASS_EDGE
    if c == 0 and ori == ORI2IDX["W"]:   return 0, CLASS_EDGE
    if c == n_cols-1 and ori == ORI2IDX["E"]: return 0, CLASS_EDGE

    while True:
        rr += dr; cc += dc; dist += 1
        # outside => edge encountered just before leaving grid
        if rr < 0 or rr >= n_rows or cc < 0 or cc >= n_cols:
            # This branch should not trigger due to pre-check; keep for safety
            return dist-1, CLASS_EDGE

        if (rr, cc) == reward_pos:
            return dist, CLASS_GREEN
        if (rr, cc) == punish_pos:
            return dist, CLASS_RED

        # If next step would go out, the terminal is the boundary ahead (EDGE)
        nxt_r, nxt_c = rr + dr, cc + dc
        if nxt_r < 0 or nxt_r >= n_rows or nxt_c < 0 or nxt_c >= n_cols:
            return dist, CLASS_EDGE
        # else continue (EMPTY squares are ignored by spec)


def _build_A(n_rows:int, n_cols:int,
             reward_pos:Tuple[int,int], punish_pos:Tuple[int,int],
             a_obs_noise:float=0.0) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Returns A as tuple (A1, A2, A3) for modalities:
      M1: distance ∈ {0..max_range}
      M2: terminal class in look direction ∈ {EDGE, RED, GREEN}
      M3: current cell class ∈ {EMPTY, EDGE, RED, GREEN}
    """
    S = n_rows * n_cols * 4
    max_range = max(n_rows, n_cols) - 1
    O1 = max_range + 1
    O2 = 3
    O3 = 4

    A1 = np.zeros((O1, S), dtype=np.float64)
    A2 = np.zeros((O2, S), dtype=np.float64)
    A3 = np.zeros((O3, S), dtype=np.float64)

    for s in range(S):
        r, c, ori = _from_state_index(s, n_rows, n_cols)
        dist, terminal = _raycast_terminal(r, c, ori, reward_pos, punish_pos, n_rows, n_cols)
        A1[dist, s] = 1.0
        # M2 maps {EDGE,RED,GREEN} -> indices {0,1,2}
        if terminal == CLASS_EDGE:   A2[M2_EDGE,  s] = 1.0
        elif terminal == CLASS_RED:  A2[M2_RED,   s] = 1.0
        else:                        A2[M2_GREEN, s] = 1.0

        curr_class = _class_of_cell(r, c, reward_pos, punish_pos, n_rows, n_cols)
        A3[curr_class, s] = 1.0

    if a_obs_noise > 0.0:
        # Mix with uniform noise and renormalize columns
        def _noisify(A):
            O, S = A.shape
            An = (1.0 - a_obs_noise) * A + a_obs_noise * (1.0 / O) * np.ones_like(A)
            An /= np.clip(An.sum(axis=0, keepdims=True), 1e-12, None)
            return An
        A1, A2, A3 = _noisify(A1), _noisify(A2), _noisify(A3)

    return A1, A2, A3


def _build_B(n_rows:int, n_cols:int, b_model_noise:float=0.0) -> np.ndarray:
    """
    Single factor with U=3 actions. B shape: (S, S, U).
    Deterministic transitions for forward/turns, then optional model noise.
    """
    S = n_rows * n_cols * 4
    U = 3
    B = np.zeros((S, S, U), dtype=np.float64)

    for s in range(S):
        r, c, ori = _from_state_index(s, n_rows, n_cols)

        # forward
        nr, nc = _step_forward(r, c, ori, n_rows, n_cols)
        fwd_s = _state_index(nr, nc, ori, n_rows, n_cols)

        # turns
        l_s = _state_index(r, c, _turn_left(ori),  n_rows, n_cols)
        r_s = _state_index(r, c, _turn_right(ori), n_rows, n_cols)

        B[fwd_s, s, FWD]   = 1.0
        B[l_s,   s, TURN_L] = 1.0
        B[r_s,   s, TURN_R] = 1.0

    if b_model_noise > 0.0:
        S = B.shape[0]
        B = (1.0 - b_model_noise) * B + b_model_noise * (1.0 / S) * np.ones_like(B)
        # normalize columns per action
        colsum = B.sum(axis=0, keepdims=True)
        B = B / np.clip(colsum, 1e-12, None)

    return B


def _build_C(n_rows:int, n_cols:int,
             pref_green:float=3.0, pref_red:float=-3.0) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Preferences over outcomes (as log-probs/energies). We put preferences on:
      - M3 (current class): GREEN high, RED low, others ~0
      - M1, M2 kept neutral (0), but you can shape curiosity by adding bonuses.
    """
    max_range = max(n_rows, n_cols) - 1
    O1 = max_range + 1
    O2 = 3
    O3 = 4

    C1 = np.zeros((O1,), dtype=np.float64)  # neutral distances
    C2 = np.zeros((O2,), dtype=np.float64)  # neutral terminal class
    C3 = np.zeros((O3,), dtype=np.float64)
    C3[CLASS_GREEN] = pref_green
    C3[CLASS_RED]   = pref_red
    # EDGE / EMPTY remain neutral

    return C1, C2, C3


def _build_D(n_rows:int, n_cols:int,
             start_pos:Optional[Tuple[int,int]]=None,
             start_ori:str="N") -> np.ndarray:
    """
    Prior over (pos,ori). If start_pos is None => uniform over all cells and orientations.
    Else a peaked prior at (start_pos, start_ori).
    """
    S = n_rows * n_cols * 4
    D = np.zeros(S, dtype=np.float64)
    if start_pos is None:
        D[:] = 1.0 / S
    else:
        r, c = start_pos
        ori = ORI2IDX[start_ori]
        D[_state_index(r, c, ori, n_rows, n_cols)] = 1.0
    return D


def _to_obj_array(*arrays: np.ndarray) -> np.ndarray:
    obj = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        obj[i] = a
    return obj


def build_trimodal_nav_agent(
    n_rows: int,
    n_cols: int,
    reward_pos: Tuple[int, int],
    punish_pos: Tuple[int, int],
    start_pos: Optional[Tuple[int, int]] = None,
    start_ori: str = "N",
    a_obs_noise: float = 0.0,
    b_model_noise: float = 0.0,
    policy_len: int = 4,
    gamma: float = 16.0,
    action_selection: str = "stochastic",
    c_green: float = 3.0,
    c_red: float = -3.0,
    sophisticated: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Build a tri-modal navigation agent for oriented grid environments.

    Constructs an Active Inference agent for environments with position,
    orientation, and three observation modalities: distance to terminal,
    terminal type, and current cell type.

    Args:
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        reward_pos: (row, col) position of reward terminal
        punish_pos: (row, col) position of punish terminal
        start_pos: (row, col) starting position, or None for uniform prior
        start_ori: Starting orientation ("N", "E", "S", "W")
        a_obs_noise: Observation noise for A matrices (0.0 = deterministic)
        b_model_noise: Model uncertainty for B matrix (0.0 = deterministic)
        policy_len: Planning horizon length
        gamma: Policy precision parameter
        action_selection: Action selection method ("stochastic" or "deterministic")
        c_green: Preference strength for green/reward observations
        c_red: Preference strength for red/punish observations
        sophisticated: Whether to use sophisticated inference if available

    Returns:
        Tuple of (agent, model, controls) where:
        - agent: pymdp Agent instance
        - model: Dict with generative model components and metadata
        - controls: Dict with policy inference wrapper

    Raises:
        ValueError: If grid dimensions or positions are invalid
    """
    # Validate inputs
    if n_rows < 2 or n_cols < 2:
        raise ValueError("Grid must be at least 2×2 for meaningful navigation")

    if not (0 <= reward_pos[0] < n_rows and 0 <= reward_pos[1] < n_cols):
        raise ValueError(f"Reward position {reward_pos} out of bounds for {n_rows}×{n_cols} grid")

    if not (0 <= punish_pos[0] < n_rows and 0 <= punish_pos[1] < n_cols):
        raise ValueError(f"Punish position {punish_pos} out of bounds for {n_rows}×{n_cols} grid")

    if start_ori not in ORIENTS:
        raise ValueError(f"Invalid start orientation '{start_ori}', must be one of {ORIENTS}")

    logger.info(
        f"Building tri-modal navigation agent: {n_rows}×{n_cols} grid, "
        f"reward={reward_pos}, punish={punish_pos}, start_pos={start_pos}, "
        f"start_ori={start_ori}, obs_noise={a_obs_noise}, model_noise={b_model_noise}"
    )
    # A (three modalities)
    A1, A2, A3 = _build_A(n_rows, n_cols, reward_pos, punish_pos, a_obs_noise=a_obs_noise)
    A = _to_obj_array(A1, A2, A3)

    # B (single factor, U=3)
    B = _build_B(n_rows, n_cols, b_model_noise=b_model_noise)
    B = _to_obj_array(B)

    # C preferences
    C1, C2, C3 = _build_C(n_rows, n_cols, pref_green=c_green, pref_red=c_red)
    C = _to_obj_array(C1, C2, C3)

    # D prior
    D = _to_obj_array(_build_D(n_rows, n_cols, start_pos=start_pos, start_ori=start_ori))

    # Build agent
    agent = Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        policy_len=policy_len,
        gamma=gamma,
        action_selection=action_selection,
        use_utility=True,
        use_states_info_gain=True,     # enable epistemic value
        use_param_info_gain=False
    )

    # A small control shim so your scripts can call sophisticated inference if available
    def infer_policies_shim():
        for kw in ({"mode": "sophisticated"}, {"method": "sophisticated"}):
            try:
                return agent.infer_policies(**kw)
            except TypeError:
                pass
        return agent.infer_policies()

    model = {
        "A": A, "B": B, "C": C, "D": D,
        "spaces": {
            "S": n_rows * n_cols * 4,
            "O1": A1.shape[0], "O2": A2.shape[0], "O3": A3.shape[0],
            "U": 3
        },
        "semantics": {
            "modalities": ["distance", "terminal_class", "current_class"],
            "classes": {
                "current_class": ["EMPTY","EDGE","RED","GREEN"],
                "terminal_class": ["EDGE","RED","GREEN"]
            },
            "actions": ["forward","turn_left","turn_right"],
            "orientations": ORIENTS
        }
    }
    controls = {"infer_policies": infer_policies_shim}

    return agent, model, controls
