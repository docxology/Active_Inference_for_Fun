"""
Compatibility helpers for pymdp agent API differences.

This module provides compatibility functions that handle differences in pymdp
agent APIs across versions and use cases (single vs multi-modality observations).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def infer_states_compat(
    agent: Any,
    obs: Union[int, List[int], Tuple[int, ...], List[Tuple[int, ...]]],
    modalities: Optional[List[str]] = None,
) -> None:
    """
    Infer states from observations, handling different pymdp API versions.

    Args:
        agent: pymdp agent instance
        obs: Observation(s). Can be:
            - Single scalar for single-modality agents
            - List/tuple of scalars for single-modality agents
            - Tuple of scalars for multi-modality agents
            - List of tuples for batched multi-modality observations
        modalities: Optional list of modality names for logging

    Raises:
        RuntimeError: If all inference attempts fail
    """
    obs_type = type(obs).__name__
    logger.debug(f"Inferring states from observation (type: {obs_type})")

    # Try different API patterns
    attempts = [
        lambda: agent.infer_states(obs),  # Direct pass
        lambda: agent.infer_states([obs]) if isinstance(obs, (int, float)) else None,  # Wrap scalar
        lambda: agent.infer_states(list(obs)) if hasattr(obs, '__iter__') and not isinstance(obs, str) else None,  # Convert to list
    ]

    for attempt in attempts:
        if attempt is None:
            continue
        try:
            attempt()
            return
        except Exception as e:
            logger.debug(f"Inference attempt failed: {e}")
            continue

    raise RuntimeError(f"Could not infer states from observation {obs}")


def sample_action_compat(agent: Any) -> int:
    """
    Sample an action from the agent, returning a single integer.

    Handles different pymdp API versions that may return actions as
    scalars, lists, tuples, or arrays.

    Args:
        agent: pymdp agent instance

    Returns:
        Action as integer

    Raises:
        RuntimeError: If action sampling fails or returns unexpected format
    """
    try:
        action = agent.sample_action()
        logger.debug(f"Sampled raw action: {action} (type: {type(action).__name__})")

        # Handle different return types
        if isinstance(action, (int, float)):
            return int(action)
        elif isinstance(action, (list, tuple)):
            if len(action) == 1:
                return int(action[0])
            else:
                raise ValueError(f"Expected single action, got {len(action)} actions: {action}")
        elif hasattr(action, '__array__'):  # numpy array
            action_array = action.__array__()
            if action_array.ndim == 0:  # scalar array
                return int(action_array.item())
            elif action_array.ndim == 1 and len(action_array) == 1:
                return int(action_array[0])
            else:
                raise ValueError(f"Expected scalar action array, got shape {action_array.shape}")
        else:
            raise ValueError(f"Unexpected action type: {type(action)}")

    except Exception as e:
        logger.error(f"Failed to sample action: {e}")
        raise RuntimeError(f"Action sampling failed: {e}") from e


def reset_agent_compat(agent: Any) -> None:
    """
    Reset agent beliefs if the reset method is available.

    This is a soft reset that doesn't raise errors if the method
    doesn't exist or fails, as it's not always necessary.

    Args:
        agent: pymdp agent instance
    """
    if hasattr(agent, "reset") and callable(agent.reset):
        try:
            agent.reset()
            logger.debug("Agent beliefs reset successfully")
        except Exception as e:
            logger.debug(f"Agent reset failed (non-critical): {e}")
    else:
        logger.debug("Agent does not have reset method")


def infer_policies_compat(
    agent: Any,
    sophisticated: bool = False,
    controls: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Infer policies using appropriate method for agent and sophistication level.

    Args:
        agent: pymdp agent instance
        sophisticated: Whether to use sophisticated inference if available
        controls: Optional controls dict from agent factory

    Returns:
        Result of policy inference (typically None)

    Raises:
        RuntimeError: If all inference attempts fail
    """
    # Prefer factory-provided wrapper if available
    if controls and callable(controls.get("infer_policies")):
        logger.debug("Using factory-provided policy inference wrapper")
        return controls["infer_policies"]()

    # Try sophisticated inference if requested
    if sophisticated:
        sophisticated_kwargs = [
            {"mode": "sophisticated"},
            {"method": "sophisticated"},
        ]

        for kwargs in sophisticated_kwargs:
            try:
                result = agent.infer_policies(**kwargs)
                logger.debug(f"Successfully used sophisticated inference with kwargs: {kwargs}")
                return result
            except TypeError:
                continue
            except Exception as e:
                logger.debug(f"Sophisticated inference failed with {kwargs}: {e}")
                continue

        logger.warning("Sophisticated inference requested but not available, falling back to default")

    # Fall back to default inference
    try:
        result = agent.infer_policies()
        logger.debug("Used default policy inference")
        return result
    except Exception as e:
        logger.error(f"All policy inference attempts failed: {e}")
        raise RuntimeError(f"Policy inference failed: {e}") from e
