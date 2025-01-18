import os
import random
import numpy as np
import torch
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define seed value bounds
min_seed_value = 0
max_seed_value = (2**32) - 1

def _select_seed_randomly(min_seed: int, max_seed: int) -> int:
    """Helper function to select a random seed within the range."""
    return random.randint(min_seed, max_seed)

def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets the seed for pseudo-random number generators in pytorch, numpy, and random.
    Sets the following environment variables:
    - `PL_GLOBAL_SEED`: Passed to spawned subprocesses (if applicable).
    - `PL_SEED_WORKERS`: (optional) Set to 1 if `workers=True`.

    Args:
        seed: The integer value for the global random state seed. If `None`, a random seed is selected.
        workers: If `True`, sets environment variable `PL_SEED_WORKERS` for worker initialization.

    Returns:
        The seed value used.
    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            log.warning(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                log.warning(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        log.warning(f"{seed} is out of bounds, resetting seed to a random value.")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed
