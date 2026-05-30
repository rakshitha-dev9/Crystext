"""
TRL-compatible reward function for CrysText-RL / GRPO.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Project root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crystext_rewards import reward_from_completion


def crystext_reward_func(
    completions: List[str],
    formula: List[str],
    spacegroup: List[str],
    reference_cif: List[str],
    **kwargs,
) -> List[float]:
    """
    GRPO reward: multi-stage CIF validation (paper Fig. S1 / Method II-B).

    Dataset columns passed by TRL: formula, spacegroup, reference_cif.
    """
    rewards: List[float] = []
    for completion, f, _sg, ref in zip(completions, formula, spacegroup, reference_cif):
        rewards.append(
            reward_from_completion(
                completion=completion,
                expected_formula=f,
                reference_cif=ref,
            )
        )
    return rewards
