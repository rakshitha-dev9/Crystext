"""
Shared CrysText reward logic (paper / CrysText-RL).
Used by inference API and GRPO training.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Paper: penalize invalid CIF parses during RL
REWARD_PARSE_FAILURE = -0.5
REWARD_PARSE_OK = 0.10
REWARD_PHYSICAL = 0.20
REWARD_COMPOSITION = 0.20
REWARD_STRUCTURE_LOW = 0.50
REWARD_STRUCTURE_MED = 0.35
REWARD_STRUCTURE_HIGH = 0.20


def build_prompt(
    formula: Optional[str] = None,
    spacegroup: Optional[str] = None,
    energy_above_hull: Optional[float] = None,
) -> str:
    if energy_above_hull is None:
        instruction = "Generate CIF for the given material description"
        user_input = (
            f"Material composition is {formula}. "
            f"It has a space group number {spacegroup}."
        )
    else:
        instruction = "Generate CIF for a stable material based on the given description"
        user_input = f"Energy above hull in eV/atom is {energy_above_hull:.6f}."

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{user_input}\n\n"
        "### Response:\n"
    )


def extract_cif(raw_output: str) -> str:
    if "### Response:" in raw_output:
        cif = raw_output.split("### Response:")[-1].strip()
    else:
        cif = raw_output.strip()

    clean: List[str] = []
    for line in cif.split("\n"):
        if line.strip().startswith("###"):
            break
        clean.append(line)
    return "\n".join(clean).strip()


def _parse_structure(cif_text: str):
    from pymatgen.core import Structure

    return Structure.from_str(cif_text, fmt="cif")


def _physical_validity(structure) -> Tuple[bool, Dict[str, Any]]:
    min_distance = None
    if len(structure) >= 2:
        dist_matrix = structure.distance_matrix
        non_diag = dist_matrix[dist_matrix > 0]
        min_distance = float(non_diag.min()) if len(non_diag) else None

    volume_ok = float(structure.volume) > 0.1
    distance_ok = (min_distance is None) or (min_distance > 0.5)
    is_valid = volume_ok and distance_ok
    return is_valid, {
        "volume": float(structure.volume),
        "min_distance": min_distance,
        "volume_ok": volume_ok,
        "distance_ok": distance_ok,
    }


def _structure_match_reward(generated_structure, reference_structure) -> Tuple[float, Dict[str, bool]]:
    from pymatgen.analysis.structure_matcher import StructureMatcher

    levels = {
        "high": StructureMatcher(stol=0.9, ltol=0.7, angle_tol=20),
        "medium": StructureMatcher(stol=0.7, ltol=0.5, angle_tol=15),
        "low": StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10),
    }
    matches = {
        name: matcher.fit(generated_structure, reference_structure)
        for name, matcher in levels.items()
    }
    if matches["low"]:
        return REWARD_STRUCTURE_LOW, matches
    if matches["medium"]:
        return REWARD_STRUCTURE_MED, matches
    if matches["high"]:
        return REWARD_STRUCTURE_HIGH, matches
    return 0.0, matches


def compute_reward(
    cif_text: str,
    expected_formula: Optional[str] = None,
    reference_cif: Optional[str] = None,
) -> float:
    """Scalar reward for GRPO (group-normalized by TRL trainer)."""
    details = evaluate_reward(cif_text, expected_formula, reference_cif)
    return float(details["reward"])


def evaluate_reward(
    cif_text: str,
    expected_formula: Optional[str] = None,
    reference_cif: Optional[str] = None,
) -> Dict[str, Any]:
    reward = 0.0
    details: Dict[str, Any] = {
        "parsed": False,
        "physical_valid": False,
        "composition_match": False,
        "structure_match": {"high": False, "medium": False, "low": False},
        "components": {
            "parse": 0.0,
            "physical": 0.0,
            "composition": 0.0,
            "structure": 0.0,
        },
        "meta": {},
        "error": None,
    }

    try:
        from pymatgen.core import Composition

        structure = _parse_structure(cif_text)
        details["parsed"] = True
        details["components"]["parse"] = REWARD_PARSE_OK
        reward += REWARD_PARSE_OK

        physical_valid, physical_meta = _physical_validity(structure)
        details["meta"]["physical"] = physical_meta
        if physical_valid:
            details["physical_valid"] = True
            details["components"]["physical"] = REWARD_PHYSICAL
            reward += REWARD_PHYSICAL

        if expected_formula:
            expected_els = {str(e) for e in Composition(expected_formula).elements}
            detected_els = {str(e) for e in structure.composition.elements}
            if expected_els == detected_els:
                details["composition_match"] = True
                details["components"]["composition"] = REWARD_COMPOSITION
                reward += REWARD_COMPOSITION

        if reference_cif:
            ref_structure = _parse_structure(reference_cif)
            structure_reward, matches = _structure_match_reward(structure, ref_structure)
            details["structure_match"] = matches
            details["components"]["structure"] = structure_reward
            reward += structure_reward
    except Exception as e:
        details["error"] = str(e)
        reward = REWARD_PARSE_FAILURE
        details["components"]["parse"] = REWARD_PARSE_FAILURE

    details["reward"] = round(reward, 6)
    return details


def reward_from_completion(
    completion: str,
    expected_formula: str,
    reference_cif: str,
) -> float:
    """Reward for one generated completion string."""
    cif_text = extract_cif(completion)
    if not cif_text.strip():
        return REWARD_PARSE_FAILURE
    return compute_reward(cif_text, expected_formula, reference_cif)
