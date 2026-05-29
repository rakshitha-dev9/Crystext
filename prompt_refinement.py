"""
Prompt refinement: normalize and auto-correct user formula / space group input
before sending to the CrysText model.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple

from pymatgen.core import Composition, Element

# Periodic table symbols (1–118)
ELEMENT_SYMBOLS: List[str] = [str(e.symbol) for e in Element]

# Common typos seen in free-text formula entry
ELEMENT_TYPOS: Dict[str, str] = {
    "Sodim": "Na",
    "Sodium": "Na",
    "Potasium": "K",
    "Potassium": "K",
    "Magnesuim": "Mg",
    "Magnesium": "Mg",
    "Aluminium": "Al",
    "Aluminum": "Al",
    "Silcon": "Si",
    "Silicon": "Si",
    "Titanum": "Ti",
    "Titanium": "Ti",
    "Iron": "Fe",
    "Oxygene": "O",
    "Oxygen": "O",
    "Chlorine": "Cl",
    "Florine": "F",
    "Fluorine": "F",
    "Baryum": "Ba",
    "Barium": "Ba",
    "Lantanum": "La",
    "Lanthanum": "La",
}

# Demo / MP-20-friendly formulas for whole-string fuzzy match
KNOWN_FORMULAS: List[str] = [
    "NaCl",
    "GaAs",
    "BaTiO3",
    "TiO2",
    "Fe2O3",
    "MgO",
    "SiO2",
    "Al2O3",
    "ZnO",
    "CuO",
    "SrTiO3",
    "CaTiO3",
    "LiFePO4",
    "YBa2Cu3O7",
    "NdAgHg2",
    "HfMnGe6",
    "RbNdS2",
    "Nd2Fe2Se2O3",
    "HoRe2SiC",
    "LuGaAu",
    "BaSrDyNbO6",
]

# Plain-language hints -> (formula, space group)
MATERIAL_ALIASES: Dict[str, Tuple[str, str]] = {
    "rock salt": ("NaCl", "225"),
    "rocksalt": ("NaCl", "225"),
    "halite": ("NaCl", "225"),
    "zinc blende": ("GaAs", "216"),
    "zincblende": ("GaAs", "216"),
    "perovskite": ("BaTiO3", "99"),
    "rutile": ("TiO2", "136"),
    "hematite": ("Fe2O3", "167"),
    "wurtzite": ("ZnO", "186"),
}

FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?|\d+)")
SPACEGROUP_SYMBOLS: Dict[str, int] = {
    "p1": 1,
    "p-1": 2,
    "fm-3m": 225,
    "fd-3m": 227,
    "pm-3m": 221,
    "i4/mmm": 139,
    "p6/mmm": 191,
    "r-3m": 166,
    "cmcm": 63,
    "c2/m": 12,
}


def _note(corrections: List[str], message: str) -> None:
    if message not in corrections:
        corrections.append(message)


def _fuzzy_element(token: str) -> Optional[str]:
    if token in ELEMENT_SYMBOLS:
        return token
    if token in ELEMENT_TYPOS:
        return ELEMENT_TYPOS[token]

    lower_map = {s.lower(): s for s in ELEMENT_SYMBOLS}
    if token.lower() in lower_map:
        return lower_map[token.lower()]

    typo_key = token.capitalize()
    if typo_key in ELEMENT_TYPOS:
        return ELEMENT_TYPOS[typo_key]

    matches = get_close_matches(token, ELEMENT_SYMBOLS, n=1, cutoff=0.72)
    if matches:
        return matches[0]
    return None


def refine_formula(raw: str, corrections: List[str]) -> str:
    text = raw.strip()
    if not text:
        return text

    text = re.sub(r"\s+", "", text)

    # Whole-formula fuzzy match (Nacl, BatIO3, etc.)
    norm_key = re.sub(r"[^A-Za-z0-9]", "", text).lower()
    for known in KNOWN_FORMULAS:
        if re.sub(r"[^A-Za-z0-9]", "", known).lower() == norm_key:
            if known != text:
                _note(corrections, f'Formula "{raw}" → "{known}" (known compound)')
            return known

    close = get_close_matches(
        norm_key,
        [re.sub(r"[^A-Za-z0-9]", "", k).lower() for k in KNOWN_FORMULAS],
        n=1,
        cutoff=0.82,
    )
    if close:
        idx = [re.sub(r"[^A-Za-z0-9]", "", k).lower() for k in KNOWN_FORMULAS].index(close[0])
        known = KNOWN_FORMULAS[idx]
        _note(corrections, f'Formula "{raw}" → "{known}" (close match)')
        return known

    tokens = FORMULA_TOKEN_RE.findall(text)
    if not tokens:
        return text

    rebuilt: List[str] = []
    for token in tokens:
        if token.isdigit():
            rebuilt.append(token)
            continue
        fixed = _fuzzy_element(token)
        if fixed is None:
            rebuilt.append(token)
        else:
            if fixed != token:
                _note(corrections, f'Element "{token}" → "{fixed}"')
            rebuilt.append(fixed)

    candidate = "".join(rebuilt)

    try:
        comp = Composition(candidate)
        canonical = comp.reduced_formula
        if canonical != candidate:
            _note(corrections, f'Formula normalized to "{canonical}"')
        return canonical
    except Exception:
        return candidate


def refine_spacegroup(raw: str, corrections: List[str]) -> Optional[str]:
    text = raw.strip()
    if not text:
        return None

    lowered = text.lower().replace(" ", "")
    if lowered in SPACEGROUP_SYMBOLS:
        sg_num = SPACEGROUP_SYMBOLS[lowered]
        _note(corrections, f'Space group "{raw}" → "{sg_num}" (symbol)')
        return str(sg_num)

    # OCR fixes: 22O -> 225, 22O5 -> 2205 -> trim
    cleaned = re.sub(r"(?<=\d)[Oo](?=\d)", "0", text)
    cleaned = re.sub(r"(?<=\d)[Oo]$", "5", cleaned)
    cleaned = re.sub(r"[Oo]$", "5", cleaned)
    cleaned = cleaned.replace(" ", "")

    digits = re.sub(r"[^\d]", "", cleaned)
    if not digits:
        return None

    try:
        sg = int(digits)
    except ValueError:
        return None
    try:
        if sg != int(text) and cleaned != text:
            _note(corrections, f'Space group "{raw}" → "{sg}" (digit cleanup)')
    except ValueError:
        _note(corrections, f'Space group "{raw}" → "{sg}" (digit cleanup)')

    if 1 <= sg <= 230:
        return str(sg)

    # Fuzzy to valid range (e.g. 252 -> 225)
    close = get_close_matches(str(sg), [str(i) for i in range(1, 231)], n=1, cutoff=0.85)
    if close:
        _note(corrections, f'Space group "{raw}" → "{close[0]}" (nearest valid)')
        return close[0]

    return None


def refine_from_description(description: str, corrections: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not description:
        return None, None
    lower = description.lower()
    for phrase, (formula, sg) in MATERIAL_ALIASES.items():
        if phrase in lower:
            _note(corrections, f'Description matched "{phrase}" → {formula}, SG {sg}')
            return formula, sg
    return None, None


def _rule_based_refine(
    formula: str,
    spacegroup: str,
    description: str,
    corrections: List[str],
) -> Tuple[str, str]:
    refined_formula = formula
    refined_spacegroup = spacegroup

    desc_formula, desc_sg = refine_from_description(description, corrections)
    if desc_formula and not formula:
        refined_formula = desc_formula
    if desc_sg and not spacegroup:
        refined_spacegroup = desc_sg

    if formula:
        refined_formula = refine_formula(formula, corrections)

    if spacegroup:
        sg = refine_spacegroup(spacegroup, corrections)
        if sg is not None:
            refined_spacegroup = sg

    return refined_formula, refined_spacegroup


def refine_user_input(
    formula: Optional[str] = None,
    spacegroup: Optional[str] = None,
    description: Optional[str] = None,
    use_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Refine user inputs. Returns original, refined values, correction notes, and changed flag.
    Pipeline: optional small LLM -> rule-based validation/normalization.
    """
    corrections: List[str] = []
    original_formula = (formula or "").strip()
    original_spacegroup = (spacegroup or "").strip()
    original_description = (description or "").strip()

    refined_formula = original_formula
    refined_spacegroup = original_spacegroup
    method = "rules"

    llm_enabled = use_llm
    if llm_enabled is None:
        try:
            from llm_prompt_refiner import is_llm_refiner_enabled

            llm_enabled = is_llm_refiner_enabled()
        except ImportError:
            llm_enabled = False

    if llm_enabled:
        try:
            from llm_prompt_refiner import llm_refine_user_input

            llm_out = llm_refine_user_input(
                formula=original_formula or None,
                spacegroup=original_spacegroup or None,
                description=original_description or None,
            )
            method = "llm+rules"
            if llm_out.get("formula"):
                if llm_out["formula"] != original_formula:
                    _note(
                        corrections,
                        f'LLM formula: "{original_formula or "(empty)"}" → "{llm_out["formula"]}"',
                    )
                refined_formula = llm_out["formula"]
            if llm_out.get("spacegroup"):
                if llm_out["spacegroup"] != original_spacegroup:
                    _note(
                        corrections,
                        f'LLM space group: "{original_spacegroup or "(empty)"}" → "{llm_out["spacegroup"]}"',
                    )
                refined_spacegroup = llm_out["spacegroup"]
            if llm_out.get("notes"):
                _note(corrections, f'LLM: {llm_out["notes"]}')
        except Exception as exc:
            _note(corrections, f"LLM refiner skipped ({exc})")

    refined_formula, refined_spacegroup = _rule_based_refine(
        refined_formula,
        refined_spacegroup,
        original_description,
        corrections,
    )

    changed = (
        refined_formula != original_formula
        or refined_spacegroup != original_spacegroup
    )

    return {
        "original": {
            "formula": original_formula,
            "spacegroup": original_spacegroup,
            "description": original_description,
        },
        "refined": {
            "formula": refined_formula,
            "spacegroup": refined_spacegroup,
        },
        "corrections": corrections,
        "changed": changed,
        "method": method,
    }
