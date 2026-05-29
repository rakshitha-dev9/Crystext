"""
Small LLM-based prompt refiner for CrysText user inputs.

Uses a lightweight instruct model (default: Qwen2.5-0.5B-Instruct) on CPU by default
so it does not compete with the main Mistral generation model for VRAM.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

import torch
from pymatgen.core import Composition
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 160

SYSTEM_PROMPT = (
    "You normalize user input for a crystal-structure generation system. "
    "Fix spelling mistakes, infer missing fields from the description when obvious, "
    "and output valid chemistry notation. "
    "Space group must be an integer from 1 to 230. "
    "Chemical formula must use correct element symbols (e.g. NaCl, BaTiO3, Fe2O3). "
    "Reply with JSON only, no markdown."
)

USER_TEMPLATE = """Fix and extract fields from this user input.

formula: {formula}
space_group: {spacegroup}
description: {description}

Return exactly one JSON object with keys:
- chemical_formula (string or null)
- space_group (string integer 1-230 or null)
- notes (short string explaining corrections)"""


class SmallLLMRefiner:
    _lock = threading.Lock()
    _instance: Optional["SmallLLMRefiner"] = None

    def __init__(self) -> None:
        self.model_id = os.getenv("LLM_REFINE_MODEL", DEFAULT_MODEL_ID)
        self.device = os.getenv("LLM_REFINE_DEVICE", "cpu").lower()
        self._loaded = False
        self.model = None
        self.tokenizer = None

    @classmethod
    def get(cls) -> "SmallLLMRefiner":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            print(f"[LLM Refiner] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print("[LLM Refiner] Ready")

    def refine(
        self,
        formula: Optional[str] = None,
        spacegroup: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()

        user_text = USER_TEMPLATE.format(
            formula=formula or "",
            spacegroup=spacegroup or "",
            description=description or "",
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{user_text}\n\nJSON:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        parsed = _parse_json_object(generated)
        validated = _validate_llm_output(parsed)

        return {
            "formula": validated.get("chemical_formula"),
            "spacegroup": validated.get("space_group"),
            "notes": validated.get("notes") or "",
            "raw": generated.strip(),
            "model": self.model_id,
        }


def is_llm_refiner_enabled() -> bool:
    return os.getenv("ENABLE_LLM_REFINER", "true").strip().lower() in {"1", "true", "yes", "on"}


def llm_refine_user_input(
    formula: Optional[str] = None,
    spacegroup: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    refiner = SmallLLMRefiner.get()
    return refiner.refine(formula=formula, spacegroup=spacegroup, description=description)


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _validate_llm_output(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "chemical_formula": None,
        "space_group": None,
        "notes": str(data.get("notes", "")).strip(),
    }

    formula = data.get("chemical_formula") or data.get("formula")
    if formula:
        formula_str = re.sub(r"\s+", "", str(formula).strip())
        try:
            out["chemical_formula"] = Composition(formula_str).reduced_formula
        except Exception:
            out["chemical_formula"] = formula_str

    sg = data.get("space_group") or data.get("spacegroup")
    if sg is not None and str(sg).strip():
        sg_digits = re.sub(r"[^\d]", "", str(sg))
        if sg_digits:
            sg_num = int(sg_digits)
            if 1 <= sg_num <= 230:
                out["space_group"] = str(sg_num)

    return out
