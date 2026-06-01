"""
CrysText Flask Backend
======================
Paper-aligned backend for text-conditioned crystal generation:
- SFT-style prompt format
- Optional energy-above-hull conditioning
- Multi-sample generation
- Deterministic reward evaluation inspired by CrysText-RL/GRPO
"""

import traceback
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from crystext_rewards import build_prompt, evaluate_reward, extract_cif
from crystext_rewards import _parse_structure
from prompt_refinement import refine_user_input

app = Flask(__name__)
CORS(app)

BASE_MODEL_ID = "mistralai/Mistral-7B-v0.3"
LORA_MODEL_ID = "vaishna28/shuffle_20k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'=' * 60}")
print("CrysText Backend Starting")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'=' * 60}\n")

print("Step 1/3: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded")

print("Step 2/3: Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
print("Base model loaded")

print("Step 3/3: Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID)
model.eval()
print("LoRA adapter loaded")
print("\nCrysText is ready! Server starting on http://localhost:5000\n")


def validate_cif(cif_text: str, expected_formula: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "valid": False,
        "composition_match": False,
        "detected_spacegroup": None,
        "detected_formula": None,
        "conventional_cif": None,
        "error": None,
    }
    try:
        from pymatgen.core import Composition
        from pymatgen.io.cif import CifWriter
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = _parse_structure(cif_text)
        result["valid"] = True

        try:
            sga = SpacegroupAnalyzer(structure)
            conventional = sga.get_conventional_standard_structure()
            result["detected_spacegroup"] = sga.get_space_group_number()
            result["conventional_cif"] = str(CifWriter(conventional, symprec=0.1))
            structure = conventional
        except Exception as ex:
            print(f"Conventional cell conversion failed: {ex}")
            result["detected_spacegroup"] = None

        result["detected_formula"] = structure.composition.reduced_formula

        if expected_formula:
            try:
                expected_els = {str(e) for e in Composition(expected_formula).elements}
                detected_els = {str(e) for e in structure.composition.elements}
                result["composition_match"] = expected_els == detected_els
            except Exception:
                result["composition_match"] = False
    except Exception as e:
        result["error"] = str(e)

    return result


def _generate_raw_outputs(
    prompt: str,
    num_samples: int = 1,
    max_new_tokens: int = 1536,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_texts: List[str] = []

    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return output_texts


def _apply_refinement(data: Dict[str, Any]) -> Dict[str, Any]:
    """Refine formula/spacegroup unless client sets auto_refine=false."""
    if data.get("auto_refine") is False:
        return {
            "refinement": {"changed": False, "corrections": []},
            "formula": str(data.get("formula", "")).strip(),
            "spacegroup": str(data.get("spacegroup", "")).strip(),
        }

    refinement = refine_user_input(
        formula=data.get("formula"),
        spacegroup=data.get("spacegroup"),
        description=data.get("description"),
        use_llm=data.get("use_llm"),
    )
    return {
        "refinement": refinement,
        "formula": refinement["refined"]["formula"],
        "spacegroup": refinement["refined"]["spacegroup"],
    }


@app.route("/refine_prompt", methods=["POST"])
def refine_prompt() -> Any:
    """Auto-correct formula / space group before generation."""
    try:
        data = request.get_json() or {}
        result = refine_user_input(
            formula=data.get("formula"),
            spacegroup=data.get("spacegroup"),
            description=data.get("description"),
            use_llm=data.get("use_llm"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate() -> Any:
    """
    Backward-compatible endpoint for the current frontend.
    """
    try:
        data = request.get_json() or {}
        energy_above_hull = data.get("energy_above_hull")
        refined = _apply_refinement(data)
        formula = refined["formula"]
        spacegroup = refined["spacegroup"]
        refinement = refined["refinement"]

        if energy_above_hull is None:
            if not formula:
                return jsonify({"error": "Formula is required"}), 400
            if not spacegroup:
                return jsonify({"error": "Space group number is required"}), 400
            prompt = build_prompt(formula=formula, spacegroup=spacegroup)
        else:
            prompt = build_prompt(energy_above_hull=float(energy_above_hull))

        raw_output = _generate_raw_outputs(prompt=prompt, num_samples=1, do_sample=False)[0]
        cif_text = extract_cif(raw_output)
        validation = validate_cif(cif_text, expected_formula=formula if formula else None)
        display_cif = validation.get("conventional_cif") or cif_text

        return jsonify(
            {
                "cif": display_cif,
                "validation": validation,
                "refinement": refinement,
                "used_input": {"formula": formula, "spacegroup": spacegroup},
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate_batch", methods=["POST"])
def generate_batch() -> Any:
    """
    Paper-style multi-sampling endpoint (N candidates).
    """
    try:
        data = request.get_json() or {}
        energy_above_hull = data.get("energy_above_hull")
        refined = _apply_refinement(data)
        formula = refined["formula"]
        spacegroup = refined["spacegroup"]
        refinement = refined["refinement"]
        num_samples = int(data.get("num_samples", 6))
        max_new_tokens = int(data.get("max_new_tokens", 1536))
        do_sample = bool(data.get("do_sample", True))
        temperature = float(data.get("temperature", 1.0))
        top_p = float(data.get("top_p", 0.95))

        if num_samples < 1 or num_samples > 50:
            return jsonify({"error": "num_samples must be between 1 and 50"}), 400

        if energy_above_hull is None:
            if not formula:
                return jsonify({"error": "Formula is required"}), 400
            if not spacegroup:
                return jsonify({"error": "Space group number is required"}), 400
            prompt = build_prompt(formula=formula, spacegroup=spacegroup)
        else:
            prompt = build_prompt(energy_above_hull=float(energy_above_hull))

        raw_outputs = _generate_raw_outputs(
            prompt=prompt,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        results = []
        for raw in raw_outputs:
            cif_text = extract_cif(raw)
            validation = validate_cif(cif_text, expected_formula=formula if formula else None)
            display_cif = validation.get("conventional_cif") or cif_text
            results.append({"cif": display_cif, "validation": validation})

        return jsonify(
            {
                "count": len(results),
                "sampling": {
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_samples": num_samples,
                },
                "results": results,
                "refinement": refinement,
                "used_input": {"formula": formula, "spacegroup": spacegroup},
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate_reward", methods=["POST"])
def evaluate_reward_endpoint() -> Any:
    """
    Reward endpoint for generated CIFs (paper-inspired deterministic stages).
    """
    try:
        data = request.get_json() or {}
        cif = data.get("cif")
        formula = data.get("formula")
        reference_cif = data.get("reference_cif")

        if not cif:
            return jsonify({"error": "cif is required"}), 400

        reward_details = evaluate_reward(
            cif_text=str(cif),
            expected_formula=str(formula).strip() if formula else None,
            reference_cif=str(reference_cif) if reference_cif else None,
        )
        return jsonify(reward_details)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health() -> Any:
    llm_refiner = {"enabled": False, "model": None}
    try:
        from llm_prompt_refiner import DEFAULT_MODEL_ID, is_llm_refiner_enabled

        llm_refiner = {
            "enabled": is_llm_refiner_enabled(),
            "model": DEFAULT_MODEL_ID,
        }
    except ImportError:
        pass

    return jsonify(
        {
            "status": "ok",
            "device": DEVICE,
            "base_model": BASE_MODEL_ID,
            "adapter_model": LORA_MODEL_ID,
            "llm_refiner": llm_refiner,
        }
    )

@app.route("/chat", methods=["POST"])
def chat():
    try:
        import urllib.request
        import json as json_lib

        data = request.get_json() or {}
        messages = data.get('messages', [])
        api_key = data.get('api_key', '')
        groq_key = data.get('groq_key', '')

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        SYSTEM_PROMPT = """You are a specialized materials science assistant embedded in CrysText. You ONLY answer questions related to crystal structures, space groups, CIF files, materials science, QLoRA, DFT, and the MP-20 dataset. For anything else say: I'm specialized in materials science only!"""

        # 1. Try Groq first (fastest + most reliable)
        if groq_key:
            try:
                groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in messages:
                    groq_messages.append({"role": msg['role'], "content": msg['content']})

                groq_payload = json_lib.dumps({
                    "model": "llama-3.3-70b-versatile",
                    "messages": groq_messages,
                    "max_tokens": 500,
                    "temperature": 0.7
                }).encode('utf-8')

                req = urllib.request.Request(
                    'https://api.groq.com/openai/v1/chat/completions',
                    data=groq_payload,
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {groq_key}'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json_lib.loads(resp.read().decode('utf-8'))
                    reply = result['choices'][0]['message']['content']
                    return jsonify({"reply": reply, "model_used": "groq/llama-3.3-70b"})
            except Exception as e:
                print(f"Groq failed, trying Gemini: {e}")

        # 2. Fallback — Gemini
        if not api_key:
            return jsonify({"error": "No API keys provided"}), 400

        gemini_contents = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_contents.append({'role': role, 'parts': [{'text': msg['content']}]})

        payload = json_lib.dumps({
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": gemini_contents,
            "generationConfig": {"maxOutputTokens": 500, "temperature": 0.7}
        }).encode('utf-8')

        GEMINI_MODELS = ['gemini-2.5-flash', 'gemini-2.5-flash-lite']
        last_error = None

        for model in GEMINI_MODELS:
            try:
                url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'
                req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json_lib.loads(resp.read().decode('utf-8'))
                    reply = result['candidates'][0]['content']['parts'][0]['text']
                    return jsonify({"reply": reply, "model_used": model})
            except urllib.error.HTTPError as e:
                error_body = e.read().decode('utf-8')
                print(f"Gemini {model} error: {e.code} - {error_body}")
                if e.code in (503, 429):
                    last_error = f"Gemini error {e.code}: {error_body}"
                    continue
                return jsonify({"error": f"Gemini error {e.code}: {error_body}"}), 500
            except Exception as e:
                print(f"Gemini {model} failed: {e}")
                last_error = str(e)
                continue

        return jsonify({"error": f"All models unavailable. Last error: {last_error}"}), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)