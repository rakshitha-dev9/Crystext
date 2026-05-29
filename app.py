"""
CrysText Flask Backend
======================
Loads rakshitha9/crystext-mistral-10k from HuggingFace
Generates CIF files from formula + space group
Validates with pymatgen, returns conventional cell
Runs on localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import traceback

app = Flask(__name__)
CORS(app)

BASE_MODEL_ID = "mistralai/Mistral-7B-v0.3"
LORA_MODEL_ID = "rakshitha9/crystext-mistral-10k"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*50}")
print(f"CrysText Backend Starting")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*50}\n")

print("Step 1/3: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded")

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
    dtype=torch.float16,
)
print("✅ Base model loaded")

print("Step 3/3: Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID)
model.eval()
print("✅ LoRA adapter loaded")
print("\n🚀 CrysText is ready! Server starting on http://localhost:5000\n")


def build_prompt(formula: str, spacegroup: str) -> str:
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "Generate CIF for the given material description\n\n"
        "### Input:\n"
        f"Material composition is {formula}. It has a space group number {spacegroup}.\n\n"
        "### Response:\n"
    )


def validate_cif(cif_text: str, expected_formula: str):
    result = {
        "valid": False,
        "composition_match": False,
        "detected_spacegroup": None,
        "detected_formula": None,
        "conventional_cif": None,
        "error": None,
    }
    try:
        from pymatgen.io.cif import CifParser, CifWriter
        from pymatgen.core import Composition
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        # Parse — get primitive first (most reliable from model output)
        parser = CifParser.from_str(cif_text)
        structures = parser.parse_structures(primitive=True)

        if not structures:
            result["error"] = "Pymatgen could not parse any structure from CIF"
            return result

        structure = structures[0]
        result["valid"] = True

        # Convert primitive -> conventional standard cell
        # This gives us a=5.64 cubic for NaCl instead of a=3.99 rhombohedral
        try:
            sga = SpacegroupAnalyzer(structure)
            conventional = sga.get_conventional_standard_structure()
            structure = conventional
            result["detected_spacegroup"] = sga.get_space_group_number()
            # Re-export as proper CIF with symmetry info
            cif_writer = CifWriter(structure, symprec=0.1)
            result["conventional_cif"] = str(cif_writer)
        except Exception as ex:
            print(f"  Conventional cell conversion failed: {ex}")
            result["detected_spacegroup"] = "Could not determine"
            result["conventional_cif"] = None

        # Formula info
        result["detected_formula"] = structure.composition.reduced_formula

        # Composition match
        try:
            expected_els = set(str(e) for e in Composition(expected_formula).elements)
            detected_els = set(str(e) for e in structure.composition.elements)
            result["composition_match"] = (expected_els == detected_els)
        except Exception:
            result["composition_match"] = False

    except Exception as e:
        result["error"] = str(e)

    return result


def extract_cif(raw_output: str) -> str:
    if "### Response:" in raw_output:
        cif = raw_output.split("### Response:")[-1].strip()
    else:
        cif = raw_output.strip()
    lines = cif.split('\n')
    clean = []
    for line in lines:
        if line.strip().startswith("###"):
            break
        clean.append(line)
    return '\n'.join(clean).strip()


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        formula = data.get('formula', '').strip()
        spacegroup = data.get('spacegroup', '').strip()

        if not formula:
            return jsonify({"error": "Formula is required"}), 400
        if not spacegroup:
            return jsonify({"error": "Space group number is required"}), 400

        print(f"\n→ Generating CIF for: {formula}, SG {spacegroup}")

        prompt = build_prompt(formula, spacegroup)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        print("  Running inference...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1536,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cif_text = extract_cif(raw)

        print("  Validating with pymatgen...")
        validation = validate_cif(cif_text, formula)

        # Use conventional cell CIF if pymatgen could generate it
        # Otherwise fall back to raw model output
        display_cif = validation.get("conventional_cif") or cif_text

        print(f"  ✅ Done — valid={validation['valid']}, formula_match={validation['composition_match']}, detected_sg={validation['detected_spacegroup']}")

        return jsonify({
            "cif": display_cif,
            "validation": validation,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "device": DEVICE})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        import urllib.request
        import json as json_lib

        data = request.get_json()
        messages = data.get('messages', [])
        api_key = data.get('api_key', '')

        if not api_key:
            return jsonify({"error": "API key missing"}), 400
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        SYSTEM_PROMPT = """You are a specialized materials science assistant embedded in CrysText, an AI tool for crystal structure generation. You ONLY answer questions related to:
- Crystal structures, crystallography, and CIF files
- Space groups and symmetry
- Materials science concepts (band gaps, lattice parameters, unit cells, etc.)
- The MP-20 dataset and Materials Project
- Common materials like NaCl, GaAs, BaTiO3, TiO2, Fe2O3, MgO
- QLoRA, fine-tuning, and how CrysText works
- DFT (Density Functional Theory) basics
- Indian materials science applications and research

If asked anything outside materials science, politely say: "I'm specialized in materials science only. Please ask me about crystal structures, space groups, or materials!"

Keep answers clear, concise, and friendly. Use simple language for non-experts."""

        # Convert messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = 'user' if msg['role'] == 'user' else 'model'
            gemini_contents.append({
                'role': role,
                'parts': [{'text': msg['content']}]
            })

        payload = json_lib.dumps({
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": gemini_contents,
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.7
            }
        }).encode('utf-8')

        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}'

        req = urllib.request.Request(
            url,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json_lib.loads(resp.read().decode('utf-8'))
            reply = result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({"reply": reply})

    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"Gemini API error: {e.code} - {error_body}")
        traceback.print_exc()
        return jsonify({"error": f"Gemini error {e.code}: {error_body}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)