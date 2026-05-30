# CrysText — Text-Conditioned Crystal Structure Generation

Generate valid crystal structure CIF files from a material formula and space group number using a fine-tuned large language model.

---

## What It Does

Type a formula like `NaCl` and space group `225` → CrysText generates a complete Crystallographic Information File (CIF) with cell parameters, atom positions, and symmetry information, rendered as an interactive 3D ball-and-stick structure in a dedicated structure viewer page.

---

## Model

- **Base:** Mistral-7B-v0.3
- **Fine-tuning:** QLoRA (r=16, lora_alpha=16), 4-bit quantization
- **Dataset:** MP-20 (27,136 experimentally verified crystal structures)
- **Training:** Full 27k samples, supervised fine-tuning complete
- **HuggingFace:** https://huggingface.co/Charanya-2026/crystext-mistral-27k

---

## Requirements

- Python 3.10+
- CUDA GPU with at least **8GB VRAM**
- CUDA drivers installed

---

## Installation

```bash
git clone https://github.com/rakshitha-dev9/Crystext.git
cd Crystext
pip install -r requirements.txt
```

---

## Running The App

You need two terminals open at the same time.

**Terminal 1 — Start the Flask backend:**
```bash
python app.py
```
Wait until you see:
```
CrysText is ready! Server starting on http://localhost:5000
```
This takes 10-15 minutes on first run — the model is downloading and loading.

**Terminal 2 — Start the frontend:**
```bash
python -m http.server 8080
```

**Then open your browser and go to:**
```
http://localhost:8080/index.html
```

---

## How It Works

1. User enters formula + space group in the frontend
2. Frontend generates the CIF (with loading bar) then opens a new tab
3. Structure page (`structure.html`) shows the result instantly from cache
4. Full Materials Project-style layout with 3D viewer, cell parameters, properties, CIF file

---

## Good Demo Compounds

All of these are in MP-20 and generate well:

| Formula | Space Group | Structure Type |
|---|---|---|
| NaCl | 225 | Rock salt (cubic) |
| GaAs | 216 | Zinc blende |
| BaTiO3 | 99 | Perovskite |
| TiO2 | 136 | Rutile |
| Fe2O3 | 167 | Hematite |
| MgO | 225 | Rock salt (cubic) |

---

## Project Structure

```
Crystext/
├── app.py                     ← Flask backend (port 5000)
├── index.html                 ← Main UI — input page (port 8080)
├── structure.html             ← Structure viewer page (opens in new tab)
├── prompt_refinement.py       ← Auto-corrects user input typos
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
├── HOW_TO_TRAIN.md            ← SFT training instructions
└── training/
    ├── crystext_training.ipynb      ← Kaggle SFT training notebook
    ├── grpo_train.py                ← GRPO/CrysText-RL training script
    ├── crystext_rewards.py          ← Paper-aligned reward function
    ├── crystext_grpo_reward.py      ← TRL-compatible reward wrapper
    ├── dataset_utils.py             ← GRPO dataset preparation
    ├── prepare_grpo_dataset.py      ← Export MP-20 CSV to JSONL
    └── __init__.py
```

---

## Backend API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Single CIF generation |
| `/generate_batch` | POST | Multi-sample generation (N candidates) |
| `/refine_prompt` | POST | Auto-correct formula/space group typos |
| `/evaluate_reward` | POST | Score a CIF using paper reward function |
| `/chat` | POST | Materials science chatbot (Gemini) |
| `/health` | GET | Backend status check |

---

## Features

### Prompt Refinement
Automatically fixes user input errors before generation:
- `nacl` → `NaCl`
- `22O` → `225` (OCR-style typos)
- `rock salt` → `NaCl, 225` (plain English)
- `Barium Titanate` → `BaTiO3`

### GRPO Reward Function (Paper-Aligned)
Scores generated CIFs in 4 stages:
1. CIF parse validity (+0.10)
2. Physical validity — bond distances, volume (+0.20)
3. Composition match — correct elements (+0.20)
4. Structure match vs ground truth (+0.50)

The full GRPO training pipeline (`training/grpo_train.py`) is implemented and ready to run. Training requires 24GB+ VRAM.

### Materials Science Chatbot
Floating chat widget powered by Gemini API. Specialized in crystal structures, space groups, CIF files, and materials science. Add your Gemini API key in `index.html`.

### Structure Viewer Page
Materials Project-inspired layout:
- Large 3D ball-and-stick viewer (FCC/BCC lattice aware)
- Cell parameters panel
- Properties table
- CIF file display (symmetry operations hidden by default)

---

## Testing

```bash
python test_api.py
```

Tests all endpoints — health, refine, generate, batch, reward.

---

## Known Limitations

- Works best for compounds present in MP-20
- Generation takes ~60 seconds per structure
- Requires CUDA GPU — CPU inference is very slow
- GRPO training requires 24GB+ VRAM (pipeline implemented, not yet trained)
