#!/usr/bin/env python
"""
CrysText-RL: GRPO training on top of SFT (QLoRA) checkpoint.

Paper: Group Relative Policy Optimization with n completions per prompt,
deterministic pymatgen rewards, KL penalty vs reference policy.

Usage (from repo root):
  python training/grpo_train.py --dataset_jsonl data/mp20_grpo_train.jsonl
  python training/grpo_train.py --csv_path DiffCSP/data/mp_20/train.csv --start 0 --end 2000

Kaggle / multi-GPU:
  accelerate launch training/grpo_train.py --dataset_jsonl ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.crystext_grpo_reward import crystext_reward_func
from training.dataset_utils import default_csv_path, load_jsonl, load_mp20_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrysText-RL GRPO training")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-v0.3")
    p.add_argument(
        "--sft_adapter",
        default=os.getenv("CRYSTEXT_SFT_ADAPTER", "Charanya-2026/crystext-mistral-27k"),
        help="HuggingFace id or path to SFT LoRA (CrysText supervised model)",
    )
    p.add_argument("--csv_path", default=None, help="MP-20 train.csv (DiffCSP layout)")
    p.add_argument("--dataset_jsonl", default=None, help="Prebuilt JSONL with prompt, formula, reference_cif")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None, help="Cap dataset size for debugging")
    p.add_argument("--output_dir", default="outputs/crystext-rl-grpo")
    p.add_argument("--hub_repo", default=None, help="Push adapter to HF if set")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_generations", type=int, default=6, help="Paper: n=6 per prompt")
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=1536)
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient vs reference")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    return p.parse_args()


def load_policy_model(base_model_id: str, sft_adapter: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    return model, tokenizer


def build_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset_jsonl:
        ds = load_jsonl(args.dataset_jsonl, limit=args.max_samples)
    else:
        csv_path = args.csv_path or default_csv_path()
        end = args.end
        if args.max_samples is not None and end is None:
            end = args.start + args.max_samples
        ds = load_mp20_csv(csv_path, start=args.start, end=end)
        if args.max_samples is not None and len(ds) > args.max_samples:
            ds = ds.select(range(args.max_samples))
    if len(ds) == 0:
        raise ValueError("Empty training dataset")
    print(f"GRPO dataset size: {len(ds)}")
    return ds


def main() -> None:
    args = parse_args()

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit("Install TRL: pip install 'trl>=0.17.0'") from exc

    train_dataset = build_dataset(args)
    model, tokenizer = load_policy_model(args.base_model, args.sft_adapter)

    # Continue training existing SFT LoRA (CrysText -> CrysText-RL), not a new adapter stack.
    model.train()
    for _name, param in model.named_parameters():
        if param.requires_grad is False and "lora" in _name.lower():
            param.requires_grad = True

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
        # GRPO-specific (paper-aligned defaults)
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        scale_rewards=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=crystext_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting CrysText-RL GRPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

    if args.hub_repo:
        model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)
        print(f"Pushed to {args.hub_repo}")


if __name__ == "__main__":
    main()
