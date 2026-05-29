"""
Quick smoke tests for CrysText backend endpoints.

Usage:
  python test_api.py
  python test_api.py --base-url http://localhost:5000 --formula NaCl --spacegroup 225
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


def post_json(url: str, payload: Dict[str, Any], timeout: int = 180) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            return resp.status, json.loads(data) if data else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(text)
        except Exception:
            return exc.code, {"error": text}


def get_json(url: str, timeout: int = 30) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
        return resp.status, json.loads(data) if data else {}


def print_step(name: str) -> None:
    print(f"\n=== {name} ===")


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_health(base_url: str) -> bool:
    print_step("Health")
    try:
        status, data = get_json(f"{base_url}/health")
    except Exception as exc:
        fail(f"Could not reach backend: {exc}")
        return False

    if status != 200:
        fail(f"/health returned {status}: {data}")
        return False

    ok(f"/health returned 200 (device={data.get('device')})")
    return True


def check_generate(base_url: str, formula: str, spacegroup: str) -> Optional[Dict[str, Any]]:
    print_step("Generate (single)")
    payload = {"formula": formula, "spacegroup": spacegroup}
    started = time.time()
    status, data = post_json(f"{base_url}/generate", payload, timeout=600)
    elapsed = time.time() - started

    if status != 200:
        fail(f"/generate returned {status}: {data}")
        return None

    cif = data.get("cif", "")
    validation = data.get("validation", {})
    if not cif:
        fail("No CIF returned")
        return None

    ok(
        f"/generate returned CIF in {elapsed:.1f}s "
        f"(valid={validation.get('valid')}, "
        f"composition_match={validation.get('composition_match')})"
    )
    return data


def check_generate_batch(base_url: str, formula: str, spacegroup: str, num_samples: int) -> Optional[Dict[str, Any]]:
    print_step("Generate Batch")
    payload = {
        "formula": formula,
        "spacegroup": spacegroup,
        "num_samples": num_samples,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    started = time.time()
    status, data = post_json(f"{base_url}/generate_batch", payload, timeout=1200)
    elapsed = time.time() - started

    if status != 200:
        fail(f"/generate_batch returned {status}: {data}")
        return None

    results = data.get("results", [])
    if not results:
        fail("/generate_batch returned no results")
        return None

    valid_count = sum(1 for r in results if r.get("validation", {}).get("valid"))
    ok(
        f"/generate_batch returned {len(results)} samples in {elapsed:.1f}s "
        f"(valid={valid_count})"
    )
    return data


def check_refine(base_url: str, formula: str, spacegroup: str) -> bool:
    print_step("Refine Prompt")
    payload = {"formula": formula, "spacegroup": spacegroup}
    status, data = post_json(f"{base_url}/refine_prompt", payload, timeout=30)

    if status != 200:
        fail(f"/refine_prompt returned {status}: {data}")
        return False

    ok(
        f"/refine_prompt changed={data.get('changed')} "
        f"refined={data.get('refined')}"
    )
    return True


def check_reward(base_url: str, cif: str, formula: str) -> bool:
    print_step("Evaluate Reward")
    payload = {"cif": cif, "formula": formula}
    status, data = post_json(f"{base_url}/evaluate_reward", payload, timeout=120)

    if status != 200:
        fail(f"/evaluate_reward returned {status}: {data}")
        return False

    reward = data.get("reward")
    parsed = data.get("parsed")
    ok(f"/evaluate_reward returned reward={reward}, parsed={parsed}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test CrysText backend endpoints.")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Backend base URL")
    parser.add_argument("--formula", default="NaCl", help="Formula for generation tests")
    parser.add_argument("--spacegroup", default="225", help="Space group number for generation tests")
    parser.add_argument("--num-samples", type=int, default=3, help="Samples for /generate_batch")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    all_ok = True

    if not check_health(base_url):
        return 1

    if not check_refine(base_url, "Nacl", "22O"):
        all_ok = False

    single = check_generate(base_url, args.formula, args.spacegroup)
    if single is None:
        all_ok = False

    batch = check_generate_batch(base_url, args.formula, args.spacegroup, args.num_samples)
    if batch is None:
        all_ok = False

    cif_for_reward = None
    if single and single.get("cif"):
        cif_for_reward = single["cif"]
    elif batch and batch.get("results"):
        cif_for_reward = batch["results"][0].get("cif")

    if cif_for_reward:
        if not check_reward(base_url, cif_for_reward, args.formula):
            all_ok = False
    else:
        fail("No CIF available to test /evaluate_reward")
        all_ok = False

    print_step("Summary")
    if all_ok:
        ok("All endpoint checks completed")
        return 0

    fail("One or more checks failed")
    return 2


if __name__ == "__main__":
    sys.exit(main())
