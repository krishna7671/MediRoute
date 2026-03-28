"""
inference.py — Baseline inference script for MediRoute OpenEnv.

Runs a rule-based / LLM agent against all three tasks and reports scores.
Can be used with the local environment or the HF Spaces REST API.

Usage (local):
    python inference.py

Usage (against deployed Space):
    python inference.py --api-url https://vega952-mediroute7.hf.space

Usage (with OpenAI LLM baseline):
    OPENAI_API_KEY=sk-... python inference.py --llm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Rule-based agent (no API key needed) ──────────────────────────────────────

def rule_based_action(obs: dict) -> dict:
    """
    Simple heuristic agent:
    - Task 1: classifies ESI based on SpO2 and heart rate thresholds
    - Task 2: returns empty extractions (zero-shot baseline)
    - Task 3: assigns top-3 critical patients to all resources
    """
    task_id = obs["task_id"]
    patients = obs.get("patients", [])

    if task_id == "vitals_triage":
        classifications = []
        for p in patients:
            v = p["vitals"]
            hr, spo2, gcs = v["heart_rate"], v["spo2"], v["gcs"]
            sbp = v["systolic_bp"]
            # Simple clinical rules
            if gcs < 9 or spo2 < 90 or (hr > 140 and sbp < 90):
                esi = 1
            elif spo2 < 94 or hr > 130 or sbp < 100:
                esi = 2
            elif hr > 110 or spo2 < 96:
                esi = 3
            elif v["pain_scale"] >= 6:
                esi = 4
            else:
                esi = 5
            classifications.append({
                "patient_id": p["patient_id"],
                "esi_level": esi,
                "rationale": f"HR={hr}, SpO2={spo2}%, GCS={gcs}, SBP={sbp}",
            })
        return {"task_id": task_id, "classifications": classifications}

    elif task_id == "clinical_extraction":
        return {"task_id": task_id, "extractions": [
            {"patient_id": p["patient_id"], "diagnoses": [], "medications": [],
             "allergies": [], "procedures": [], "follow_up": []}
            for p in patients
        ]}

    else:  # resource_optimization
        # Sort by simplistic urgency (pain + inverted spo2)
        def urgency(p):
            v = p["vitals"]
            return -(v["pain_scale"] + (100 - v["spo2"]))
        sorted_pts = sorted(patients, key=urgency)
        resources = obs.get("resources", {})
        beds = resources.get("beds_available", 3)
        physicians = resources.get("physicians_available", 2)
        nurses = resources.get("nurses_available", 3)
        assignments = []
        for rank, p in enumerate(sorted_pts, 1):
            assignments.append({
                "patient_id": p["patient_id"],
                "assigned_bed": rank <= beds,
                "assigned_physician": rank <= physicians,
                "assigned_nurse": rank <= nurses,
                "assigned_imaging": "none",
                "assigned_icu": False,
                "priority_rank": rank,
            })
        return {"task_id": task_id, "assignments": assignments}


# ── Local runner ───────────────────────────────────────────────────────────────

def run_local():
    """Run inference using the local Python environment."""
    from env.environment import MediRouteEnv
    from env.models import Action, PatientClassification, ExtractedEntities, ResourceAssignment

    tasks = ["vitals_triage", "clinical_extraction", "resource_optimization"]
    results = []

    for task_id in tasks:
        env = MediRouteEnv(seed=42)
        obs = env.reset(task_id)
        obs_dict = obs.model_dump()
        action_dict = rule_based_action(obs_dict)

        # Convert to typed Action
        if task_id == "vitals_triage":
            action = Action(task_id=task_id, classifications=[
                PatientClassification(**c) for c in action_dict["classifications"]
            ])
        elif task_id == "clinical_extraction":
            action = Action(task_id=task_id, extractions=[
                ExtractedEntities(**e) for e in action_dict["extractions"]
            ])
        else:
            action = Action(task_id=task_id, assignments=[
                ResourceAssignment(**a) for a in action_dict["assignments"]
            ])

        _, reward, done, info = env.step(action)
        results.append({
            "task": task_id,
            "score": round(reward.total, 4),
            "feedback": reward.feedback,
        })
        print(f"  {task_id:30s}  score={reward.total:.4f}  done={done}")

    return results


# ── REST API runner (against deployed Space) ───────────────────────────────────

def run_api(api_url: str):
    """Run inference against a deployed MediRoute REST API."""
    try:
        import httpx
    except ImportError:
        os.system(f"{sys.executable} -m pip install httpx -q")
        import httpx

    tasks = ["vitals_triage", "clinical_extraction", "resource_optimization"]
    results = []

    with httpx.Client(base_url=api_url, timeout=30) as client:
        for task_id in tasks:
            # Reset
            r_reset = client.post("/reset", json={"task_id": task_id})
            r_reset.raise_for_status()
            obs = r_reset.json()

            # Generate action
            action = rule_based_action(obs)

            # Step
            r_step = client.post("/step", json=action)
            r_step.raise_for_status()
            result = r_step.json()

            score = result["reward"]["total"]
            feedback = result["reward"]["feedback"]
            results.append({"task": task_id, "score": round(score, 4), "feedback": feedback})
            print(f"  {task_id:30s}  score={score:.4f}  done={result['done']}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MediRoute OpenEnv baseline inference")
    parser.add_argument("--api-url", type=str, default=None,
                        help="URL of deployed MediRoute API (e.g. https://vega952-mediroute7.hf.space)")
    parser.add_argument("--output", type=str, default="inference_results.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    print("=" * 55)
    print("  MediRoute OpenEnv — Baseline Inference")
    print("=" * 55)

    if args.api_url:
        print(f"Mode: REST API ({args.api_url})\n")
        results = run_api(args.api_url)
    else:
        print("Mode: Local Python environment\n")
        results = run_local()

    mean_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Mean score across all tasks: {mean_score:.4f}")
    print("=" * 55)

    with open(args.output, "w") as f:
        json.dump({"results": results, "mean_score": round(mean_score, 4)}, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
