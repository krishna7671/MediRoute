"""
MediRoute — Baseline Inference Script.

Uses the OpenAI API to run a language model against all three MediRoute tasks
and produce reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=your-key
    python baseline.py
    python baseline.py --model gpt-4o --episodes 5
    python baseline.py --task vitals_triage --episodes 3
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from env.environment import MediRouteEnv
from env.models import Action, PatientClassification, ExtractedEntities, ResourceAssignment

app = typer.Typer(help="MediRoute Baseline Inference — runs LLM against all 3 tasks.")
console = Console()

# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────

def build_vitals_prompt(obs_dict: dict) -> str:
    patients = obs_dict["patients"]
    lines = ["PATIENT CASES FOR TRIAGE:\n"]
    for p in patients:
        v = p["vitals"]
        lines.append(
            f"Patient {p['patient_id']}:\n"
            f"  Age: {p['age']} | Sex: {p['sex']}\n"
            f"  Chief Complaint: {p['chief_complaint']}\n"
            f"  HR: {v['heart_rate']} bpm | BP: {v['systolic_bp']}/{v['diastolic_bp']} mmHg\n"
            f"  RR: {v['respiratory_rate']} br/min | SpO2: {v['spo2']}%\n"
            f"  Temp: {v['temperature']}°C | GCS: {v['gcs']} | Pain: {v['pain_scale']}/10\n"
        )
    return "\n".join(lines)


def build_clinical_prompt(obs_dict: dict) -> str:
    patients = obs_dict["patients"]
    lines = ["CLINICAL NOTES FOR EXTRACTION:\n"]
    for p in patients:
        lines.append(f"--- Patient {p['patient_id']} ---\n{p.get('clinical_note', '')}\n")
    return "\n".join(lines)


def build_resource_prompt(obs_dict: dict) -> str:
    patients = obs_dict["patients"]
    resources = obs_dict.get("resources", {})
    lines = ["PATIENT SURGE — RESOURCE ASSIGNMENT REQUIRED\n"]
    lines.append(
        f"Available: {resources.get('beds_available', '?')} beds | "
        f"{resources.get('physicians_available', '?')} physicians | "
        f"{resources.get('nurses_available', '?')} nurses | "
        f"{resources.get('ct_scanners_available', '?')} CT | "
        f"{resources.get('xray_available', '?')} X-ray | "
        f"{resources.get('icu_beds_available', '?')} ICU beds\n"
    )
    for i, p in enumerate(patients, 1):
        v = p["vitals"]
        lines.append(
            f"Patient {p['patient_id']} (arrived T+{p.get('arrival_time_minutes', '?')} min):\n"
            f"  Age: {p['age']} | Complaint: {p['chief_complaint']}\n"
            f"  HR: {v['heart_rate']} | BP: {v['systolic_bp']}/{v['diastolic_bp']} | "
            f"SpO2: {v['spo2']}% | GCS: {v['gcs']}\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Action parsers
# ─────────────────────────────────────────────

def parse_vitals_action(response_text: str, obs_dict: dict) -> Action:
    """Extract classifications from LLM JSON response."""
    try:
        data = json.loads(response_text)
        clfs = data.get("classifications", [])
        return Action(
            task_id="vitals_triage",
            classifications=[PatientClassification(**c) for c in clfs],
        )
    except Exception:
        # Fallback: classify all patients as ESI 3 (middle-ground)
        patients = obs_dict.get("patients", [])
        return Action(
            task_id="vitals_triage",
            classifications=[
                PatientClassification(patient_id=p["patient_id"], esi_level=3, rationale="fallback")
                for p in patients
            ],
        )


def parse_clinical_action(response_text: str, obs_dict: dict) -> Action:
    try:
        data = json.loads(response_text)
        exts = data.get("extractions", [])
        return Action(
            task_id="clinical_extraction",
            extractions=[ExtractedEntities(**e) for e in exts],
        )
    except Exception:
        patients = obs_dict.get("patients", [])
        return Action(
            task_id="clinical_extraction",
            extractions=[
                ExtractedEntities(patient_id=p["patient_id"])
                for p in patients
            ],
        )


def parse_resource_action(response_text: str, obs_dict: dict) -> Action:
    try:
        data = json.loads(response_text)
        assigns = data.get("assignments", [])
        return Action(
            task_id="resource_optimization",
            assignments=[ResourceAssignment(**a) for a in assigns],
        )
    except Exception:
        patients = obs_dict.get("patients", [])
        return Action(
            task_id="resource_optimization",
            assignments=[
                ResourceAssignment(
                    patient_id=p["patient_id"],
                    assigned_bed=True,
                    assigned_physician=True,
                    assigned_nurse=True,
                    assigned_imaging="none",
                    assigned_icu=False,
                    priority_rank=i + 1,
                )
                for i, p in enumerate(patients)
            ],
        )


# ─────────────────────────────────────────────
# Task runners
# ─────────────────────────────────────────────

def run_task(
    client: OpenAI,
    model: str,
    task_id: str,
    n_episodes: int,
    seed: int = 42,
) -> list[float]:
    """Run n_episodes of a given task and return episode scores."""
    scores = []
    prompt_builders = {
        "vitals_triage": build_vitals_prompt,
        "clinical_extraction": build_clinical_prompt,
        "resource_optimization": build_resource_prompt,
    }
    action_parsers = {
        "vitals_triage": parse_vitals_action,
        "clinical_extraction": parse_clinical_action,
        "resource_optimization": parse_resource_action,
    }
    instructions = {
        "vitals_triage": (
            "You are an ED triage AI. Classify each patient's ESI level (1-5). "
            "Return ONLY valid JSON with a `classifications` array. No extra text."
        ),
        "clinical_extraction": (
            "You are a clinical NLP system. Extract entities from clinical notes. "
            "Return ONLY valid JSON with an `extractions` array. No extra text."
        ),
        "resource_optimization": (
            "You are an ED resource allocation AI. Assign hospital resources optimally. "
            "Return ONLY valid JSON with an `assignments` array. No extra text. "
            "Do NOT exceed the stated resource limits."
        ),
    }

    env = MediRouteEnv(task_id=task_id, seed=seed)

    for ep in range(n_episodes):
        obs = env.reset(task_id)
        obs_dict = obs.model_dump()

        system_msg = instructions[task_id]
        user_msg = f"{obs.instructions}\n\n{prompt_builders[task_id](obs_dict)}"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                seed=seed + ep,
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            console.print(f"[red]API error on episode {ep+1}: {e}[/red]")
            scores.append(0.0)
            continue

        action = action_parsers[task_id](response_text, obs_dict)

        # For resource optimization task (multi-step), step up to max_steps
        task_obj = env._task
        max_steps = getattr(task_obj, "max_steps", 1)
        episode_score = 0.0

        for step_i in range(max_steps):
            _, reward, done, info = env.step(action)
            episode_score = info.get("cumulative_score", reward.total)
            if done:
                break

        scores.append(episode_score)
        console.print(
            f"  Episode {ep+1}/{n_episodes}: score={episode_score:.4f} "
            f"| critical={info.get('critical_error', reward.critical_error)}"
        )
        time.sleep(0.5)  # Rate limiting

    return scores


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

@app.command()
def main(
    model: str = typer.Option(
        None, "--model", "-m",
        help="OpenAI model name. Defaults to OPENAI_MODEL env var or gpt-4o-mini"
    ),
    episodes: int = typer.Option(5, "--episodes", "-e", help="Episodes per task"),
    task: Optional[str] = typer.Option(
        None, "--task", "-t",
        help="Run single task: vitals_triage | clinical_extraction | resource_optimization"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
):
    """Run baseline inference against MediRoute environment tasks."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
        raise typer.Exit(1)

    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    tasks_to_run = (
        ["vitals_triage", "clinical_extraction", "resource_optimization"]
        if task is None
        else [task]
    )

    console.print(Panel.fit(
        f"[bold cyan]MediRoute Baseline Inference[/bold cyan]\n"
        f"Model: [yellow]{model_name}[/yellow] | Episodes: {episodes} | Seed: {seed}",
        border_style="cyan"
    ))

    results = {}
    for t in tasks_to_run:
        difficulty_map = {
            "vitals_triage": "Easy",
            "clinical_extraction": "Medium",
            "resource_optimization": "Hard",
        }
        console.print(f"\n[bold green]Running: {t} ({difficulty_map.get(t, '')})[/bold green]")
        scores = run_task(client, model_name, t, episodes, seed)
        results[t] = scores

    # Summary table
    table = Table(title="\n📊 MediRoute Baseline Results", border_style="cyan")
    table.add_column("Task", style="bold")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Episodes", justify="right")
    table.add_column("Mean Score", justify="right", style="green")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for t, scores in results.items():
        diff = {"vitals_triage": "Easy", "clinical_extraction": "Medium", "resource_optimization": "Hard"}
        table.add_row(
            t,
            diff.get(t, ""),
            str(len(scores)),
            f"{sum(scores)/max(len(scores),1):.4f}",
            f"{min(scores):.4f}",
            f"{max(scores):.4f}",
        )

    console.print(table)

    # Save results
    output = {
        "model": model_name,
        "seed": seed,
        "episodes": episodes,
        "results": {t: {"scores": s, "mean": sum(s)/max(len(s),1)} for t, s in results.items()},
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    console.print("\n[dim]Results saved to baseline_results.json[/dim]")


if __name__ == "__main__":
    app()
