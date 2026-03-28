"""
MediRoute — FastAPI + Gradio for Hugging Face Spaces.

Exposes the full OpenEnv REST API:
  POST /reset   → Observation
  POST /step    → {observation, reward, done, info}
  GET  /state   → dict
  GET  /        → Gradio interactive demo UI
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import gradio as gr

from env.environment import MediRouteEnv
from env.models import (
    Action,
    PatientClassification,
    ExtractedEntities,
    ResourceAssignment,
)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediRoute OpenEnv API",
    description="Emergency Department Triage AI Environment — OpenEnv Hackathon",
    version="1.0.0",
)

# Shared environment instance (thread-safe enough for single-user demo)
_env: Optional[MediRouteEnv] = None


def _get_env() -> MediRouteEnv:
    global _env
    if _env is None:
        _env = MediRouteEnv(seed=42)
    return _env


# ── OpenEnv REST Endpoints ─────────────────────────────────────────────────────

@app.post("/reset")
async def reset(body: Dict[str, Any] = Body(default={"task_id": "vitals_triage"})):
    """Reset the environment for a given task. Returns the first Observation."""
    task_id = body.get("task_id", "vitals_triage")
    try:
        env = _get_env()
        obs = env.reset(task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: Dict[str, Any] = Body(...)):
    """Submit an action. Returns {observation, reward, done, info}."""
    try:
        env = _get_env()
        task_id = action.get("task_id", "vitals_triage")

        if task_id == "vitals_triage":
            action_obj = Action(
                task_id=task_id,
                classifications=[
                    PatientClassification(**c)
                    for c in action.get("classifications", [])
                ],
            )
        elif task_id == "clinical_extraction":
            action_obj = Action(
                task_id=task_id,
                extractions=[
                    ExtractedEntities(**e)
                    for e in action.get("extractions", [])
                ],
            )
        elif task_id == "resource_optimization":
            action_obj = Action(
                task_id=task_id,
                assignments=[
                    ResourceAssignment(**a)
                    for a in action.get("assignments", [])
                ],
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

        obs, reward, done, info = env.step(action_obj)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": {k: str(v) for k, v in info.items()},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Return the current internal state of the environment."""
    try:
        env = _get_env()
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "MediRoute OpenEnv"}


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1200px; margin: auto; }
#title { text-align: center; }
"""

TASK_LABELS = {
    "vitals_triage":        "🩺 Task 1 — Vitals Triage (Easy)",
    "clinical_extraction":  "📋 Task 2 — Clinical Extraction (Medium)",
    "resource_optimization":"🔧 Task 3 — Resource Optimization (Hard)",
}

ui_env_state: Dict[str, Any] = {"env": None, "obs": None, "task_id": None}


def _format_vitals(v: dict) -> str:
    return (
        f"HR: {v['heart_rate']} bpm | BP: {v['systolic_bp']}/{v['diastolic_bp']} mmHg | "
        f"RR: {v['respiratory_rate']} br/min\n"
        f"SpO₂: {v['spo2']}% | Temp: {v['temperature']}°C | "
        f"GCS: {v['gcs']} | Pain: {v['pain_scale']}/10"
    )


def _format_patient(p: dict, idx: int) -> str:
    lines = [
        f"### 🏥 Patient {idx}: {p['patient_id']}",
        f"**Age:** {p['age']} | **Sex:** {p['sex']}",
        f"**Chief Complaint:** _{p['chief_complaint']}_",
        "", "**Vital Signs:**", f"```\n{_format_vitals(p['vitals'])}\n```",
    ]
    if p.get("clinical_note"):
        lines += ["", "**Clinical Note:**", f"```\n{p['clinical_note']}\n```"]
    if p.get("arrival_time_minutes") is not None:
        lines += [f"**Arrival:** T+{p['arrival_time_minutes']} min"]
    return "\n".join(lines)


def _display_obs(obs_dict: dict) -> str:
    parts = []
    for i, p in enumerate(obs_dict.get("patients", []), 1):
        parts.append(_format_patient(p, i))
        parts.append("---")
    if obs_dict.get("resources"):
        r = obs_dict["resources"]
        parts.append(
            f"### 🏨 Available Resources\n"
            f"Beds: **{r['beds_available']}** | Physicians: **{r['physicians_available']}** | "
            f"Nurses: **{r['nurses_available']}**\n"
            f"CT: **{r['ct_scanners_available']}** | X-Ray: **{r['xray_available']}** | "
            f"ICU: **{r['icu_beds_available']}**"
        )
    return "\n\n".join(parts)


def _build_template(task_id: str, obs_dict: dict) -> str:
    patients = obs_dict.get("patients", [])
    if task_id == "vitals_triage":
        action = {"task_id": task_id, "classifications": [
            {"patient_id": p["patient_id"], "esi_level": 3, "rationale": "your reasoning"} for p in patients
        ]}
    elif task_id == "clinical_extraction":
        action = {"task_id": task_id, "extractions": [
            {"patient_id": p["patient_id"], "diagnoses": [], "medications": [], "allergies": [], "procedures": [], "follow_up": []} for p in patients
        ]}
    else:
        action = {"task_id": task_id, "assignments": [
            {"patient_id": p["patient_id"], "assigned_bed": True, "assigned_physician": True, "assigned_nurse": True, "assigned_imaging": "none", "assigned_icu": False, "priority_rank": i+1} for i, p in enumerate(patients)
        ]}
    return json.dumps(action, indent=2)


def do_reset(task_label: str):
    task_id = {v: k for k, v in TASK_LABELS.items()}[task_label]
    env = MediRouteEnv(seed=42)
    obs = env.reset(task_id)
    obs_dict = obs.model_dump()
    ui_env_state.update({"env": env, "obs": obs_dict, "task_id": task_id})
    return (
        _display_obs(obs_dict),
        obs.instructions,
        _build_template(task_id, obs_dict),
        "✅ Environment reset! Edit the JSON and click Submit.",
        "",
    )


def do_step(action_json_str: str):
    if ui_env_state["env"] is None:
        return "", "", "⚠️ Reset first.", ""
    try:
        data = json.loads(action_json_str)
        task_id = ui_env_state["task_id"]
        env = ui_env_state["env"]

        if task_id == "vitals_triage":
            action_obj = Action(task_id=task_id, classifications=[PatientClassification(**c) for c in data.get("classifications", [])])
        elif task_id == "clinical_extraction":
            action_obj = Action(task_id=task_id, extractions=[ExtractedEntities(**e) for e in data.get("extractions", [])])
        else:
            action_obj = Action(task_id=task_id, assignments=[ResourceAssignment(**a) for a in data.get("assignments", [])])

        _, reward, done, info = env.step(action_obj)
        b = reward.breakdown
        reward_md = (
            f"## 🏆 Score: **{reward.total:.4f}** / 1.0\n\n"
            f"{'🚨 **CRITICAL ERROR** — Life-threatening misclassification!' if reward.critical_error else ''}\n\n"
            f"### Breakdown\n| Component | Score |\n|---|---|\n"
            f"| Accuracy | {b.accuracy:.4f} |\n"
            f"| Partial Credit | {b.partial_credit:.4f} |\n"
            f"| Extraction F1 | {b.extraction_f1:.4f} |\n"
            f"| Resource Efficiency | {b.resource_efficiency:.4f} |\n"
            f"| Penalties | {b.penalties:.4f} |\n\n"
            f"### Feedback\n_{reward.feedback}_\n\n"
            f"{'✅ Episode complete!' if done else '🔄 Continue refining.'}"
        )
        truth_md = f"```json\n{json.dumps({k: str(v) for k, v in info.items()}, indent=2)}\n```"
        return reward_md, truth_md, "✅ Action submitted!", ""
    except Exception as e:
        return "", "", f"❌ Error: {e}", traceback.format_exc()


with gr.Blocks(title="MediRoute — Emergency Triage OpenEnv", css=CSS) as demo:
    gr.Markdown(
        "# 🏥 MediRoute: Emergency Department Triage OpenEnv\n"
        "**AI-powered triage decision-support environment — OpenEnv Hackathon.**",
        elem_id="title",
    )
    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(choices=list(TASK_LABELS.values()), value=list(TASK_LABELS.values())[0], label="Select Task")
            reset_btn = gr.Button("🔄 Generate New Episode", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)
        with gr.Column(scale=3):
            patient_display = gr.Markdown(value="_Click Generate New Episode to start._")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📖 Task Instructions")
            instructions_box = gr.Markdown(value="_Instructions will appear here._")
        with gr.Column(scale=2):
            gr.Markdown("### ✏️ Your Action (JSON)")
            action_box = gr.Code(language="json", label="Action JSON", lines=20, value='{\n  "task_id": "vitals_triage",\n  "classifications": []\n}')
            submit_btn = gr.Button("🚀 Submit Action", variant="secondary")

    with gr.Row():
        with gr.Column(scale=2):
            reward_display = gr.Markdown(value="_Reward will appear after submitting._")
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Ground Truth")
            truth_box = gr.Code(language="json", label="Info", lines=12)

    error_box = gr.Textbox(label="Error Traceback", interactive=False, lines=5)

    reset_btn.click(fn=do_reset, inputs=[task_dd], outputs=[patient_display, instructions_box, action_box, status_box, error_box])
    submit_btn.click(fn=do_step, inputs=[action_box], outputs=[reward_display, truth_box, status_box, error_box])

    gr.Markdown("---\n**MediRoute** | OpenEnv Hackathon 2025 | Built with ❤️ for better healthcare AI")

# Mount Gradio on FastAPI at root
app = gr.mount_gradio_app(app, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
