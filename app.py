"""
MediRoute — FastAPI + Gradio for Hugging Face Spaces.

OpenEnv REST API:
  POST /reset   → Observation JSON
  POST /step    → {observation, reward, done, info}
  GET  /state   → dict
  GET  /health  → {"status": "ok"}
  GET  /        → redirect to Gradio UI
  GET  /ui      → Gradio interactive demo
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
import gradio as gr

from env.environment import MediRouteEnv
from env.models import (
    Action,
    PatientClassification,
    ExtractedEntities,
    ResourceAssignment,
)

# ── Global env instance ────────────────────────────────────────────────────────
_env: Optional[MediRouteEnv] = None

def get_env() -> MediRouteEnv:
    global _env
    if _env is None:
        _env = MediRouteEnv(seed=42)
    return _env

# ── Build FastAPI first ────────────────────────────────────────────────────────
api = FastAPI(
    title="MediRoute OpenEnv API",
    description="Emergency Department Triage AI — OpenEnv Hackathon",
    version="1.0.0",
)

@api.get("/health")
async def health():
    return {"status": "ok", "service": "MediRoute"}

@api.get("/")
async def root():
    return RedirectResponse(url="/ui")

@api.post("/reset")
async def reset(request: Request):
    """Reset the environment. Body: {\"task_id\": \"vitals_triage\"} (optional)."""
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        task_id = body.get("task_id", "vitals_triage") if isinstance(body, dict) else "vitals_triage"
        env = get_env()
        obs = env.reset(task_id)
        return JSONResponse(content=obs.model_dump())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@api.post("/step")
async def step(request: Request):
    """Submit an action. Returns observation, reward, done, info."""
    try:
        body = await request.json()
        task_id = body.get("task_id", "vitals_triage")
        env = get_env()

        if task_id == "vitals_triage":
            action_obj = Action(
                task_id=task_id,
                classifications=[PatientClassification(**c) for c in body.get("classifications", [])],
            )
        elif task_id == "clinical_extraction":
            action_obj = Action(
                task_id=task_id,
                extractions=[ExtractedEntities(**e) for e in body.get("extractions", [])],
            )
        else:
            action_obj = Action(
                task_id=task_id,
                assignments=[ResourceAssignment(**a) for a in body.get("assignments", [])],
            )

        obs, reward, done, info = env.step(action_obj)
        return JSONResponse(content={
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": {k: str(v) for k, v in info.items()},
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

@api.get("/state")
async def state():
    """Return current environment state."""
    try:
        return JSONResponse(content=get_env().state())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ── Gradio UI ──────────────────────────────────────────────────────────────────
TASK_LABELS = {
    "vitals_triage":         "🩺 Task 1 — Vitals Triage (Easy)",
    "clinical_extraction":   "📋 Task 2 — Clinical Extraction (Medium)",
    "resource_optimization": "🔧 Task 3 — Resource Optimization (Hard)",
}

_ui_state: Dict[str, Any] = {"env": None, "task_id": None}

def _fmt_vitals(v):
    return (f"HR: {v['heart_rate']} bpm | BP: {v['systolic_bp']}/{v['diastolic_bp']} mmHg | "
            f"RR: {v['respiratory_rate']} br/min\n"
            f"SpO₂: {v['spo2']}% | Temp: {v['temperature']}°C | GCS: {v['gcs']} | Pain: {v['pain_scale']}/10")

def _fmt_patient(p, idx):
    lines = [f"### 🏥 Patient {idx}: {p['patient_id']}",
             f"**Age:** {p['age']} | **Sex:** {p['sex']}",
             f"**Chief Complaint:** _{p['chief_complaint']}_",
             "", "**Vital Signs:**", f"```\n{_fmt_vitals(p['vitals'])}\n```"]
    if p.get("clinical_note"):
        lines += ["", "**Clinical Note:**", f"```\n{p['clinical_note']}\n```"]
    if p.get("arrival_time_minutes") is not None:
        lines += [f"**Arrival:** T+{p['arrival_time_minutes']} min"]
    return "\n".join(lines)

def _fmt_obs(obs_dict):
    parts = []
    for i, p in enumerate(obs_dict.get("patients", []), 1):
        parts.append(_fmt_patient(p, i)); parts.append("---")
    if obs_dict.get("resources"):
        r = obs_dict["resources"]
        parts.append(f"### 🏨 Resources\nBeds: **{r['beds_available']}** | "
                     f"Physicians: **{r['physicians_available']}** | Nurses: **{r['nurses_available']}**")
    return "\n\n".join(parts)

def _template(task_id, obs_dict):
    pts = obs_dict.get("patients", [])
    if task_id == "vitals_triage":
        return json.dumps({"task_id": task_id, "classifications": [
            {"patient_id": p["patient_id"], "esi_level": 3, "rationale": "your reasoning"} for p in pts]}, indent=2)
    elif task_id == "clinical_extraction":
        return json.dumps({"task_id": task_id, "extractions": [
            {"patient_id": p["patient_id"], "diagnoses": [], "medications": [],
             "allergies": [], "procedures": [], "follow_up": []} for p in pts]}, indent=2)
    else:
        return json.dumps({"task_id": task_id, "assignments": [
            {"patient_id": p["patient_id"], "assigned_bed": True, "assigned_physician": True,
             "assigned_nurse": True, "assigned_imaging": "none", "assigned_icu": False, "priority_rank": i+1}
            for i, p in enumerate(pts)]}, indent=2)

def do_reset(task_label):
    task_id = {v: k for k, v in TASK_LABELS.items()}[task_label]
    env = MediRouteEnv(seed=42)
    obs = env.reset(task_id)
    obs_dict = obs.model_dump()
    _ui_state.update({"env": env, "task_id": task_id})
    return _fmt_obs(obs_dict), obs.instructions, _template(task_id, obs_dict), "✅ Reset! Edit JSON and click Submit.", ""

def do_step(action_json):
    if _ui_state["env"] is None:
        return "", "", "⚠️ Reset first.", ""
    try:
        data = json.loads(action_json)
        task_id = _ui_state["task_id"]
        env = _ui_state["env"]
        if task_id == "vitals_triage":
            a = Action(task_id=task_id, classifications=[PatientClassification(**c) for c in data.get("classifications", [])])
        elif task_id == "clinical_extraction":
            a = Action(task_id=task_id, extractions=[ExtractedEntities(**e) for e in data.get("extractions", [])])
        else:
            a = Action(task_id=task_id, assignments=[ResourceAssignment(**r) for r in data.get("assignments", [])])
        _, reward, done, info = env.step(a)
        b = reward.breakdown
        md = (f"## 🏆 Score: **{reward.total:.4f}** / 1.0\n\n"
              f"{'🚨 **CRITICAL ERROR!**' if reward.critical_error else ''}\n\n"
              f"| Component | Score |\n|---|---|\n"
              f"| Accuracy | {b.accuracy:.4f} |\n| Partial Credit | {b.partial_credit:.4f} |\n"
              f"| Extraction F1 | {b.extraction_f1:.4f} |\n| Resource Efficiency | {b.resource_efficiency:.4f} |\n"
              f"| Penalties | {b.penalties:.4f} |\n\n_{reward.feedback}_\n\n"
              f"{'✅ Done!' if done else '🔄 Continue.'}")
        truth = f"```json\n{json.dumps({k: str(v) for k, v in info.items()}, indent=2)}\n```"
        return md, truth, "✅ Submitted!", ""
    except Exception as e:
        return "", "", f"❌ {e}", traceback.format_exc()

with gr.Blocks(title="MediRoute — Emergency Triage OpenEnv") as demo:
    gr.Markdown("# 🏥 MediRoute: Emergency Department Triage OpenEnv\n"
                "**AI triage decision-support — OpenEnv Hackathon 2025**")
    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(choices=list(TASK_LABELS.values()),
                                  value=list(TASK_LABELS.values())[0], label="Select Task")
            reset_btn = gr.Button("🔄 Generate New Episode", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)
        with gr.Column(scale=3):
            patient_display = gr.Markdown(value="_Click Generate New Episode to start._")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📖 Instructions"); instructions_box = gr.Markdown()
        with gr.Column(scale=2):
            gr.Markdown("### ✏️ Action JSON")
            action_box = gr.Code(language="json", lines=18,
                                 value='{\n  "task_id": "vitals_triage",\n  "classifications": []\n}')
            submit_btn = gr.Button("🚀 Submit Action", variant="secondary")
    with gr.Row():
        reward_display = gr.Markdown(value="_Submit an action to see rewards._")
        truth_box = gr.Code(language="json", label="Ground Truth", lines=10)
    error_box = gr.Textbox(label="Error", interactive=False, lines=3)
    reset_btn.click(do_reset, [task_dd], [patient_display, instructions_box, action_box, status_box, error_box])
    submit_btn.click(do_step, [action_box], [reward_display, truth_box, status_box, error_box])
    gr.Markdown("---\n**MediRoute** | OpenEnv Hackathon 2025 | Built for better healthcare AI")

# ── Mount Gradio on FastAPI ────────────────────────────────────────────────────
app = gr.mount_gradio_app(api, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
