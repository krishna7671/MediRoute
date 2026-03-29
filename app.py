"""
MediRoute — FastAPI REST API + Gradio UI for HF Spaces.

The OpenEnv REST API endpoints are served by FastAPI,
and the Gradio interactive demo is mounted at /ui and /.
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

import gradio as gr
from env.environment import MediRouteEnv
from env.models import (
    Action,
    PatientClassification,
    ExtractedEntities,
    ResourceAssignment,
)

# ── Shared environment ─────────────────────────────────────────────────────────
_env: Optional[MediRouteEnv] = None

def get_env() -> MediRouteEnv:
    global _env
    if _env is None:
        _env = MediRouteEnv(seed=42)
    return _env

def parse_and_step(body: dict):
    """Shared logic to parse an action dict and step the env."""
    task_id = body.get("task_id", "vitals_triage")
    env = get_env()
    if task_id == "vitals_triage":
        action = Action(
            task_id=task_id,
            classifications=[PatientClassification(**c) for c in body.get("classifications", [])],
        )
    elif task_id == "clinical_extraction":
        action = Action(
            task_id=task_id,
            extractions=[ExtractedEntities(**e) for e in body.get("extractions", [])],
        )
    else:
        action = Action(
            task_id=task_id,
            assignments=[ResourceAssignment(**a) for a in body.get("assignments", [])],
        )
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": {k: str(v) for k, v in info.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI (serves as the main app for HF Spaces)
# ══════════════════════════════════════════════════════════════════════════════

TASK_LABELS = {
    "vitals_triage":         "🩺 Task 1 — Vitals Triage (Easy)",
    "clinical_extraction":   "📋 Task 2 — Clinical Extraction (Medium)",
    "resource_optimization": "🔧 Task 3 — Resource Optimization (Hard)",
}
_ui: Dict[str, Any] = {"env": None, "task_id": None}


def _fv(v):
    return (f"HR: {v['heart_rate']} bpm | BP: {v['systolic_bp']}/{v['diastolic_bp']} mmHg | "
            f"RR: {v['respiratory_rate']}\n"
            f"SpO2: {v['spo2']}% | Temp: {v['temperature']}C | GCS: {v['gcs']} | Pain: {v['pain_scale']}/10")


def _fp(p, i):
    lines = [f"### Patient {i}: {p['patient_id']}",
             f"**Age:** {p['age']} | **Sex:** {p['sex']}",
             f"**Complaint:** _{p['chief_complaint']}_",
             f"```\n{_fv(p['vitals'])}\n```"]
    if p.get("clinical_note"):
        lines += [f"**Note:**\n```\n{p['clinical_note']}\n```"]
    return "\n".join(lines)


def _fo(obs):
    parts = [_fp(p, i) + "\n---" for i, p in enumerate(obs.get("patients", []), 1)]
    if obs.get("resources"):
        r = obs["resources"]
        parts.append(f"**Resources:** Beds={r['beds_available']} Physicians={r['physicians_available']} "
                     f"Nurses={r['nurses_available']}")
    return "\n\n".join(parts)


def _tmpl(tid, obs):
    pts = obs.get("patients", [])
    if tid == "vitals_triage":
        return json.dumps({"task_id": tid, "classifications": [
            {"patient_id": p["patient_id"], "esi_level": 3, "rationale": ""} for p in pts]}, indent=2)
    elif tid == "clinical_extraction":
        return json.dumps({"task_id": tid, "extractions": [
            {"patient_id": p["patient_id"], "diagnoses": [], "medications": [],
             "allergies": [], "procedures": [], "follow_up": []} for p in pts]}, indent=2)
    return json.dumps({"task_id": tid, "assignments": [
        {"patient_id": p["patient_id"], "assigned_bed": True, "assigned_physician": True,
         "assigned_nurse": True, "assigned_imaging": "none", "assigned_icu": False, "priority_rank": i+1}
        for i, p in enumerate(pts)]}, indent=2)


def do_reset(label):
    tid = {v: k for k, v in TASK_LABELS.items()}[label]
    env = MediRouteEnv(seed=42)
    obs = env.reset(tid)
    od = obs.model_dump()
    _ui.update({"env": env, "task_id": tid})
    return _fo(od), obs.instructions, _tmpl(tid, od), "Reset OK", ""


def do_step(aj):
    if not _ui["env"]:
        return "", "", "Reset first", ""
    try:
        d = json.loads(aj)
        tid, env = _ui["task_id"], _ui["env"]
        if tid == "vitals_triage":
            a = Action(task_id=tid, classifications=[PatientClassification(**c) for c in d.get("classifications", [])])
        elif tid == "clinical_extraction":
            a = Action(task_id=tid, extractions=[ExtractedEntities(**e) for e in d.get("extractions", [])])
        else:
            a = Action(task_id=tid, assignments=[ResourceAssignment(**r) for r in d.get("assignments", [])])
        _, rw, done, info = env.step(a)
        b = rw.breakdown
        md = (f"## Score: **{rw.total:.4f}**\n\n"
              f"| Component | Score |\n|---|---|\n"
              f"| Accuracy | {b.accuracy:.4f} |\n| Partial | {b.partial_credit:.4f} |\n"
              f"| F1 | {b.extraction_f1:.4f} |\n| Resources | {b.resource_efficiency:.4f} |\n"
              f"| Penalties | {b.penalties:.4f} |\n\n_{rw.feedback}_")
        return md, json.dumps({k: str(v) for k, v in info.items()}, indent=2), "Submitted!", ""
    except Exception as e:
        return "", "", str(e), traceback.format_exc()


# ── Gradio API functions (exposed as Gradio API endpoints) ─────────────────────

def api_reset(task_id: str = "vitals_triage"):
    """Gradio API: POST /reset via Gradio's built-in API."""
    env = get_env()
    obs = env.reset(task_id)
    return json.dumps(obs.model_dump())


def api_step(action_json: str):
    """Gradio API: POST /step via Gradio's built-in API."""
    body = json.loads(action_json)
    result = parse_and_step(body)
    return json.dumps(result)


def api_state():
    """Gradio API: GET /state via Gradio's built-in API."""
    return json.dumps(get_env().state())


# ── Build the Gradio Blocks app ────────────────────────────────────────────────

with gr.Blocks(title="MediRoute — Emergency Triage OpenEnv") as demo:
    gr.Markdown("# 🏥 MediRoute: Emergency Triage OpenEnv\n"
                "**AI triage decision-support — OpenEnv Hackathon 2025**")

    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(choices=list(TASK_LABELS.values()),
                                  value=list(TASK_LABELS.values())[0], label="Task")
            reset_btn = gr.Button("🔄 New Episode", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=3):
            patients_md = gr.Markdown(value="_Click New Episode._")

    with gr.Row():
        with gr.Column(scale=1):
            instr_md = gr.Markdown(label="Instructions")
        with gr.Column(scale=2):
            action_code = gr.Code(language="json", lines=16,
                                  value='{"task_id":"vitals_triage","classifications":[]}')
            step_btn = gr.Button("🚀 Submit", variant="secondary")

    with gr.Row():
        reward_md = gr.Markdown()
        truth_code = gr.Code(language="json", lines=8)
    err = gr.Textbox(label="Error", interactive=False, lines=2)

    reset_btn.click(do_reset, [task_dd], [patients_md, instr_md, action_code, status, err])
    step_btn.click(do_step, [action_code], [reward_md, truth_code, status, err])

    # ── Hidden Gradio API interfaces for the OpenEnv checker ──────────────────
    # These create proper /api/reset, /api/step, /api/state endpoints
    reset_iface = gr.Interface(fn=api_reset, inputs=gr.Textbox(visible=False),
                                outputs=gr.JSON(), title="reset", api_name="reset")
    step_iface = gr.Interface(fn=api_step, inputs=gr.Textbox(visible=False),
                               outputs=gr.JSON(), title="step", api_name="step")
    state_iface = gr.Interface(fn=api_state, inputs=[], outputs=gr.JSON(),
                                title="state", api_name="state")


# ══════════════════════════════════════════════════════════════════════════════
# ALSO: Standalone FastAPI app so uvicorn works too
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

fastapi_app = FastAPI()

@fastapi_app.get("/")
async def root_redirect():
    return RedirectResponse(url="/ui")

@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}

@fastapi_app.post("/reset")
@fastapi_app.post("/reset/")
@fastapi_app.post("/{p1}/reset")
@fastapi_app.post("/{p1}/{p2}/reset")
@fastapi_app.post("/api/reset")
@fastapi_app.post("/api/reset/")
@fastapi_app.post("/run/reset")
@fastapi_app.post("/run/reset/")
async def reset_endpoint(request: Request, p1: str = None, p2: str = None):
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

@fastapi_app.post("/step")
@fastapi_app.post("/step/")
@fastapi_app.post("/{p1}/step")
@fastapi_app.post("/{p1}/{p2}/step")
@fastapi_app.post("/api/step")
@fastapi_app.post("/api/step/")
@fastapi_app.post("/run/step")
@fastapi_app.post("/run/step/")
async def step_endpoint(request: Request, p1: str = None, p2: str = None):
    try:
        body = await request.json()
        result = parse_and_step(body)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@fastapi_app.get("/state")
@fastapi_app.get("/state/")
@fastapi_app.get("/{p1}/state")
@fastapi_app.get("/{p1}/{p2}/state")
@fastapi_app.get("/api/state")
@fastapi_app.get("/api/state/")
@fastapi_app.get("/run/state")
@fastapi_app.get("/run/state/")
async def state_endpoint(p1: str = None, p2: str = None):
    try:
        return JSONResponse(content=get_env().state())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount Gradio onto FastAPI cleanly under /ui to avoid route conflicts
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
