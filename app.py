"""
MediRoute — Gradio Web Interface for Hugging Face Spaces.

Provides an interactive demo where users can explore all three tasks,
experiment with different patient scenarios, and observe reward signals.
"""
from __future__ import annotations

import json
import os
import traceback

import gradio as gr

from env.environment import MediRouteEnv
from env.models import (
    Action,
    PatientClassification,
    ExtractedEntities,
    ResourceAssignment,
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def format_vitals(v: dict) -> str:
    return (
        f"❤️  HR: {v['heart_rate']} bpm  |  "
        f"🩸 BP: {v['systolic_bp']}/{v['diastolic_bp']} mmHg  |  "
        f"🌬️  RR: {v['respiratory_rate']} br/min\n"
        f"💉 SpO₂: {v['spo2']}%  |  "
        f"🌡️  Temp: {v['temperature']}°C  |  "
        f"🧠 GCS: {v['gcs']}  |  "
        f"😣 Pain: {v['pain_scale']}/10"
    )


def format_patient(p: dict, idx: int) -> str:
    lines = [
        f"### 🏥 Patient {idx}: {p['patient_id']}",
        f"**Age:** {p['age']} | **Sex:** {p['sex']}",
        f"**Chief Complaint:** _{p['chief_complaint']}_",
        "",
        "**Vital Signs:**",
        f"```\n{format_vitals(p['vitals'])}\n```",
    ]
    if p.get("clinical_note"):
        lines += ["", "**Clinical Note:**", f"```\n{p['clinical_note']}\n```"]
    if p.get("arrival_time_minutes") is not None:
        lines += [f"**Arrival:** T+{p['arrival_time_minutes']} min"]
    return "\n".join(lines)


def display_obs(obs_dict: dict) -> str:
    patients = obs_dict.get("patients", [])
    parts = []
    for i, p in enumerate(patients, 1):
        parts.append(format_patient(p, i))
        parts.append("---")
    if obs_dict.get("resources"):
        r = obs_dict["resources"]
        parts.append(
            f"### 🏨 Available Resources\n"
            f"🛏️  Beds: **{r['beds_available']}**  |  "
            f"👨‍⚕️ Physicians: **{r['physicians_available']}**  |  "
            f"👩‍⚕️ Nurses: **{r['nurses_available']}**\n"
            f"🔬 CT: **{r['ct_scanners_available']}**  |  "
            f"📷 X-Ray: **{r['xray_available']}**  |  "
            f"🚨 ICU Beds: **{r['icu_beds_available']}**"
        )
    return "\n\n".join(parts)


def build_action_template(task_id: str, obs_dict: dict) -> str:
    patients = obs_dict.get("patients", [])
    if task_id == "vitals_triage":
        action = {
            "task_id": "vitals_triage",
            "classifications": [
                {"patient_id": p["patient_id"], "esi_level": 3, "rationale": "your reasoning here"}
                for p in patients
            ],
        }
    elif task_id == "clinical_extraction":
        action = {
            "task_id": "clinical_extraction",
            "extractions": [
                {
                    "patient_id": p["patient_id"],
                    "diagnoses": [],
                    "medications": [],
                    "allergies": [],
                    "procedures": [],
                    "follow_up": [],
                }
                for p in patients
            ],
        }
    else:
        action = {
            "task_id": "resource_optimization",
            "assignments": [
                {
                    "patient_id": p["patient_id"],
                    "assigned_bed": True,
                    "assigned_physician": True,
                    "assigned_nurse": True,
                    "assigned_imaging": "none",
                    "assigned_icu": False,
                    "priority_rank": i + 1,
                }
                for i, p in enumerate(patients)
            ],
        }
    return json.dumps(action, indent=2)


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

env_state = {"env": None, "obs": None, "task_id": None}

TASK_LABELS = {
    "vitals_triage": "🩺 Task 1 — Vitals Triage (Easy)",
    "clinical_extraction": "📋 Task 2 — Clinical Extraction (Medium)",
    "resource_optimization": "🔧 Task 3 — Resource Optimization (Hard)",
}


def do_reset(task_label: str) -> tuple:
    task_id = {v: k for k, v in TASK_LABELS.items()}[task_label]
    env = MediRouteEnv(task_id=task_id, seed=42)
    obs = env.reset(task_id)
    obs_dict = obs.model_dump()
    env_state["env"] = env
    env_state["obs"] = obs_dict
    env_state["task_id"] = task_id

    patient_display = display_obs(obs_dict)
    instructions = obs.instructions
    action_template = build_action_template(task_id, obs_dict)
    return patient_display, instructions, action_template, "✅ Environment reset! Edit the action JSON and click **Submit Action**.", ""


def do_step(action_json_str: str) -> tuple:
    if env_state["env"] is None:
        return "", "", "⚠️ Please reset the environment first.", ""
    try:
        action_data = json.loads(action_json_str)
        task_id = env_state["task_id"]

        # Build action object
        if task_id == "vitals_triage":
            action = Action(
                task_id=task_id,
                classifications=[
                    PatientClassification(**c) for c in action_data.get("classifications", [])
                ],
            )
        elif task_id == "clinical_extraction":
            action = Action(
                task_id=task_id,
                extractions=[
                    ExtractedEntities(**e) for e in action_data.get("extractions", [])
                ],
            )
        else:
            action = Action(
                task_id=task_id,
                assignments=[
                    ResourceAssignment(**a) for a in action_data.get("assignments", [])
                ],
            )

        _, reward, done, info = env_state["env"].step(action)

        # Format reward display
        b = reward.breakdown
        reward_md = (
            f"## 🏆 Score: **{reward.total:.4f}** / 1.0\n\n"
            f"{'🚨 **CRITICAL ERROR** — Life-threatening misclassification!' if reward.critical_error else ''}\n\n"
            f"### Breakdown\n"
            f"| Component | Score |\n|---|---|\n"
            f"| Accuracy | {b.accuracy:.4f} |\n"
            f"| Partial Credit | {b.partial_credit:.4f} |\n"
            f"| Extraction F1 | {b.extraction_f1:.4f} |\n"
            f"| Resource Efficiency | {b.resource_efficiency:.4f} |\n"
            f"| Penalties | {b.penalties:.4f} |\n\n"
            f"### Feedback\n_{reward.feedback}_\n\n"
            f"{'✅ Episode complete!' if done else '🔄 Submit another action to refine your assignment.'}"
        )

        truth_md = f"```json\n{json.dumps({k: v for k, v in info.items() if k != 'episode'}, indent=2, default=str)}\n```"
        return reward_md, truth_md, "✅ Action submitted!", ""

    except Exception as e:
        return "", "", f"❌ Error: {e}", traceback.format_exc()


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1200px; margin: auto; }
#title { text-align: center; }
#score-box { background: #1a1a2e; border-radius: 8px; padding: 16px; }
"""

with gr.Blocks(
    title="MediRoute — Emergency Triage OpenEnv",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        # 🏥 MediRoute: Emergency Department Triage OpenEnv
        **An AI-powered triage decision-support environment for the OpenEnv Hackathon.**

        Train and evaluate AI agents on real-world emergency department tasks: patient urgency classification,
        clinical note extraction, and multi-patient resource optimization.
        """,
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=list(TASK_LABELS.values()),
                value=list(TASK_LABELS.values())[0],
                label="Select Task",
            )
            reset_btn = gr.Button("🔄 Generate New Episode", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Column(scale=3):
            patient_display = gr.Markdown(label="Patient Scenario", value="_Click **Generate New Episode** to start._")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📖 Task Instructions")
            instructions_box = gr.Markdown(value="_Instructions will appear here._")
        with gr.Column(scale=2):
            gr.Markdown("### ✏️ Your Action (JSON)")
            action_box = gr.Code(
                language="json",
                label="Action JSON",
                lines=20,
                value='{\n  "task_id": "vitals_triage",\n  "classifications": []\n}',
            )
            submit_btn = gr.Button("🚀 Submit Action", variant="secondary", size="lg")

    with gr.Row():
        with gr.Column(scale=2):
            reward_display = gr.Markdown(label="Reward Signal", value="_Reward will appear after submitting an action._", elem_id="score-box")
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Ground Truth (revealed after submission)")
            truth_box = gr.Code(language="json", label="Info", lines=12)

    error_box = gr.Textbox(label="Error Traceback", visible=True, interactive=False, lines=5)

    # Wire up events
    reset_btn.click(
        fn=do_reset,
        inputs=[task_dropdown],
        outputs=[patient_display, instructions_box, action_box, status_box, error_box],
    )
    submit_btn.click(
        fn=do_step,
        inputs=[action_box],
        outputs=[reward_display, truth_box, status_box, error_box],
    )

    gr.Markdown(
        """
        ---
        **MediRoute** | OpenEnv Hackathon 2025 | 
        [GitHub](https://github.com/your-repo/openenv-mediRoute) |
        Built with ❤️ for better healthcare AI
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
