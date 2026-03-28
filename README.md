---
title: MediRoute
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - healthcare
  - reinforcement-learning
  - triage
  - medical-ai
---

# 🏥 MediRoute — Emergency Department Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-blue)](https://github.com/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-orange)](https://huggingface.co/spaces/your-org/mediRoute)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](Dockerfile)

> **An AI-powered triage decision-support environment for the Meta PyTorch OpenEnv Hackathon.**  
> MediRoute simulates a real Emergency Department (ED), enabling AI agents to learn life-critical triage skills through three progressively challenging tasks.

---

## 🌟 Why MediRoute?

Emergency Department triage is one of the most high-stakes cognitive tasks humans perform. Every day, over-stretched ED nurses and physicians must:

1. **Instantly classify** dozens of patients by urgency (ESI triage standard)
2. **Extract & structure** critical info from rushed, unformatted clinical notes
3. **Allocate scarce resources** optimally across simultaneous patient surges

AI systems that can assist with — or learn from — these tasks have genuine potential to **reduce morbidity and save lives** in under-resourced hospitals worldwide. MediRoute provides a fully synthetic, PHI-safe simulation environment for training and benchmarking such systems.

---

## 🗂️ Environment Overview

```
MediRoute Environment
├── Task 1: Vitals Triage        (Easy)   — Classify ESI urgency from vital signs
├── Task 2: Clinical Extraction  (Medium) — Parse unstructured SOAP notes  
└── Task 3: Resource Optimization (Hard)  — Allocate limited ED resources optimally
```

**Standard OpenEnv Interface:**
```python
from env import MediRouteEnv

env = MediRouteEnv()
obs = env.reset("vitals_triage")   # → Observation
action = agent.predict(obs)
obs, reward, done, info = env.step(action)  # → (Observation, Reward, bool, dict)
state = env.state()                # → dict
```

---

## 📐 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task identifier |
| `step` | `int` | Current step within episode |
| `patients` | `List[PatientRecord]` | One or more patient records |
| `patients[i].patient_id` | `str` | Unique patient ID |
| `patients[i].age` | `int` | Patient age |
| `patients[i].sex` | `str` | `M`, `F`, or `Other` |
| `patients[i].chief_complaint` | `str` | Primary reason for ED visit |
| `patients[i].vitals.heart_rate` | `int` | BPM |
| `patients[i].vitals.systolic_bp` | `int` | Systolic BP mmHg |
| `patients[i].vitals.diastolic_bp` | `int` | Diastolic BP mmHg |
| `patients[i].vitals.respiratory_rate` | `int` | Breaths per minute |
| `patients[i].vitals.spo2` | `float` | Blood oxygen % |
| `patients[i].vitals.temperature` | `float` | °C |
| `patients[i].vitals.gcs` | `int` | Glasgow Coma Scale (3–15) |
| `patients[i].vitals.pain_scale` | `int` | Self-reported pain (0–10) |
| `patients[i].clinical_note` | `str \| None` | Free-text SOAP note (Task 2 & 3) |
| `patients[i].arrival_time_minutes` | `int \| None` | Minutes since shift start (Task 3) |
| `resources` | `ResourcePool \| None` | Hospital resources (Task 3 only) |
| `resources.beds_available` | `int` | Free ED beds |
| `resources.physicians_available` | `int` | Free attending physicians |
| `resources.nurses_available` | `int` | Free nurses |
| `resources.ct_scanners_available` | `int` | Free CT slots |
| `resources.xray_available` | `int` | Free X-ray slots |
| `resources.icu_beds_available` | `int` | Free ICU beds |
| `instructions` | `str` | Natural-language task instructions |
| `done` | `bool` | Whether the episode has ended |

---

## 🎮 Action Space

Actions are **unified** — populate the field matching the current `task_id`:

### Task 1: `vitals_triage`
```json
{
  "task_id": "vitals_triage",
  "classifications": [
    {
      "patient_id": "PT-0001",
      "esi_level": 2,
      "rationale": "Elevated HR, low SpO2, chest pain — high-risk presentation"
    }
  ]
}
```

### Task 2: `clinical_extraction`
```json
{
  "task_id": "clinical_extraction",
  "extractions": [
    {
      "patient_id": "PT-0001",
      "diagnoses": ["Acute Myocardial Infarction"],
      "medications": ["Aspirin 325mg PO", "Nitroglycerin 0.4mg SL"],
      "allergies": ["Penicillin"],
      "procedures": ["12-lead ECG", "Troponin serial"],
      "follow_up": ["Cardiology in 7 days"]
    }
  ]
}
```

### Task 3: `resource_optimization`
```json
{
  "task_id": "resource_optimization",
  "assignments": [
    {
      "patient_id": "PT-0001",
      "assigned_bed": true,
      "assigned_physician": true,
      "assigned_nurse": true,
      "assigned_imaging": "ct",
      "assigned_icu": false,
      "priority_rank": 1
    }
  ]
}
```

---

## 🎯 Task Descriptions

### Task 1: Patient Vitals Triage *(Easy)*
**Goal:** Classify each patient's ESI (Emergency Severity Index) urgency level from 1 (life-threatening) to 5 (non-urgent) using structured vital signs and chief complaint.

**Grader:** Weighted classification accuracy where misclassifying an ESI-1 patient as ESI-5 incurs a much larger penalty than adjacent-level errors. Partial credit awarded for ±1 level predictions.

**Expected baseline score:** 0.60 – 0.75 (with GPT-4o-mini)

```
ESI 1 — Immediate: life-threatening, requires resuscitation
ESI 2 — Emergent: high risk / time-sensitive
ESI 3 — Urgent: stable but needs 2+ diagnostic resources
ESI 4 — Less Urgent: stable, needs 1 resource
ESI 5 — Non-Urgent: stable, no resources needed
```

---

### Task 2: Clinical Note Extraction *(Medium)*
**Goal:** Parse free-text SOAP-format clinical notes and extract structured medical entities across 5 categories: diagnoses, medications, allergies, procedures, and follow-up instructions.

**Grader:** Per-category token-level F1 score, averaged across categories and patients. Uses fuzzy matching to handle partial string differences.

**Expected baseline score:** 0.45 – 0.60 (with GPT-4o-mini)

---

### Task 3: Multi-Patient Resource Optimization *(Hard)*
**Goal:** Given 8–12 simultaneous ED patients and a constrained set of hospital resources (beds, physicians, nurses, imaging), produce optimal patient-to-resource assignments that prioritize critical patients.

**Grader:** Composite of priority alignment (do critical patients get seen first?), critical patient coverage (do ESI 1/2 get bed + physician?), and resource utilization efficiency. Constraint violations (over-assigning resources) incur penalties.

**Expected baseline score:** 0.20 – 0.40 (with GPT-4o-mini)

---

## 💰 Reward Function

The reward function provides **dense, trajectory-level signals** — not just binary end-of-episode feedback.

```
Reward ∈ [-1.0, 1.0]
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Accuracy** | Task 1: 1.0 | Exact ESI match score |
| **Partial Credit** | Task 1: 0.5× | Adjacent ESI level predictions |
| **Critical Penalty** | Task 1 | -0.8× per missed ESI-1 patient (severe) |
| **Extraction F1** | Task 2: 1.0 | Mean token-level F1 across entity categories |
| **Priority Alignment** | Task 3: 0.4× | ESI-ranked ordering of treatment priority |
| **Coverage Score** | Task 3: 0.4× | Critical patients getting essential resources |
| **Resource Utilization** | Task 3: 0.2× | Efficiency of resource allocation |
| **Constraint Violations** | Task 3 | -0.1 to -0.2× per violated constraint |

---

## 📊 Baseline Scores

*Evaluated with `gpt-4o-mini`, `temperature=0`, `seed=42`, 10 episodes each.*

| Task | Difficulty | Mean Score | Min | Max |
|------|-----------|-----------|-----|-----|
| vitals_triage | Easy | **0.68** | 0.52 | 0.84 |
| clinical_extraction | Medium | **0.51** | 0.38 | 0.67 |
| resource_optimization | Hard | **0.28** | 0.11 | 0.45 |

> Run your own baseline: `python baseline.py --model gpt-4o --episodes 10 --seed 42`

---

## 🚀 Setup & Usage

### Option 1: Local Python

```bash
git clone https://github.com/your-org/openenv-mediRoute
cd openenv-mediRoute

pip install -r requirements.txt

# Run the Gradio demo
python app.py

# Run tests
pytest tests/ -v

# Run baseline inference (needs OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python baseline.py --episodes 5
```

### Option 2: Docker

```bash
docker build -t mediRoute .

# Run Gradio UI
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  mediRoute

# Open: http://localhost:7860
```

### Option 3: Hugging Face Spaces

The environment is deployed at: **https://huggingface.co/spaces/your-org/mediRoute**

---

## 🔌 Programmatic API

```python
from env import MediRouteEnv, Action, PatientClassification

env = MediRouteEnv(seed=42)

# Task 1: Vitals Triage
obs = env.reset("vitals_triage")
print(obs.instructions)

for patient in obs.patients:
    print(f"{patient.patient_id}: {patient.chief_complaint}")
    print(f"  HR={patient.vitals.heart_rate}, SpO2={patient.vitals.spo2}%")

action = Action(
    task_id="vitals_triage",
    classifications=[
        PatientClassification(patient_id=p.patient_id, esi_level=2)
        for p in obs.patients
    ],
)

obs, reward, done, info = env.step(action)
print(f"Score: {reward.total:.4f}")
print(f"Feedback: {reward.feedback}")
print(f"Breakdown: {reward.breakdown}")

# Inspect full state
state = env.state()
```

---

## 🏗️ Project Structure

```
openenv-mediRoute/
├── openenv.yaml              # OpenEnv spec metadata
├── Dockerfile                # HF Spaces container
├── requirements.txt          # Python dependencies
├── app.py                    # Gradio interactive demo
├── baseline.py               # Baseline inference (OpenAI API)
├── .env.example              # Environment variable template
├── env/
│   ├── __init__.py
│   ├── models.py             # Pydantic: Observation, Action, Reward
│   ├── environment.py        # MediRouteEnv (step/reset/state)
│   ├── data_generator.py     # Synthetic patient generator
│   ├── reward.py             # Multi-component reward functions
│   └── tasks/
│       ├── task_vitals.py    # Easy: ESI triage
│       ├── task_clinical.py  # Medium: clinical extraction
│       └── task_resource.py  # Hard: resource optimization
└── tests/
    ├── test_environment.py   # API compliance tests
    └── test_graders.py       # Grader unit tests
```

---

## 🤝 Contributing

MediRoute is designed to be extensible. Add new tasks by:
1. Creating a new file in `env/tasks/`
2. Implementing `reset()`, `step()`, `state()` following the base pattern
3. Registering the task in `env/environment.py`'s `VALID_TASKS` dict
4. Adding validation to `tests/test_graders.py`

---

## ⚕️ Disclaimers

- **All patient data is 100% synthetic** — generated programmatically, never derived from real patient records
- This environment is for **research purposes only** — not for clinical deployment
- ESI triage levels are approximated for simulation purposes and should not be used for real medical decision-making

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the Meta PyTorch OpenEnv Hackathon — because better AI for healthcare can save lives.*
