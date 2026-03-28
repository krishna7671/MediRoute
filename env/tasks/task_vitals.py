"""
Task 1: Patient Vitals Triage (Easy)
Agent receives structured vitals + chief complaint → must classify ESI level 1–5.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..data_generator import PatientDataGenerator
from ..models import Action, Observation, PatientRecord, Reward
from ..reward import compute_triage_reward


TASK_INSTRUCTIONS = """
You are an ED triage decision support system.

For each patient listed below, assess their ESI (Emergency Severity Index) level:
  ESI 1 — Immediate: life-threatening, requires immediate intervention
  ESI 2 — Emergent: high risk / time-sensitive, could deteriorate
  ESI 3 — Urgent: stable but needs multiple diagnostic resources
  ESI 4 — Less Urgent: stable, needs one resource
  ESI 5 — Non-Urgent: stable, no resources needed

Review vitals and chief complaint carefully.
Return your classifications as a JSON action with the `classifications` field.

Example action:
{
  "task_id": "vitals_triage",
  "classifications": [
    {"patient_id": "PT-0001", "esi_level": 2, "rationale": "Elevated BP, chest pain"},
    {"patient_id": "PT-0002", "esi_level": 4, "rationale": "Low-grade fever, ear pain"}
  ]
}
""".strip()


class VitalsTriageTask:
    """Easy task: classify patient urgency from structured vital signs."""

    task_id = "vitals_triage"
    difficulty = "easy"
    max_steps = 1

    def __init__(self, n_patients: int = 5, seed: int = 42):
        self.n_patients = n_patients
        self.generator = PatientDataGenerator(seed=seed)
        self._patients: List[PatientRecord] = []
        self._step = 0
        self._done = False

    def reset(self) -> Observation:
        self.generator.reset_counter()
        self._step = 0
        self._done = False

        # Generate patients once — keep full records internally, hide ESI from agent
        self._patients_with_truth = []
        for _ in range(self.n_patients):
            patient, _ = self.generator.generate_patient(include_note=False)
            self._patients_with_truth.append(patient)

        self._display_patients = [
            p.model_copy(update={"esi_level_true": None})
            for p in self._patients_with_truth
        ]

        return Observation(
            task_id=self.task_id,
            step=self._step,
            patients=self._display_patients,
            instructions=TASK_INSTRUCTIONS,
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        reward = compute_triage_reward(action, self._patients_with_truth)
        self._done = True
        self._step += 1

        obs = Observation(
            task_id=self.task_id,
            step=self._step,
            patients=self._display_patients,
            instructions=TASK_INSTRUCTIONS,
            done=True,
        )
        info = {
            "true_esi_levels": {p.patient_id: p.esi_level_true for p in self._patients_with_truth},
            "score": reward.total,
        }
        return obs, reward, True, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step,
            "done": self._done,
            "n_patients": len(self._patients_with_truth),
            "patients": [p.model_dump() for p in self._patients_with_truth],
        }
