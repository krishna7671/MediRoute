"""
Task 2: Clinical Note Extraction (Medium)
Agent receives a free-text SOAP clinical note → must extract structured entities.
"""
from __future__ import annotations

from typing import Dict, List

from ..data_generator import PatientDataGenerator
from ..models import Action, Observation, PatientRecord, Reward
from ..reward import compute_extraction_reward


TASK_INSTRUCTIONS = """
You are a clinical information extraction system.

You will receive one or more clinical notes in SOAP format. Your job is to extract:
  - diagnoses: List of medical diagnoses or conditions mentioned
  - medications: List of medications (include dosage if present)
  - allergies: List of drug/substance allergies
  - procedures: List of ordered tests, procedures, or imaging
  - follow_up: List of follow-up instructions or referrals

Return your extractions as a JSON action with the `extractions` field.

Example action:
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

Be thorough — extract ALL entities mentioned, not just the most prominent ones.
""".strip()


class ClinicalExtractionTask:
    """Medium task: extract structured entities from free-text clinical notes."""

    task_id = "clinical_extraction"
    difficulty = "medium"
    max_steps = 1

    def __init__(self, n_patients: int = 3, seed: int = 42):
        self.n_patients = n_patients
        self.generator = PatientDataGenerator(seed=seed + 100)
        self._step = 0
        self._done = False
        self._display_patients: List[PatientRecord] = []
        self._true_entities_map: Dict[str, Dict] = {}

    def reset(self) -> Observation:
        self.generator.reset_counter()
        self._step = 0
        self._done = False
        self._display_patients = []
        self._true_entities_map = {}

        for _ in range(self.n_patients):
            patient, entities = self.generator.generate_patient(include_note=True)
            self._true_entities_map[patient.patient_id] = entities
            # Hide ESI level from agent
            display = patient.model_copy(update={"esi_level_true": None})
            self._display_patients.append(display)

        return Observation(
            task_id=self.task_id,
            step=self._step,
            patients=self._display_patients,
            instructions=TASK_INSTRUCTIONS,
            context={"note_field": "clinical_note"},
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        reward = compute_extraction_reward(action, self._true_entities_map)
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
            "true_entities": self._true_entities_map,
            "score": reward.total,
        }
        return obs, reward, True, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step,
            "done": self._done,
            "n_patients": len(self._display_patients),
            "true_entities": self._true_entities_map,
        }
