"""
Task 3: Multi-Patient Resource Optimization (Hard)
Agent receives 8–12 patients with varying ESI levels + constrained hospital resources.
Must assign resources optimally, prioritising critical patients.
"""
from __future__ import annotations

from typing import List

from ..data_generator import PatientDataGenerator
from ..models import Action, Observation, PatientRecord, ResourcePool, Reward
from ..reward import compute_resource_reward


TASK_INSTRUCTIONS = """
You are an ED resource allocation system managing a patient surge.

You will receive a list of patients each with vital signs, chief complaints, and arrival times,
plus the available hospital resources.

Your job is to assign resources and set priority rankings. For each patient, specify:
  - assigned_bed: true/false (whether to allocate an ED bed)
  - assigned_physician: true/false (whether to assign an attending physician)
  - assigned_nurse: true/false (whether to assign a nurse)
  - assigned_imaging: "ct" | "xray" | "none"
  - assigned_icu: true/false (whether to transfer to ICU — use sparingly!)
  - priority_rank: integer 1..N (1 = seen first, N = seen last)

CONSTRAINTS — you must NOT exceed the available counts listed under "resources".
OBJECTIVE — assign resources so that:
  1. Most critical patients (likely ESI 1/2 based on vitals) are seen first
  2. Each critical patient gets at minimum a bed + physician
  3. Maximize resource utilization without over-assigning

Return your assignments as a JSON action with the `assignments` field.

Example action:
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
""".strip()


class ResourceOptimizationTask:
    """Hard task: allocate limited ER resources across multiple simultaneous patients."""

    task_id = "resource_optimization"
    difficulty = "hard"
    max_steps = 3

    def __init__(self, n_patients: int = 10, seed: int = 42):
        self.n_patients = n_patients
        self.generator = PatientDataGenerator(seed=seed + 200)
        self._step = 0
        self._done = False
        self._display_patients: List[PatientRecord] = []
        self._true_patients: List[PatientRecord] = []
        self._resources: ResourcePool = None
        self._cumulative_score: float = 0.0
        self._steps_taken: int = 0

    def reset(self) -> Observation:
        self.generator.reset_counter()
        self._step = 0
        self._done = False
        self._cumulative_score = 0.0
        self._steps_taken = 0
        self._display_patients = []
        self._true_patients = []

        base_time = 0
        for _ in range(self.n_patients):
            patient, _ = self.generator.generate_patient(
                include_note=False,
                include_arrival_time=True,
                base_time=base_time,
            )
            base_time = (patient.arrival_time_minutes or 0) + 5
            self._true_patients.append(patient)
            display = patient.model_copy(update={"esi_level_true": None})
            self._display_patients.append(display)

        self._resources = self.generator.generate_resource_pool(self.n_patients)

        # Build context string with resource constraints
        r = self._resources
        resource_summary = (
            f"Available resources: {r.beds_available} beds, "
            f"{r.physicians_available} physicians, "
            f"{r.nurses_available} nurses, "
            f"{r.ct_scanners_available} CT scanners, "
            f"{r.xray_available} X-ray slots, "
            f"{r.icu_beds_available} ICU beds."
        )

        return Observation(
            task_id=self.task_id,
            step=self._step,
            patients=self._display_patients,
            resources=self._resources,
            instructions=TASK_INSTRUCTIONS,
            context={"resource_summary": resource_summary, "n_patients": self.n_patients},
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        reward = compute_resource_reward(action, self._true_patients, self._resources)
        self._step += 1
        self._steps_taken += 1
        self._cumulative_score += reward.total

        # Allow up to max_steps; task ends when max steps reached or agent signals done
        done = self._steps_taken >= self.max_steps
        self._done = done

        obs = Observation(
            task_id=self.task_id,
            step=self._step,
            patients=self._display_patients,
            resources=self._resources,
            instructions=TASK_INSTRUCTIONS,
            done=done,
            context={
                "resource_summary": (
                    f"Available resources: {self._resources.beds_available} beds, "
                    f"{self._resources.physicians_available} physicians."
                ),
                "step_score": reward.total,
                "cumulative_score": self._cumulative_score / self._steps_taken,
            },
        )
        info = {
            "true_esi_levels": {p.patient_id: p.esi_level_true for p in self._true_patients},
            "step_score": reward.total,
            "cumulative_score": self._cumulative_score / self._steps_taken,
            "steps_taken": self._steps_taken,
        }
        return obs, reward, done, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step,
            "done": self._done,
            "n_patients": len(self._true_patients),
            "resources": self._resources.model_dump() if self._resources else {},
            "cumulative_score": (
                self._cumulative_score / max(self._steps_taken, 1)
            ),
        }
