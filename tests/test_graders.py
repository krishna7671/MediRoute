"""
MediRoute — Unit Tests for Task Graders.
Tests the scoring logic for all three task graders.
"""
import pytest

from env.environment import MediRouteEnv
from env.models import (
    Action,
    PatientClassification,
    ExtractedEntities,
    ResourceAssignment,
)
from env.reward import compute_triage_reward, compute_extraction_reward, compute_resource_reward


class TestVitralsTriageGrader:

    def _get_patients(self, seed=42, n=5):
        env = MediRouteEnv(seed=seed)
        env.reset("vitals_triage")
        return env._task._patients_with_truth

    def test_perfect_score(self):
        patients = self._get_patients()
        action = Action(
            task_id="vitals_triage",
            classifications=[
                PatientClassification(patient_id=p.patient_id, esi_level=p.esi_level_true)
                for p in patients
            ],
        )
        reward = compute_triage_reward(action, patients)
        assert reward.total == pytest.approx(1.0, abs=0.05)
        assert reward.critical_error is False

    def test_all_wrong_critical_miss(self):
        patients = self._get_patients()
        # Assign all as ESI 5 (non-urgent)
        action = Action(
            task_id="vitals_triage",
            classifications=[
                PatientClassification(patient_id=p.patient_id, esi_level=5)
                for p in patients
            ],
        )
        reward = compute_triage_reward(action, patients)
        assert reward.total < 0.5

    def test_adjacent_gets_partial_credit(self):
        patients = self._get_patients()
        # Find one patient and assign ±1
        p = patients[0]
        adjacent = max(1, min(5, p.esi_level_true + 1))
        action = Action(
            task_id="vitals_triage",
            classifications=[
                PatientClassification(patient_id=patients[0].patient_id, esi_level=adjacent)
            ] + [
                PatientClassification(patient_id=px.patient_id, esi_level=px.esi_level_true)
                for px in patients[1:]
            ],
        )
        reward = compute_triage_reward(action, patients)
        # Should be between 0.5 and 1.0 due to partial credit on one patient
        assert 0.5 < reward.total < 1.0

    def test_empty_classifications_penalty(self):
        patients = self._get_patients()
        action = Action(task_id="vitals_triage", classifications=[])
        reward = compute_triage_reward(action, patients)
        assert reward.total < 0.0

    def test_reward_in_range(self):
        patients = self._get_patients()
        for esi_level in [1, 2, 3, 4, 5]:
            action = Action(
                task_id="vitals_triage",
                classifications=[
                    PatientClassification(patient_id=p.patient_id, esi_level=esi_level)
                    for p in patients
                ],
            )
            reward = compute_triage_reward(action, patients)
            assert -1.0 <= reward.total <= 1.0, f"Reward out of range for ESI {esi_level}"


class TestClinicalExtractionGrader:

    def _get_setup(self, seed=42):
        env = MediRouteEnv(seed=seed)
        env.reset("clinical_extraction")
        return env._task._display_patients, env._task._true_entities_map

    def test_perfect_extraction(self):
        patients, true_map = self._get_setup()
        extractions = []
        for p in patients:
            true = true_map[p.patient_id]
            extractions.append(ExtractedEntities(
                patient_id=p.patient_id,
                diagnoses=true.get("diagnoses", []),
                medications=true.get("medications", []),
                allergies=true.get("allergies", []),
                procedures=true.get("procedures", []),
                follow_up=true.get("follow_up", []),
            ))
        action = Action(task_id="clinical_extraction", extractions=extractions)
        reward = compute_extraction_reward(action, true_map)
        assert reward.total == pytest.approx(1.0, abs=0.01)

    def test_empty_gives_low_score(self):
        patients, true_map = self._get_setup()
        action = Action(
            task_id="clinical_extraction",
            extractions=[
                ExtractedEntities(patient_id=p.patient_id) for p in patients
            ],
        )
        reward = compute_extraction_reward(action, true_map)
        assert reward.total < 0.3

    def test_reward_in_range(self):
        patients, true_map = self._get_setup()
        action = Action(
            task_id="clinical_extraction",
            extractions=[ExtractedEntities(patient_id=p.patient_id) for p in patients],
        )
        reward = compute_extraction_reward(action, true_map)
        assert -1.0 <= reward.total <= 1.0


class TestResourceOptimizationGrader:

    def _get_setup(self, seed=42, n=10):
        env = MediRouteEnv(seed=seed, n_patients_hard=n)
        env.reset("resource_optimization")
        return env._task._true_patients, env._task._resources

    def test_correct_priority_order(self):
        patients, resources = self._get_setup()
        sorted_patients = sorted(patients, key=lambda p: p.esi_level_true or 5)
        assignments = [
            ResourceAssignment(
                patient_id=p.patient_id,
                assigned_bed=True,
                assigned_physician=True,
                assigned_nurse=True,
                assigned_imaging="none",
                assigned_icu=False,
                priority_rank=i + 1,
            )
            for i, p in enumerate(sorted_patients)
        ]
        action = Action(task_id="resource_optimization", assignments=assignments)
        reward = compute_resource_reward(action, patients, resources)
        assert reward.total >= 0.3

    def test_constraint_violation_penalty(self):
        patients, resources = self._get_setup(n=5)
        # Over-assign beds way beyond capacity
        assignments = [
            ResourceAssignment(
                patient_id=p.patient_id,
                assigned_bed=True,  # All get beds (may exceed capacity)
                assigned_physician=True,
                assigned_nurse=True,
                assigned_imaging="ct",  # All get CT (definitely exceeds)
                assigned_icu=True,      # All get ICU (definitely exceeds)
                priority_rank=i + 1,
            )
            for i, p in enumerate(patients)
        ]
        action = Action(task_id="resource_optimization", assignments=assignments)
        reward = compute_resource_reward(action, patients, resources)
        # Should have penalties reducing score
        assert reward.breakdown.penalties < 0

    def test_empty_assignment_penalty(self):
        patients, resources = self._get_setup()
        action = Action(task_id="resource_optimization", assignments=[])
        reward = compute_resource_reward(action, patients, resources)
        assert reward.total < 0
