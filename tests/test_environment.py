"""
MediRoute — Unit Tests for Environment API.
Tests step(), reset(), state() interface compliance.
"""
import pytest

from env.environment import MediRouteEnv
from env.models import Action, PatientClassification, ExtractedEntities, ResourceAssignment


class TestMediRouteInterface:
    """Tests for the core OpenEnv interface."""

    def test_reset_vitals_triage(self):
        env = MediRouteEnv(seed=42)
        obs = env.reset("vitals_triage")
        assert obs.task_id == "vitals_triage"
        assert obs.step == 0
        assert len(obs.patients) > 0
        assert obs.instructions != ""
        assert obs.done is False

    def test_reset_clinical_extraction(self):
        env = MediRouteEnv(seed=42)
        obs = env.reset("clinical_extraction")
        assert obs.task_id == "clinical_extraction"
        for p in obs.patients:
            assert p.clinical_note is not None and len(p.clinical_note) > 10

    def test_reset_resource_optimization(self):
        env = MediRouteEnv(seed=42)
        obs = env.reset("resource_optimization")
        assert obs.task_id == "resource_optimization"
        assert obs.resources is not None
        assert obs.resources.beds_available >= 0

    def test_step_returns_correct_tuple(self):
        env = MediRouteEnv(seed=42)
        obs = env.reset("vitals_triage")
        action = Action(
            task_id="vitals_triage",
            classifications=[
                PatientClassification(patient_id=p.patient_id, esi_level=3)
                for p in obs.patients
            ],
        )
        result = env.step(action)
        assert len(result) == 4
        new_obs, reward, done, info = result
        assert hasattr(reward, "total")
        assert -1.0 <= reward.total <= 1.0
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_dict(self):
        env = MediRouteEnv(seed=42)
        env.reset("vitals_triage")
        state = env.state()
        assert isinstance(state, dict)
        assert "current_task" in state
        assert "episode_count" in state
        assert state["episode_count"] == 1

    def test_invalid_task_raises(self):
        env = MediRouteEnv(seed=42)
        with pytest.raises(ValueError):
            env.reset("invalid_task_name")

    def test_step_before_reset_raises(self):
        env = MediRouteEnv(seed=42)
        action = Action(task_id="vitals_triage", classifications=[])
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_episode_count_increments(self):
        env = MediRouteEnv(seed=42)
        env.reset("vitals_triage")
        assert env.state()["episode_count"] == 1
        env.reset("vitals_triage")
        assert env.state()["episode_count"] == 2

    def test_ground_truth_hidden_from_agent(self):
        env = MediRouteEnv(seed=42)
        obs = env.reset("vitals_triage")
        for p in obs.patients:
            assert p.esi_level_true is None, "Ground truth ESI must be hidden from agent"

    def test_available_tasks(self):
        env = MediRouteEnv(seed=42)
        tasks = env.available_tasks()
        assert "vitals_triage" in tasks
        assert "clinical_extraction" in tasks
        assert "resource_optimization" in tasks

    def test_render_text(self):
        env = MediRouteEnv(seed=42)
        env.reset("vitals_triage")
        rendered = env.render("text")
        assert "MediRoute" in rendered

    def test_reproducibility_same_seed(self):
        """Same seed must produce identical scenarios."""
        env1 = MediRouteEnv(seed=99)
        obs1 = env1.reset("vitals_triage")

        env2 = MediRouteEnv(seed=99)
        obs2 = env2.reset("vitals_triage")

        ids1 = [p.patient_id for p in obs1.patients]
        ids2 = [p.patient_id for p in obs2.patients]
        assert ids1 == ids2

        vit1 = [p.vitals.heart_rate for p in obs1.patients]
        vit2 = [p.vitals.heart_rate for p in obs2.patients]
        assert vit1 == vit2
