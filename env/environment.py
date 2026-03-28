"""
MediRoute — Main Environment Orchestrator.
Implements the full OpenEnv interface: step() / reset() / state().
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

from .models import Action, Observation, Reward
from .tasks.task_vitals import VitalsTriageTask
from .tasks.task_clinical import ClinicalExtractionTask
from .tasks.task_resource import ResourceOptimizationTask


VALID_TASKS = {
    "vitals_triage": VitalsTriageTask,
    "clinical_extraction": ClinicalExtractionTask,
    "resource_optimization": ResourceOptimizationTask,
}


class MediRouteEnv:
    """
    MediRoute: Emergency Department Triage Decision-Support Environment.

    An AI agent plays the role of a triage decision-support system in a simulated
    Emergency Department. The environment provides three tasks of increasing difficulty:

    Tasks:
        - vitals_triage      (easy):   Classify patient ESI urgency from vital signs
        - clinical_extraction (medium): Extract structured entities from clinical notes
        - resource_optimization (hard): Allocate limited ED resources optimally

    Usage:
        env = MediRouteEnv()
        obs = env.reset("vitals_triage")
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
    """

    metadata = {
        "name": "MediRoute",
        "version": "1.0.0",
        "tasks": list(VALID_TASKS.keys()),
        "render_modes": ["text", "json"],
    }

    def __init__(
        self,
        task_id: Optional[str] = None,
        seed: int = 42,
        n_patients_easy: int = 5,
        n_patients_medium: int = 3,
        n_patients_hard: int = 10,
    ):
        self._seed = seed
        self._current_task_id: Optional[str] = task_id
        self._task = None
        self._episode_count = 0
        self._episode_scores: list[float] = []

        # Task configs
        self._task_configs = {
            "vitals_triage": {"n_patients": n_patients_easy, "seed": seed},
            "clinical_extraction": {"n_patients": n_patients_medium, "seed": seed},
            "resource_optimization": {"n_patients": n_patients_hard, "seed": seed},
        }

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: One of 'vitals_triage', 'clinical_extraction', 'resource_optimization'.
                     If None, uses the task_id set at construction.

        Returns:
            Observation: Initial observation for the new episode.
        """
        if task_id:
            self._current_task_id = task_id
        if not self._current_task_id or self._current_task_id not in VALID_TASKS:
            raise ValueError(
                f"Invalid task_id '{self._current_task_id}'. "
                f"Choose from: {list(VALID_TASKS.keys())}"
            )

        cfg = self._task_configs[self._current_task_id]
        self._task = VALID_TASKS[self._current_task_id](**cfg)
        self._episode_count += 1

        return self._task.reset()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Take a step in the current episode.

        Args:
            action: An Action object with the agent's decision.

        Returns:
            observation: Updated observation
            reward:      Reward object with total and breakdown
            done:        True if episode is complete
            info:        Additional diagnostic information
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        obs, reward, done, info = self._task.step(action)

        if done:
            self._episode_scores.append(reward.total)

        info["episode"] = {
            "count": self._episode_count,
            "mean_score": sum(self._episode_scores[-10:]) / max(len(self._episode_scores[-10:]), 1),
        }
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the full internal state of the current episode.

        Returns:
            dict: Task state, episode stats, metadata.
        """
        task_state = self._task.state() if self._task else {}
        return {
            "env": self.metadata,
            "current_task": self._current_task_id,
            "episode_count": self._episode_count,
            "episode_scores": self._episode_scores,
            "mean_score_last_10": (
                sum(self._episode_scores[-10:]) / max(len(self._episode_scores[-10:]), 1)
            ),
            "task_state": task_state,
        }

    def render(self, mode: str = "text") -> str:
        """Human-readable rendering of the current environment state."""
        s = self.state()
        if mode == "json":
            import json
            return json.dumps(s, indent=2)
        task = s.get("task_state", {})
        lines = [
            f"=== MediRoute Environment ===",
            f"Task:       {s['current_task']}",
            f"Episode:    {s['episode_count']}",
            f"Step:       {task.get('step', 'N/A')}",
            f"Done:       {task.get('done', False)}",
            f"Mean Score (last 10): {s['mean_score_last_10']:.4f}",
        ]
        return "\n".join(lines)

    def available_tasks(self) -> list[str]:
        """Return the list of available task IDs."""
        return list(VALID_TASKS.keys())
