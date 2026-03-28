"""
MediRoute — Pydantic models for Observation, Action, and Reward.
These define the typed interface agents interact with.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Sub-models: Patient data structures
# ─────────────────────────────────────────────

class VitalSigns(BaseModel):
    """Structured vital sign measurements."""
    heart_rate: int = Field(..., ge=0, le=300, description="Heart rate in bpm")
    systolic_bp: int = Field(..., ge=0, le=300, description="Systolic blood pressure mmHg")
    diastolic_bp: int = Field(..., ge=0, le=200, description="Diastolic blood pressure mmHg")
    respiratory_rate: int = Field(..., ge=0, le=80, description="Breaths per minute")
    spo2: float = Field(..., ge=0.0, le=100.0, description="Blood oxygen saturation %")
    temperature: float = Field(..., ge=30.0, le=45.0, description="Body temperature °C")
    gcs: int = Field(..., ge=3, le=15, description="Glasgow Coma Scale score")
    pain_scale: int = Field(..., ge=0, le=10, description="Patient-reported pain 0–10")


class ResourcePool(BaseModel):
    """Available hospital resources at a point in time."""
    beds_available: int = Field(..., ge=0, description="Free ED beds")
    physicians_available: int = Field(..., ge=0, description="Free attending physicians")
    nurses_available: int = Field(..., ge=0, description="Free nurses")
    ct_scanners_available: int = Field(..., ge=0, description="Free CT scanner slots")
    xray_available: int = Field(..., ge=0, description="Free X-ray slots")
    icu_beds_available: int = Field(..., ge=0, description="Free ICU beds")


class PatientRecord(BaseModel):
    """A single patient presentation."""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=120)
    sex: Literal["M", "F", "Other"]
    chief_complaint: str = Field(..., description="Primary reason for ED visit")
    vitals: VitalSigns
    clinical_note: Optional[str] = Field(None, description="Free-text SOAP note (Task 2 & 3)")
    arrival_time_minutes: Optional[int] = Field(None, description="Minutes since shift start")
    esi_level_true: Optional[int] = Field(None, ge=1, le=5, description="Ground truth ESI; hidden from agent")


# ─────────────────────────────────────────────
# Observation — what the agent sees
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """Full observation returned after reset() or step()."""
    task_id: str = Field(..., description="Active task identifier")
    step: int = Field(..., ge=0, description="Current step within episode")
    patients: List[PatientRecord] = Field(..., description="One or more patient records to process")
    resources: Optional[ResourcePool] = Field(None, description="Hospital resources (Task 3 only)")
    instructions: str = Field(..., description="Natural-language task instructions for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Extra task-specific context")
    done: bool = Field(False, description="Whether the episode has ended")


# ─────────────────────────────────────────────
# Action — what the agent does
# ─────────────────────────────────────────────

class PatientClassification(BaseModel):
    """Task 1 action: classify single patient ESI level."""
    patient_id: str
    esi_level: int = Field(..., ge=1, le=5, description="Predicted ESI urgency 1=most urgent")
    rationale: Optional[str] = Field(None, description="Agent's clinical reasoning (logged)")


class ExtractedEntities(BaseModel):
    """Task 2 action: extracted clinical entities from a note."""
    patient_id: str
    diagnoses: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    procedures: List[str] = Field(default_factory=list)
    follow_up: List[str] = Field(default_factory=list)


class ResourceAssignment(BaseModel):
    """Single patient → resource assignment."""
    patient_id: str
    assigned_bed: Optional[bool] = Field(None)
    assigned_physician: Optional[bool] = Field(None)
    assigned_nurse: Optional[bool] = Field(None)
    assigned_imaging: Optional[Literal["ct", "xray", "none"]] = Field("none")
    assigned_icu: Optional[bool] = Field(False)
    priority_rank: int = Field(..., ge=1, description="Treatment priority (1 = seen first)")


class Action(BaseModel):
    """
    Unified action model. Populate the field matching the current task.
    - Task 1: fill `classifications`
    - Task 2: fill `extractions`
    - Task 3: fill `assignments`
    """
    task_id: str
    classifications: Optional[List[PatientClassification]] = None
    extractions: Optional[List[ExtractedEntities]] = None
    assignments: Optional[List[ResourceAssignment]] = None


# ─────────────────────────────────────────────
# Reward — structured reward signal
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Detailed per-component reward breakdown."""
    accuracy: float = Field(0.0, ge=0.0, le=1.0)
    partial_credit: float = Field(0.0, ge=0.0, le=1.0)
    extraction_f1: float = Field(0.0, ge=0.0, le=1.0)
    resource_efficiency: float = Field(0.0, ge=0.0, le=1.0)
    penalties: float = Field(0.0, le=0.0, description="Negative value for penalties")


class Reward(BaseModel):
    """Reward returned by step()."""
    total: float = Field(..., ge=-1.0, le=1.0, description="Overall reward in [-1, 1]")
    breakdown: RewardBreakdown
    feedback: str = Field(..., description="Human-readable feedback on the action")
    critical_error: bool = Field(False, description="True if agent made a life-threatening misclassification")
