"""
MediRoute — Synthetic patient data generator.
Generates realistic (but entirely synthetic) patient scenarios for all three tasks.
"""
from __future__ import annotations

import random
from typing import Optional

from .models import (
    PatientRecord,
    VitalSigns,
    ResourcePool,
)


# ─────────────────────────────────────────────
# Clinical knowledge bases
# ─────────────────────────────────────────────

CHIEF_COMPLAINTS = {
    1: [  # ESI 1 — Immediate (life-threatening)
        "unresponsive, found down",
        "cardiac arrest, no pulse",
        "severe respiratory distress, unable to speak",
        "major trauma, multisystem injury",
        "stroke symptoms, left side paralysis",
        "status epilepticus",
        "anaphylactic shock",
        "massive GI bleed with hemodynamic collapse",
    ],
    2: [  # ESI 2 — Emergent (high risk)
        "chest pain radiating to left arm",
        "sudden severe headache 'worst of my life'",
        "altered mental status",
        "severe abdominal pain",
        "shortness of breath, SpO2 dropping",
        "suspected overdose, drowsy",
        "high fever with neck stiffness",
        "active seizure",
    ],
    3: [  # ESI 3 — Urgent (stable, needs multiple resources)
        "moderate chest pain, stable vitals",
        "laceration requiring sutures",
        "moderate shortness of breath",
        "urinary tract infection with flank pain",
        "fracture wrist after fall",
        "abdominal pain, nausea, vomiting",
        "moderate allergic reaction",
        "head injury after fall, conscious",
    ],
    4: [  # ESI 4 — Less Urgent (one resource needed)
        "ear pain",
        "sore throat, mild fever",
        "ankle sprain after sports injury",
        "minor lacerations",
        "rash, no systemic symptoms",
        "back pain, chronic",
        "nausea without vomiting",
        "minor eye irritation",
    ],
    5: [  # ESI 5 — Non-Urgent (no resources)
        "prescription refill request",
        "cold symptoms for 3 days",
        "minor finger abrasion",
        "paperwork completion",
        "minor headache, normal vitals",
        "athlete's foot",
        "routine checkup",
        "medication question",
    ],
}

# Vital sign ranges per ESI level (hr, sbp, dbp, rr, spo2, temp, gcs, pain)
VITALS_BY_ESI = {
    1: dict(
        heart_rate=(130, 200), systolic_bp=(50, 90), diastolic_bp=(30, 60),
        respiratory_rate=(30, 50), spo2=(70.0, 88.0), temperature=(34.0, 40.5),
        gcs=(3, 8), pain_scale=(0, 4),  # may not respond to pain
    ),
    2: dict(
        heart_rate=(110, 140), systolic_bp=(85, 105), diastolic_bp=(50, 70),
        respiratory_rate=(22, 32), spo2=(88.0, 94.0), temperature=(38.5, 40.5),
        gcs=(8, 13), pain_scale=(7, 10),
    ),
    3: dict(
        heart_rate=(90, 115), systolic_bp=(100, 140), diastolic_bp=(60, 90),
        respiratory_rate=(16, 24), spo2=(94.0, 98.0), temperature=(37.0, 39.0),
        gcs=(13, 15), pain_scale=(5, 8),
    ),
    4: dict(
        heart_rate=(65, 95), systolic_bp=(110, 135), diastolic_bp=(65, 85),
        respiratory_rate=(14, 18), spo2=(97.0, 99.5), temperature=(36.5, 37.8),
        gcs=(15, 15), pain_scale=(2, 5),
    ),
    5: dict(
        heart_rate=(60, 80), systolic_bp=(115, 130), diastolic_bp=(70, 82),
        respiratory_rate=(12, 16), spo2=(98.0, 100.0), temperature=(36.4, 37.2),
        gcs=(15, 15), pain_scale=(0, 3),
    ),
}

# Clinical note templates (SOAP format)
SOAP_TEMPLATES = [
    {
        "note": (
            "S: {age}yo {sex} presents with {complaint}. Reports {duration} of symptoms. "
            "Denies fever. Allergic to {allergy}.\n"
            "O: Vitals as recorded. General: {appearance}.\n"
            "A: Differential includes {diagnosis}. Ruled out {ruled_out}.\n"
            "P: Administer {medication}. Order {procedure}. Follow up with {follow_up} in {fu_days} days."
        ),
        "entities": lambda d: {
            "diagnoses": [d["diagnosis"]],
            "medications": [d["medication"]],
            "allergies": [d["allergy"]],
            "procedures": [d["procedure"]],
            "follow_up": [f"{d['follow_up']} in {d['fu_days']} days"],
        },
    },
    {
        "note": (
            "SUBJECTIVE: Patient is a {age} year old {sex}. Chief complaint: {complaint}. "
            "Duration: {duration}. Medications at home: {home_med}. "
            "No known drug allergies except {allergy}.\n"
            "OBJECTIVE: {appearance}. Labs pending.\n"
            "ASSESSMENT: {diagnosis}. Secondary concern: {ruled_out}.\n"
            "PLAN: Start {medication}. Perform {procedure}. Refer to {follow_up}."
        ),
        "entities": lambda d: {
            "diagnoses": [d["diagnosis"]],
            "medications": [d["medication"], d["home_med"]],
            "allergies": [d["allergy"]],
            "procedures": [d["procedure"]],
            "follow_up": [d["follow_up"]],
        },
    },
]

DIAGNOSES = [
    "Acute Myocardial Infarction", "STEMI", "Pneumonia", "Pulmonary Embolism",
    "Appendicitis", "Urinary Tract Infection", "Cellulitis", "Deep Vein Thrombosis",
    "COPD Exacerbation", "Asthma Exacerbation", "Migraine", "Hypertensive Crisis",
    "Anaphylaxis", "Syncope", "Atrial Fibrillation", "Sepsis", "Diabetic Ketoacidosis",
    "Ischemic Stroke", "Hemorrhagic Stroke", "Meningitis",
]
MEDICATIONS = [
    "Aspirin 325mg PO", "Nitroglycerin 0.4mg SL", "Morphine 4mg IV",
    "Ondansetron 4mg IV", "Normal Saline 1L IV", "Epinephrine 0.3mg IM",
    "Heparin infusion", "Metoprolol 5mg IV", "Dexamethasone 10mg IV",
    "Azithromycin 500mg PO", "Ceftriaxone 1g IV", "Albuterol 2.5mg nebulization",
    "Labetalol 20mg IV", "Lorazepam 2mg IV", "Acetaminophen 1g IV",
]
PROCEDURES = [
    "12-lead ECG", "Chest X-ray", "CT Head without contrast", "CT Chest with contrast",
    "CBC and BMP", "Troponin serial", "Urine culture", "Blood cultures x2",
    "Bedside ultrasound", "Lumbar puncture", "IV access x2", "Point-of-care glucose",
    "D-dimer", "BNP", "Arterial Blood Gas",
]
ALLERGIES = [
    "Penicillin", "Sulfa drugs", "Codeine", "NSAID", "Latex", "Contrast dye",
    "Erythromycin", "None known", "Morphine", "Aspirin",
]
SPECIALISTS = [
    "Cardiology", "Neurology", "Surgery", "Pulmonology", "Nephrology",
    "Hematology", "Orthopedics", "Gastroenterology", "Primary Care", "ENT",
]
APPEARANCES = [
    "Alert and oriented x3, in moderate distress",
    "Diaphoretic, pale, anxious",
    "Well-appearing, cooperative",
    "Lethargic but arousable",
    "In severe distress, unable to complete sentences",
    "Calm, mildly uncomfortable",
]
DURATIONS = [
    "30 minutes", "2 hours", "3 days", "1 week", "since this morning",
    "sudden onset", "gradually worsening over 24 hours",
]


# ─────────────────────────────────────────────
# Data Generator
# ─────────────────────────────────────────────

class PatientDataGenerator:
    """Generates synthetic patient scenarios for MediRoute tasks."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._patient_counter = 0

    def _next_id(self) -> str:
        self._patient_counter += 1
        return f"PT-{self._patient_counter:04d}"

    def _sample(self, lo, hi, is_float=False):
        if is_float:
            return round(self.rng.uniform(lo, hi), 1)
        return self.rng.randint(lo, hi)

    def generate_vitals(self, esi: int) -> VitalSigns:
        ranges = VITALS_BY_ESI[esi]
        return VitalSigns(
            heart_rate=self._sample(*ranges["heart_rate"]),
            systolic_bp=self._sample(*ranges["systolic_bp"]),
            diastolic_bp=self._sample(*ranges["diastolic_bp"]),
            respiratory_rate=self._sample(*ranges["respiratory_rate"]),
            spo2=self._sample(*ranges["spo2"], is_float=True),
            temperature=self._sample(*ranges["temperature"], is_float=True),
            gcs=self._sample(*ranges["gcs"]),
            pain_scale=self._sample(*ranges["pain_scale"]),
        )

    def _fill_template(self, age: int, sex: str, complaint: str) -> tuple[str, dict]:
        """Return (note_text, true_entities_dict)."""
        template = self.rng.choice(SOAP_TEMPLATES)
        data = {
            "age": age,
            "sex": sex,
            "complaint": complaint,
            "duration": self.rng.choice(DURATIONS),
            "appearance": self.rng.choice(APPEARANCES),
            "diagnosis": self.rng.choice(DIAGNOSES),
            "ruled_out": self.rng.choice(DIAGNOSES),
            "medication": self.rng.choice(MEDICATIONS),
            "home_med": self.rng.choice(MEDICATIONS),
            "procedure": self.rng.choice(PROCEDURES),
            "allergy": self.rng.choice(ALLERGIES),
            "follow_up": self.rng.choice(SPECIALISTS),
            "fu_days": self.rng.choice([3, 5, 7, 14, 30]),
        }
        note = template["note"].format(**data)
        entities = template["entities"](data)
        return note, entities

    def generate_patient(
        self,
        esi: Optional[int] = None,
        include_note: bool = False,
        include_arrival_time: bool = False,
        base_time: int = 0,
    ) -> tuple[PatientRecord, dict]:
        """
        Generate a synthetic patient.
        Returns (PatientRecord, true_entities_dict).
        true_entities is {} when include_note=False.
        """
        if esi is None:
            # Natural ESI distribution skewed toward 3-4 (most common)
            esi = self.rng.choices([1, 2, 3, 4, 5], weights=[2, 8, 35, 35, 20])[0]

        age = self.rng.randint(1, 95)
        sex = self.rng.choice(["M", "F"])
        complaint = self.rng.choice(CHIEF_COMPLAINTS[esi])
        vitals = self.generate_vitals(esi)

        note, true_entities = None, {}
        if include_note:
            note, true_entities = self._fill_template(age, sex, complaint)

        arrival = None
        if include_arrival_time:
            arrival = base_time + self.rng.randint(0, 30)

        patient = PatientRecord(
            patient_id=self._next_id(),
            age=age,
            sex=sex,
            chief_complaint=complaint,
            vitals=vitals,
            clinical_note=note,
            arrival_time_minutes=arrival,
            esi_level_true=esi,
        )
        return patient, true_entities

    def generate_resource_pool(self, n_patients: int) -> ResourcePool:
        """Generate a constrained resource pool for Task 3."""
        # Scarce: fewer resources than patients to force optimization
        factor = max(1, n_patients // 3)
        return ResourcePool(
            beds_available=self.rng.randint(factor, factor + 3),
            physicians_available=self.rng.randint(1, factor),
            nurses_available=self.rng.randint(factor, factor + 2),
            ct_scanners_available=self.rng.randint(1, 2),
            xray_available=self.rng.randint(1, 3),
            icu_beds_available=self.rng.randint(0, 2),
        )

    def reset_counter(self, seed: Optional[int] = None):
        self._patient_counter = 0
        if seed is not None:
            self.rng = random.Random(seed)
