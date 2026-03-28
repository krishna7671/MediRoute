"""
MediRoute — Reward functions.
Provides dense, multi-component reward signals with partial progress.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import (
    Action,
    Reward,
    RewardBreakdown,
    PatientRecord,
    ResourcePool,
)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Cost of misclassification by true ESI level
# ESI-1 patients misclassified → catastrophic; ESI-5 → minimal harm
ESI_MISCLASS_PENALTY = {1: 0.8, 2: 0.5, 3: 0.25, 4: 0.1, 5: 0.05}

# Partial credit for adjacent ESI prediction
ADJACENT_CREDIT = 0.5


# ─────────────────────────────────────────────
# Task 1: Vitals Triage Reward
# ─────────────────────────────────────────────

def compute_triage_reward(
    action: Action,
    patients: List[PatientRecord],
) -> Reward:
    """
    Grades ESI level classification.
    - Exact match: +1.0 per patient
    - Adjacent (±1): +0.5 per patient
    - Far off (≥2): penalty based on true ESI severity
    """
    if not action.classifications:
        return Reward(
            total=-0.3,
            breakdown=RewardBreakdown(penalties=-0.3),
            feedback="No classifications provided. Action must include `classifications` list.",
            critical_error=False,
        )

    patient_map = {p.patient_id: p for p in patients}
    total_score = 0.0
    penalties = 0.0
    critical = False
    n = len(action.classifications)

    for clf in action.classifications:
        patient = patient_map.get(clf.patient_id)
        if patient is None or patient.esi_level_true is None:
            continue
        true_esi = patient.esi_level_true
        pred_esi = clf.esi_level
        diff = abs(true_esi - pred_esi)

        if diff == 0:
            total_score += 1.0
        elif diff == 1:
            total_score += ADJACENT_CREDIT
        else:
            # Penalise based on severity of the TRUE ESI level being missed
            penalty = ESI_MISCLASS_PENALTY[true_esi]
            penalties -= penalty
            if true_esi <= 2 and pred_esi >= 4:
                critical = True

    # Penalise for missing patients
    provided_ids = {c.patient_id for c in action.classifications}
    for p in patients:
        if p.patient_id not in provided_ids:
            penalties -= ESI_MISCLASS_PENALTY.get(p.esi_level_true or 3, 0.2)
            if (p.esi_level_true or 5) <= 2:
                critical = True

    accuracy = total_score / max(n, 1)
    raw_total = accuracy + penalties / max(n, 1)
    clipped_total = max(-1.0, min(1.0, raw_total))

    msg_parts = [f"Scored {total_score:.1f}/{n} patients correctly."]
    if critical:
        msg_parts.append("⚠ CRITICAL: Life-threatening patient(s) severely misclassified!")
    if penalties < 0:
        msg_parts.append(f"Penalty: {penalties:.2f} for misclassifications/missing patients.")

    return Reward(
        total=round(clipped_total, 4),
        breakdown=RewardBreakdown(
            accuracy=round(accuracy, 4),
            partial_credit=round(max(0, total_score - n) / max(n, 1), 4),
            penalties=round(penalties / max(n, 1), 4),
        ),
        feedback=" ".join(msg_parts),
        critical_error=critical,
    )


# ─────────────────────────────────────────────
# Task 2: Clinical Extraction Reward
# ─────────────────────────────────────────────

def _token_set(text: str) -> set:
    """Lowercase tokenization for fuzzy string matching."""
    return set(text.lower().replace(",", " ").replace(".", " ").split())


def _f1(predicted: List[str], true: List[str]) -> float:
    """Token-level F1 for a list of strings."""
    if not true and not predicted:
        return 1.0
    if not true or not predicted:
        return 0.0

    true_tokens = set()
    for t in true:
        true_tokens |= _token_set(t)

    pred_tokens = set()
    for p in predicted:
        pred_tokens |= _token_set(p)

    tp = len(true_tokens & pred_tokens)
    precision = tp / len(pred_tokens) if pred_tokens else 0.0
    recall = tp / len(true_tokens) if true_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_extraction_reward(
    action: Action,
    true_entities_map: Dict[str, Dict[str, List[str]]],
) -> Reward:
    """
    Grades clinical entity extraction using per-category token F1.
    Categories: diagnoses, medications, allergies, procedures, follow_up
    """
    if not action.extractions:
        return Reward(
            total=-0.2,
            breakdown=RewardBreakdown(penalties=-0.2),
            feedback="No extractions provided. Action must include `extractions` list.",
            critical_error=False,
        )

    categories = ["diagnoses", "medications", "allergies", "procedures", "follow_up"]
    all_f1 = []
    penalties = 0.0

    for ext in action.extractions:
        true = true_entities_map.get(ext.patient_id, {})
        if not true:
            penalties -= 0.1  # wrong patient_id
            continue

        patient_f1s = []
        for cat in categories:
            pred_list = getattr(ext, cat, []) or []
            true_list = true.get(cat, [])
            f1 = _f1(pred_list, true_list)
            patient_f1s.append(f1)
        all_f1.append(sum(patient_f1s) / len(patient_f1s))

    if not all_f1:
        return Reward(
            total=-0.2,
            breakdown=RewardBreakdown(penalties=-0.2),
            feedback="No valid extractions matched any patient IDs.",
            critical_error=False,
        )

    avg_f1 = sum(all_f1) / len(all_f1)
    clipped = max(-1.0, min(1.0, avg_f1 + penalties))

    return Reward(
        total=round(clipped, 4),
        breakdown=RewardBreakdown(
            extraction_f1=round(avg_f1, 4),
            penalties=round(penalties, 4),
        ),
        feedback=(
            f"Mean entity F1: {avg_f1:.3f} across {len(all_f1)} patient(s). "
            f"Categories scored: {', '.join(categories)}."
        ),
        critical_error=False,
    )


# ─────────────────────────────────────────────
# Task 3: Resource Optimization Reward
# ─────────────────────────────────────────────

def compute_resource_reward(
    action: Action,
    patients: List[PatientRecord],
    resources: ResourcePool,
) -> Reward:
    """
    Grades multi-patient resource assignment using:
    1. Priority alignment: ESI-1/2 patients should have priority rank 1-2
    2. Resource constraint compliance: no over-assignment
    3. Coverage: critical patients must get physician + bed
    4. Utilization efficiency: maximize resource use
    """
    if not action.assignments:
        return Reward(
            total=-0.3,
            breakdown=RewardBreakdown(penalties=-0.3),
            feedback="No assignments provided. Action must include `assignments` list.",
            critical_error=False,
        )

    patient_map = {p.patient_id: p for p in patients}
    n = len(patients)
    penalties = 0.0
    critical = False

    # Check resource constraint compliance
    beds_used = sum(1 for a in action.assignments if a.assigned_bed)
    phys_used = sum(1 for a in action.assignments if a.assigned_physician)
    nurse_used = sum(1 for a in action.assignments if a.assigned_nurse)
    ct_used = sum(1 for a in action.assignments if a.assigned_imaging == "ct")
    xray_used = sum(1 for a in action.assignments if a.assigned_imaging == "xray")
    icu_used = sum(1 for a in action.assignments if a.assigned_icu)

    if beds_used > resources.beds_available:
        penalties -= 0.2
    if phys_used > resources.physicians_available:
        penalties -= 0.2
    if nurse_used > resources.nurses_available:
        penalties -= 0.1
    if ct_used > resources.ct_scanners_available:
        penalties -= 0.15
    if xray_used > resources.xray_available:
        penalties -= 0.1
    if icu_used > resources.icu_beds_available:
        penalties -= 0.2

    # Priority alignment scoring
    # Sort patients by ESI (1 = most urgent = should be rank 1)
    sorted_by_esi = sorted(
        [p for p in patients if p.esi_level_true is not None],
        key=lambda p: p.esi_level_true
    )
    esi_to_expected_rank = {p.patient_id: i + 1 for i, p in enumerate(sorted_by_esi)}

    priority_scores = []
    coverage_score = 0.0

    for assign in action.assignments:
        patient = patient_map.get(assign.patient_id)
        if patient is None:
            penalties -= 0.05
            continue

        expected_rank = esi_to_expected_rank.get(assign.patient_id, n)
        rank_diff = abs(assign.priority_rank - expected_rank)
        # Normalised priority score: 1.0 if exact, 0.0 if maximally wrong
        rank_score = max(0.0, 1.0 - rank_diff / max(n, 1))
        priority_scores.append(rank_score)

        # Critical patient coverage: ESI 1-2 MUST get bed + physician
        if patient.esi_level_true and patient.esi_level_true <= 2:
            if assign.assigned_bed and assign.assigned_physician:
                coverage_score += 1.0
            elif assign.assigned_bed or assign.assigned_physician:
                coverage_score += 0.5
            else:
                coverage_score += 0.0
                penalties -= 0.25
                critical = True

    priority_mean = sum(priority_scores) / max(len(priority_scores), 1)
    n_critical = sum(1 for p in patients if (p.esi_level_true or 5) <= 2)
    cov_score = coverage_score / max(n_critical, 1) if n_critical > 0 else 1.0

    # Resource utilization efficiency (maximize used / available)
    util = (beds_used / max(resources.beds_available, 1)) * 0.5 + \
           (phys_used / max(resources.physicians_available, 1)) * 0.5
    util = min(1.0, util)

    composite = 0.4 * priority_mean + 0.4 * cov_score + 0.2 * util
    raw_total = composite + penalties
    clipped = max(-1.0, min(1.0, raw_total))

    msg_parts = [
        f"Priority alignment: {priority_mean:.2f}.",
        f"Critical patient coverage: {cov_score:.2f}.",
        f"Resource utilization: {util:.2f}.",
    ]
    if penalties < 0:
        msg_parts.append(f"Constraint violation penalties: {penalties:.2f}.")
    if critical:
        msg_parts.append("⚠ CRITICAL: ESI-1/2 patient(s) left without essential resources!")

    return Reward(
        total=round(clipped, 4),
        breakdown=RewardBreakdown(
            accuracy=round(priority_mean, 4),
            resource_efficiency=round(util, 4),
            penalties=round(penalties, 4),
        ),
        feedback=" ".join(msg_parts),
        critical_error=critical,
    )
