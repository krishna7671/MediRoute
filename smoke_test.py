from env.environment import MediRouteEnv
from env.models import Action, PatientClassification, ExtractedEntities, ResourceAssignment

print('=== MediRoute End-to-End Smoke Test ===')

env = MediRouteEnv(seed=42)

# Task 1: Vitals Triage
obs = env.reset('vitals_triage')
print(f'Task 1 Reset OK - {len(obs.patients)} patients, task_id={obs.task_id}')
assert obs.task_id == 'vitals_triage'
assert len(obs.patients) == 5
assert all(p.esi_level_true is None for p in obs.patients), 'Ground truth leaking!'

action = Action(task_id='vitals_triage', classifications=[
    PatientClassification(patient_id=p.patient_id, esi_level=3) for p in obs.patients
])
obs2, reward, done, info = env.step(action)
print(f'Task 1 Step OK  - score={reward.total:.4f}, done={done}')
assert -1.0 <= reward.total <= 1.0
assert done is True

# Task 2: Clinical Extraction
obs = env.reset('clinical_extraction')
print(f'Task 2 Reset OK - {len(obs.patients)} patients with notes')
assert all(p.clinical_note for p in obs.patients), 'Notes missing!'

action = Action(task_id='clinical_extraction', extractions=[
    ExtractedEntities(patient_id=p.patient_id, diagnoses=['Test'], medications=['Test med'])
    for p in obs.patients
])
_, reward, done, info = env.step(action)
print(f'Task 2 Step OK  - score={reward.total:.4f}, done={done}')

# Task 3: Resource Optimization
obs = env.reset('resource_optimization')
print(f'Task 3 Reset OK - {len(obs.patients)} patients, resources OK')
assert obs.resources is not None

assignments = [ResourceAssignment(patient_id=p.patient_id, assigned_bed=True,
    assigned_physician=True, assigned_nurse=True, priority_rank=i+1)
    for i, p in enumerate(obs.patients[:3])]
action = Action(task_id='resource_optimization', assignments=assignments)
_, reward, done, info = env.step(action)
print(f'Task 3 Step OK  - score={reward.total:.4f}, done={done}')

# State test
state = env.state()
assert 'current_task' in state
assert 'episode_count' in state
ep = state['episode_count']
print(f'State OK        - episode_count={ep}')

print()
print('ALL SMOKE TESTS PASSED')
