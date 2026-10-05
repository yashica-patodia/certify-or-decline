"""Check release hashes and the redaction boundary, without model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from experiments.prepare_benchmarks import BENCHMARKS, manifest_for

ROOT = Path(__file__).resolve().parents[1]
ROW_KEYS = {'id','verified','verified_unconditionally','error','key_match','n_proof_steps',
            'justification_types','verdicts','metrics','repair_metrics','config_hash',
            'dataset_revision','benchmark_file_sha256','verifier_mode',
            'solver_initial_key_match','answer_grader_failed'}
CLASSES = {'hidden_premise','misapplied_theorem','fabricated_citation','arithmetic_error',
           'other','circular_reasoning','cosmetic',None}
ROLES = {'initial_state','citation','computation','problem_given','step_untyped',None}


def require(condition, message):
    if not condition:raise ValueError(message)


def metrics(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if isinstance(item,str):
                require(key.endswith('_sha256') and re.fullmatch('[0-9a-f]{64}',item), 'Free text in a metric')
            else:metrics(item)
    else:require(value is None or isinstance(value,(bool,int,float)), 'Non-numeric metric value')


def main():
    checked = 0
    for line in (ROOT/'MANIFEST.sha256').read_text().splitlines():
        expected,name = line.split('  ',1)
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, f'Hash mismatch: {name}')
        checked += 1
    cohorts = {name: {entry['id'] for entry in manifest_for(name)['entries']} for name in BENCHMARKS}
    require(not any(cohorts[a] & cohorts[b] for a in cohorts for b in cohorts if a != b), 'Cohort IDs overlap')
    rows_checked = 0
    for path in (ROOT/'experiments/public_runs').glob('benchmark_*.json'):
        rows=json.loads(path.read_text())
        ids = [row['id'] for row in rows]
        require(len(set(ids)) == len(ids), f'Duplicate run IDs in {path.name}')
        require(any(set(ids) == expected for expected in cohorts.values()), f'Run differs from a paper cohort: {path.name}')
        for row in rows:
            require(set(row) <= ROW_KEYS, f'Unexpected field in {path.name}')
            require(isinstance(row['id'],str) and re.fullmatch('[A-Za-z0-9_-]+',row['id']), 'Unexpected problem ID')
            require(row['error'] is None or re.fullmatch('[A-Za-z0-9_:@/]+',row['error']), 'Free-text error message')
            metrics(row.get('metrics'));metrics(row.get('repair_metrics'))
            require(all(t in ['citation','computation','problem_given',None] for t in row['justification_types']), 'Unexpected justification type')
            for verdict in row['verdicts']:
                require(set(verdict) <= {'step_number','role','accepted','error_classes'}, 'Unexpected verdict text')
                require(verdict['role'] in ROLES, 'Unknown verdict role')
                require(set(verdict['error_classes']) <= CLASSES, 'Unknown issue category')
            rows_checked += 1
    for path in (ROOT/'experiments').glob('audit_proofs_*.json'):
        for row in json.loads(path.read_text())['items']:
            for key,value in row.items():
                if isinstance(value,dict):
                    require(set(value)=={'verdict','critical_step'}, 'Auditor reason text retained')
                    require(value['verdict'] in ['VALID','GAP','ERROR','NONSEQUITUR',None], 'Unknown audit label')
                    require(value['critical_step'] is None or isinstance(value['critical_step'],int), 'Free text in audit step')
    for name in ['grader_validation_oss.json','grader_validation_aime2025.json','grader_validation_math500.json']:
        for case in json.loads((ROOT/'experiments/public_runs'/name).read_text())['cases']:
            require(set(case)=={'id','kind','truth','verdict'}, 'Constructed case contains benchmark text')
    for arm, names in json.loads((ROOT/'analysis/arms.json').read_text()).items():
        benchmark = 'gpqa' if arm.startswith('GPQA') else 'aime2025' if arm.startswith('AIME') else 'math500'
        for name in names:
            rows = json.loads((ROOT/'experiments/public_runs'/name).read_text())
            require({row['id'] for row in rows} == cohorts[benchmark], f'Arm/cohort mismatch: {arm}')
    votes = json.loads((ROOT/'experiments/public_runs/solver_k30_votes.json').read_text())
    require(set(votes) == cohorts['gpqa'], 'Voting cohort differs from GPQA manifest')
    diagnostic = json.loads((ROOT/'analysis/construction_diagnostic.json').read_text())
    require({row['id'] for row in diagnostic} == cohorts['gpqa'], 'Construction cohort differs from GPQA manifest')
    print(f'Verified {checked} file hashes, 3 manifests ({sum(map(len,cohorts.values()))} problems), and the redaction schema of {rows_checked} run rows; auditor explanations omitted.')


if __name__ == '__main__':main()
