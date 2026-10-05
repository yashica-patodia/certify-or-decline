"""CPU-only reproduction of the MATH-AI paper's principal results.

Run from the repository root: python -m analysis.reproduce --out results.json
No benchmark access, model calls, API keys, or third-party Python packages.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from experiments.cluster_intervals import cluster_ci, wilson
from experiments.proof_soundness import analyse as audit_summary
from experiments.token_accounting import collect as token_summary

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'experiments/public_runs'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm_summary(paths, *, bootstrap=False):
    by_problem = defaultdict(list)
    cert = correct = errors = declines = initial_correct = initial_n = 0
    per_run = []
    initial_runs = 0
    for name in paths:
        rows = read(PUB / name)
        graded_initial = sum('solver_initial_key_match' in r for r in rows)
        require(graded_initial in (0, len(rows)), f'Partially graded initial answers in {name}')
        initial_runs += bool(graded_initial)
        require(len({r['id'] for r in rows}) == len(rows), f'Duplicate IDs in {name}')
        run_cert = run_correct = 0
        for row in rows:
            selected = bool(row['verified']) and not row['error']
            if selected:
                require(isinstance(row['key_match'], bool), f'Missing grade: {name}/{row["id"]}')
            right = bool(row['key_match'])
            by_problem[row['id']].append((selected, right))
            cert += selected
            correct += selected and right
            errors += bool(row['error'])
            declines += not row['error'] and not row['verified']
            run_cert += selected
            run_correct += selected and right
            if 'solver_initial_key_match' in row:
                initial_n += 1
                initial_correct += bool(row['solver_initial_key_match'])
        per_run.append({'run':name, 'certified':run_cert, 'correct':run_correct,
                        'precision':round(100 * run_correct / run_cert, 1) if run_cert else None})
    observations = sum(map(len, by_problem.values()))
    sizes = {len(v) for v in by_problem.values()}
    require(len(sizes) == 1 and next(iter(sizes)) == len(paths), 'Incomplete repeated-run cohort')
    require(cert + errors + declines == observations, 'Termination counts do not sum')
    out = {'problems':len(by_problem), 'runs':len(paths), 'attempts':observations,
           'certified':cert, 'correct':correct, 'execution_errors':errors, 'declined':declines,
           'coverage':round(100 * cert / observations, 1),
           'precision':round(100 * correct / cert, 1), 'per_run':per_run}
    if initial_n:
        out.update(initial_correct=initial_correct, initial_attempts=initial_n, initial_runs=initial_runs,
                   initial_accuracy=round(100 * initial_correct / initial_n, 1),
                   gain=round(100 * correct / cert - 100 * initial_correct / initial_n, 1))
    if bootstrap:
        out.update(cluster_ci(dict(by_problem), 20000, 20260819))
    return out, dict(by_problem)


def tiers(paths):
    out = []
    for unconditional in (True, False):
        clusters = defaultdict(list)
        for name in paths:
            for row in read(PUB/name):
                selected = bool(row['verified']) and not row['error'] and bool(row['verified_unconditionally']) == unconditional
                clusters[row['id']].append((selected, bool(row['key_match'])))
        selected = [o for group in clusters.values() for o in group if o[0]]
        n = sum(map(len, clusters.values()))
        out.append({'tier':'unconditional' if unconditional else 'under conventions',
                    'certified':len(selected), 'correct':sum(o[1] for o in selected),
                    'coverage':round(100 * len(selected) / n, 1),
                    'precision':round(100 * sum(o[1] for o in selected) / len(selected), 1),
                    'precision_ci95_cluster':cluster_ci(dict(clusters), 20000, 20260819)['precision_ci95_cluster']})
    return out


def paired_voting(by_problem):
    votes = read(PUB/'solver_k30_votes.json')
    require(set(votes) == set(by_problem), 'Voting and verifier cohorts differ')
    baseline = {}
    for pid, row in votes.items():
        choices = row['draw_classes'][:8]
        require(len(choices) == 8, 'Fewer than eight draws')
        selected = None not in choices and len(set(choices)) == 1
        right = bool(row['draw_correct'][0]) if selected else False
        if selected:
            require(len(set(row['draw_correct'][:8])) == 1, 'Same vote class has conflicting correctness')
        baseline[pid] = (selected, right)
    sc_n = sum(v[0] for v in baseline.values())
    sc_correct = sum(v[0] and v[1] for v in baseline.values())
    verifier = {pid:(sum(o[0] for o in obs),sum(o[0] and o[1] for o in obs))
                for pid,obs in by_problem.items()}
    v_n = sum(v[0] for v in verifier.values())
    v_correct = sum(v[1] for v in verifier.values())
    # A fixed, explicitly sorted ID order makes this script reproducible.
    # Selection policies remain fixed in every resample; no intersection or
    # coverage retuning is performed.
    pids = sorted(by_problem)
    rng = random.Random(20260906)
    diffs = []
    for _ in range(20000):
        drawn = [rng.choice(pids) for _ in pids]
        vc = sum(verifier[p][1] for p in drawn)
        vn = sum(verifier[p][0] for p in drawn)
        sc = sum(baseline[p][0] and baseline[p][1] for p in drawn)
        sn = sum(baseline[p][0] for p in drawn)
        require(vn > 0 and sn > 0, 'Undefined precision in a paired resample')
        diffs.append(100 * (vc/vn - sc/sn))
    diffs.sort()
    return {'k':8,'vote_denominator':'all draws','selected':sc_n,'correct':sc_correct,
            'coverage':100 * sc_n / len(pids),'precision':round(100 * sc_correct/sc_n,1),
            'precision_ci95_wilson':wilson(sc_correct,sc_n),
            'paired_precision_difference':round(100 * (v_correct/v_n - sc_correct/sc_n),1),
            'paired_ci95':[round(diffs[int(.025*len(diffs))],1),round(diffs[int(.975*len(diffs))],1)],
            'resamples':20000,'seed':20260906,'rng':'Python random.Random','id_order':'sorted'}


def agreement(path, a, b):
    rows = read(path)['items']
    paired = [r for r in rows if (r.get(a) or {}).get('verdict') and (r.get(b) or {}).get('verdict')]
    x = [r[a]['verdict'] == 'VALID' for r in paired]
    y = [r[b]['verdict'] == 'VALID' for r in paired]
    n = len(paired)
    observed = sum(i == j for i,j in zip(x,y)) / n
    px, py = sum(x)/n, sum(y)/n
    chance = px * py + (1-px)*(1-py)
    return {'overlap':n,'a_accepted':sum(x),'b_accepted':sum(y),
            'kappa':round((observed-chance)/(1-chance),3)}


def rejections():
    corpus = read(ROOT/'analysis/rejection_events.json')
    events = [v for group in corpus for v in group['rejections']]
    issues = [i for v in events for i in v['issues']]
    classes = Counter(i['error_class'] for i in issues)
    roles = Counter(v['role'] for v in events)
    result = {'run_files_scanned':len(corpus),
              'run_files_with_rejections':sum(bool(g['rejections']) for g in corpus),
              'rejections':len(events), 'classified_issues':len(issues),
              'by_error_class':dict(classes),'by_role':dict(roles)}
    reference = read(PUB/'rejection_composition.json')
    for field in ['run_files_scanned','run_files_with_rejections','rejections','classified_issues']:
        require(result[field] == reference[field], f'Rejection corpus drift: {field}')
    require(classes == Counter({k:v['n'] for k,v in reference['by_error_class'].items()}), 'Rejection labels differ')
    return result


def construction():
    rows = read(ROOT/'analysis/construction_diagnostic.json')
    completed = [r for r in rows if not r['execution_error']]
    invalid = [r for r in completed if r['terminal_phase'] == 'formalizer_invalid']
    return {'attempts':len(rows),'completed':len(completed),
            'execution_errors':len(rows)-len(completed),'terminal_invalid':len(invalid),
            'terminal_invalid_never_judged':sum(not r['ever_judged'] for r in invalid),
            'completed_never_judged':sum(not r['ever_judged'] for r in completed),
            'proof_responses':sum(r['proof_responses'] for r in rows),
            'labelled_answer_slots':sum(r['labelled_answer_slots'] for r in rows),
            'completed_proof_responses':sum(r['proof_responses'] for r in completed),
            'completed_labelled_answer_slots':sum(r['labelled_answer_slots'] for r in completed)}


def constructed_grader_checks():
    out = {}
    for benchmark,name in [('GPQA','oss'),('AIME-2025','aime2025'),('MATH-500','math500')]:
        reference = read(PUB/f'grader_validation_{name}.json')
        cases = reference['cases']
        correct = sum(c['truth'] == c['verdict'] for c in cases)
        false_positive = sum(not c['truth'] and c['verdict'] is True for c in cases)
        false_negative = sum(c['truth'] and c['verdict'] is False for c in cases)
        require(len(cases) == reference['n_cases'] and false_positive == reference['false_positives'] and
                false_negative == reference['false_negatives'], f'Constructed grader checks differ: {benchmark}')
        out[benchmark] = {'cases':len(cases),'accuracy':round(100 * correct/len(cases),1),
                          'false_positives':false_positive,'false_negatives':false_negative}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    arm_files = read(ROOT/'analysis/arms.json')
    references = {r['label']:r for r in read(PUB/'arms_with_ci.json')}
    results = {'arms':{}}
    strong_clusters = None
    for name,paths in arm_files.items():
        result,clusters = arm_summary(paths,bootstrap=name in references)
        if name in references:
            for field in ['coverage','precision','coverage_ci95_cluster','precision_ci95_cluster']:
                require(result[field] == references[name][field], f'{name}: published {field} disagrees with records')
        results['arms'][name] = result
        if name == 'GPQA strong':strong_clusters = clusters
    results['certification_tiers'] = tiers(arm_files['GPQA strong'])
    for result,ref in zip(results['certification_tiers'],read(PUB/'certification_tiers.json')['tiers']):
        for field in ['precision','coverage','precision_ci95_cluster']:
            require(result[field] == ref[field], f'Certification tier differs: {field}')
    results['self_consistency'] = paired_voting(strong_clusters)
    require((results['self_consistency']['selected'],results['self_consistency']['correct']) == (35,29),'Voting operating point differs')
    results['tokens'] = token_summary([str(PUB/p) for p in arm_files['GPQA strong']])
    for key,value in read(PUB/'token_accounting.json').items():
        if key != 'note': require(results['tokens'][key] == value,f'Token accounting differs: {key}')
    results['external_audit'] = audit_summary(str(ROOT/'experiments/audit_proofs_strong_k3.json'))
    for field in ['grader_coverage','correct_answer_on_rejected_proof','codex_rejection_counts']:
        require(results['external_audit'][field] == read(PUB/'proof_soundness.json')[field], f'Audit differs: {field}')
    results['agreement'] = {
        'proprietary':agreement(ROOT/'experiments/audit_proofs_strong_k3.json','codex','claude'),
        '14b_20b':agreement(ROOT/'experiments/audit_proofs_strong_k3_oss.json','oss-gptoss','oss-qwen'),
        '32b_20b':agreement(ROOT/'experiments/audit_proofs_judgescale.json','oss-qwen32b','oss-gptoss')}
    require([a['kappa'] for a in results['agreement'].values()] == [.573,.038,-.005], 'Auditor agreement differs')
    results['rejections'] = rejections()
    results['construction_diagnostic'] = construction()
    require((results['construction_diagnostic']['completed'],results['construction_diagnostic']['terminal_invalid_never_judged'],
             results['construction_diagnostic']['proof_responses'],results['construction_diagnostic']['labelled_answer_slots']) == (88,52,277,117), 'Diagnostic differs')
    check = read(PUB/'scoring_consistency.json')
    require(len(check['rows']) == 72 and all(r['direct_key_match'] == r['grader_key_match'] for r in check['rows']), 'Actual-output scoring differs')
    results['scoring_consistency'] = check['counts']
    results['constructed_grader_checks'] = constructed_grader_checks()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(results,indent=2)+'\n')
    strong = results['arms']['GPQA strong']
    print(f"GPQA strong: {strong['correct']}/{strong['certified']} correct, {strong['coverage']}% coverage; CI {strong['precision_ci95_cluster']}")
    for row in results['certification_tiers']:
        print(f"{row['tier']}: {row['correct']}/{row['certified']}, {row['precision']}%; CI {row['precision_ci95_cluster']}")
    sc = results['self_consistency']
    print(f"Voting k=8: {sc['correct']}/{sc['selected']}; paired difference {sc['paired_precision_difference']} points, CI {sc['paired_ci95']}")
    print(f"Construction: {results['construction_diagnostic']}")
    print('All published arm intervals, token counts, rejection counts, audit counts, and agreement statistics reproduced.')
    if args.out: print(f'Report: {args.out}')


if __name__ == '__main__':
    main()
