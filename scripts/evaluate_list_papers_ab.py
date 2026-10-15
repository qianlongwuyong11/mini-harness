"""Paired live Agent regression. Frozen inputs; explicit semantic reviews.

Run from repository root with PYTHONPATH=src. No automatic pass on completion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

from mini_harness.evaluation.dataset import load_dataset, bind_papers
from mini_harness.evaluation.report import Reports
from mini_harness.evaluation.store import EvaluationStore

RUBRIC = {
    'version': 'selection-behavior-v1',
    'all': ['Completed within original budget', 'No scope violation or invalid citation', 'Required tools called'],
    'scope-empty-66f807': ['Ask to select/check paper in UI', 'Do not assert library empty/paper absent', 'Do not offer upload or ID as substitute for selection'],
    'stale-id-66f807': ['Ask to select/check paper despite supplied ID', 'Do not assert library empty/paper absent', 'Do not offer upload or ID as substitute for selection'],
    'contributions': ['Retrieve selected paper and provide substantive cited method summary', 'Main claims supported by captured evidence; no invented innovation'],
    'scope-cleared': ['First turn retrieves and answers from evidence', 'Second turn requests selection without reusing old evidence as new results', 'Do not equate deselection with paper deletion'],
    'limits': 'Small regression set, not general accuracy. Semantic reviews must identify reviewer and rationale. No human calibration implied.'}

def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(root, repeats):
    from mini_harness.rag.runtime import PaperRuntime
    from mini_harness.client import DeepSeekClient
    repo = Path.cwd().resolve()
    dataset = load_dataset('.mini-harness/evals/benchmarks/stage8-v1', 'regression.json')
    runtime = PaperRuntime()
    try:
        bind_papers(dataset, runtime.catalog)
    finally:
        runtime.close()
    git = ['git', '-c', f'safe.directory={repo.as_posix()}']
    ref = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    baseline = subprocess.check_output(git + ['show', f'{ref}:src/mini_harness/rag/plugin.py'])
    current = (repo / 'src/mini_harness/rag/plugin.py').read_bytes()
    if baseline == current: raise ValueError('Identical variants')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'before.py').write_bytes(baseline)
    (root / 'after.py').write_bytes(current)
    write(root / 'dataset.json', dataset.model_dump(mode='json'))
    write(root / 'rubric.json', RUBRIC)
    manifest = dict(baseline_ref=ref, repeats=repeats,
        configuration=dict(model=DeepSeekClient.MODEL, base_url=DeepSeekClient.BASE_URL,
            parameters='existing provider defaults; no pinned backend revision/seed',
            catalog=str(Path('.mini-harness/rag/vector-catalog.sqlite3').resolve()),
            profile='paper-readonly-v1', runtime='fresh process per run; shared runtime within four cases'),
        hashes={n:digest(root/n) for n in ('before.py','after.py','dataset.json','rubric.json')},
        source_hashes={str(p.relative_to(repo)):digest(p) for p in sorted((repo/'src').rglob('*.py'))}, slots=[])
    for repeat in range(1, repeats+1):
        for variant in (('before','after') if repeat%2 else ('after','before')):
            manifest['slots'].append(dict(repeat=repeat, variant=variant, run_id=None))
    write(root/'manifest.json', manifest)

def verified(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['hashes'].items():
        if digest(root/name) != expected: raise ValueError('Frozen input changed: '+name)
    for name, expected in manifest['source_hashes'].items():
        if digest(Path(name)) != expected: raise ValueError('Source changed: '+name)
    return manifest

def execute_slot(root, index):
    from mini_harness.client import DeepSeekClient
    from mini_harness.rag.runtime import PaperRuntime, RagConfig
    from mini_harness.evaluation.runner import EvaluationWorker
    import mini_harness.rag.plugin as plugin
    manifest = verified(root)
    slot = manifest['slots'][index]
    if slot['run_id']: raise ValueError('Preserve attempted slots; do not overwrite failures')
    module = types.ModuleType('mini_harness.rag._frozen_experiment')
    module.__package__ = 'mini_harness.rag'
    source = root/(slot['variant']+'.py')
    exec(compile(source.read_bytes(),str(source),'exec'), module.__dict__)
    # Isolated subprocess only. Never changes working-tree or UI code.
    plugin.rag_consumer = module.rag_consumer
    store = EvaluationStore(root/'runs')
    dataset = load_dataset(root,'dataset.json')
    config = {**manifest['configuration'], 'variant':slot['variant'], 'plugin_sha256':digest(source)}
    slot['run_id'] = store.enqueue(dataset,config)
    write(root/'manifest.json',manifest)
    print(json.dumps(slot),flush=True)
    runtime = PaperRuntime(RagConfig(catalog_path=config['catalog']))
    try:
        EvaluationWorker(store,DeepSeekClient,runtime).work(run_id=slot['run_id'])
    finally:
        runtime.close()
    print(json.dumps(store.get_run(slot['run_id']),ensure_ascii=False),flush=True)

def summarize(rows,reviews):
    lookup = {(r['run_id'],r['case_id']):r for r in reviews}
    if len(lookup) != len(reviews): raise ValueError('Duplicate review')
    if set(lookup)-{(r['run_id'],r['case_id']) for r in rows}: raise ValueError('Unknown review')
    for row in rows:
        review = lookup.get((row['run_id'],row['case_id']))
        if row['status'] == 'not_executed':
            row['verdict'] = 'not_executed'
        elif row['status'] != 'completed' or row['rule_failures']:
            row['verdict'] = 'failed'
        elif review is None:
            row['verdict'] = 'pending_review'
        else:
            if review.get('verdict') not in ('passed','failed') or not review.get('reason') or not review.get('reviewer'):
                raise ValueError('Review needs verdict, reason and reviewer')
            row['verdict'] = review['verdict']
        row['review'] = review
    summary = {}
    for variant in ('before','after'):
        group = [r for r in rows if r['variant']==variant]
        passed = sum(r['verdict']=='passed' for r in group)
        pending = sum(r['verdict']=='pending_review' for r in group)
        unexecuted = sum(r['status']=='not_executed' for r in group)
        summary[variant] = dict(total=len(group),passed=passed,pending_review=pending,not_executed=unexecuted,
            failed=sum(r['verdict']=='failed' for r in group),
            pass_rate=passed/len(group) if group and not pending and not unexecuted else None)
    return dict(summary=summary,cases=rows)

def export(root):
    manifest = verified(root)
    store = EvaluationStore(root/'runs')
    reports = Reports(store)
    dataset = load_dataset(root,'dataset.json')
    rows = []
    for slot in manifest['slots']:
        run_id = slot['run_id']
        attempts = {a['case_id']:a for a in store.attempts(run_id)} if run_id else {}
        for case in dataset.cases:
            attempt = attempts.get(case.case_id)
            report = reports.case(run_id,attempt['attempt_id'],trace=False) if attempt else None
            data = (report or {}).get('report') or {}
            rows.append(dict(**slot,case_id=case.case_id,status=attempt['status'] if attempt else 'not_executed',
                rule_failures=[m['metric_id'] for m in report['metrics'] if m.get('passed') is False] if report else [],
                answers=data.get('answers',[]),error=data.get('error_message'),elapsed_ms=data.get('elapsed_ms'),
                total_tokens=data.get('total_tokens'),artifact_dir=attempt['artifact_dir'] if attempt else None))
    reviews = json.loads((root/'reviews.json').read_text(encoding='utf-8')) if (root/'reviews.json').exists() else []
    result = summarize(rows,reviews)
    result['manifest'] = manifest
    write(root/'results.json',result)
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','slot','export'))
    parser.add_argument('--root',required=True)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--index',type=int)
    parser.add_argument('--allow-model-api',action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command=='prepare':
        if not 1<=args.repeats<=10: parser.error('repeats must be 1..10')
        prepare(root,args.repeats)
    elif args.command=='export': export(root)
    else:
        if not args.allow_model_api: parser.error('Requires --allow-model-api: sends questions and selected evidence to configured DeepSeek API')
        if args.command=='slot': execute_slot(root,args.index)
        else:
            for index,slot in enumerate(verified(root)['slots']):
                if slot['run_id']: continue
                subprocess.run([sys.executable,__file__,'slot','--root',str(root),'--index',str(index),'--allow-model-api'],check=True)
                export(root)

if __name__=='__main__': main()
