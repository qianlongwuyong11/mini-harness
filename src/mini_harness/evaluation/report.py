"""Read-only projections of persisted runs; no model calls or log repair."""
import csv
import html
import io
import json
from dataclasses import asdict

from ..persistence import JsonlStore
from ..telemetry import step_metrics, turn_metrics
from .metrics import read_artifact
from .registry import metric_catalog


class Reports:
    def __init__(self, store):
        self.store = store

    def history(self):
        with self.store.connection() as db:
            return [dict(r) for r in db.execute('SELECT * FROM runs ORDER BY created_at DESC LIMIT 200')]

    def directory(self, attempt):
        path = (self.store.root / attempt['artifact_dir']).resolve()
        if not path.is_relative_to(self.store.root):
            raise ValueError('Attempt outside evaluation directory')
        return path

    def attempt(self, run_id, attempt_id):
        return next(a for a in self.store.attempts(run_id) if a['attempt_id'] == attempt_id)

    def case(self, run_id, attempt_id, *, trace=True):
        attempt = self.attempt(run_id, attempt_id)
        directory = self.directory(attempt)
        warnings = []
        def read(root, ref):
            if not ref: return None
            try: return read_artifact(root, json.loads(ref) if isinstance(ref, str) else ref)
            except (ValueError, OSError) as error:
                warnings.append(f'报告不可用: {type(error).__name__}')
                return None
        report = read(directory / 'artifacts', attempt['report_ref'])
        scores = [{**s, 'report': read(directory / 'scores' / s['score_id'], s['report_ref'])}
                  for s in self.store.scores(attempt_id)]
        metrics = self.store.metrics(attempt_id)
        # Latest scoring attempt is authoritative; never silently reuse an older pass.
        latest = scores[-1]['report'] if scores else None
        if latest: metrics += latest.get('metrics', [])
        verdict = (latest or report or {}).get('verdict', 'incomplete')
        if any(m.get('passed') is False for m in metrics): verdict = 'failed'
        source = next(c for c in self.store.dataset(run_id).cases if c.case_id == attempt['case_id'])
        result = dict(attempt=attempt, case=source.model_dump(mode='json'), report=report,
                      scores=scores, metrics=metrics, verdict=verdict, warnings=warnings)
        with self.store.connection() as db:
            exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='human_reviews'").fetchone()
            result['reviews']=[dict(r) for r in db.execute('SELECT * FROM human_reviews WHERE attempt_id=? ORDER BY created_at',(attempt_id,))] if exists else []
        if trace:
            events = []
            path = directory / 'trace.jsonl'
            if path.exists():
                try:
                    if path.stat().st_size > 64 * 1024 * 1024: raise ValueError('Trace too large')
                    events, torn = JsonlStore(path)._read_records()
                    if torn is not None: warnings.append('正在写入或尾行不完整，仅展示完整事件')
                except (ValueError, OSError) as error:
                    warnings.append(f'轨迹不可用: {type(error).__name__}')
            result.update(events=[asdict(e) for e in events], steps=step_metrics(events), turns=turn_metrics(events))
        return result

    def run(self, run_id):
        run = self.store.get_run(run_id)
        dataset = self.store.dataset(run_id)
        attempts = self.store.attempts(run_id)
        latest = {a['case_id']: a for a in attempts}
        cases = [self.case(run_id, a['attempt_id'], trace=False) for a in latest.values()]
        aggregates = []
        for definition in metric_catalog():
            rows = [m for c in cases for m in c['metrics'] if m['metric_id'] == definition['metric_id']]
            valid = [m['value'] for m in rows if m['status'] == 'scored' and m['value'] is not None]
            aggregates.append(dict(**definition, samples=len(rows), scored=len(valid),
                mean=sum(valid)/len(valid) if valid else None,
                statuses={s:sum(m['status']==s for m in rows) for s in ('scored','missing_data','not_applicable','error')}))
        return dict(run=run, dataset=dataset.model_dump(mode='json'), cases=cases, attempts=attempts,
                    total=len(dataset.cases), finished=sum(c['attempt']['status'] in ('completed','failed','interrupted') for c in cases), metrics=aggregates)

    def artifact(self, run_id, attempt_id, ref, score_id=None):
        attempt = self.attempt(run_id, attempt_id)
        root = self.directory(attempt) / 'artifacts'
        if score_id is not None:
            if not any(s['score_id'] == score_id for s in self.store.scores(attempt_id)):
                raise ValueError('Unknown score')
            root = self.directory(attempt) / 'scores' / score_id
        # Content hashes and containment are checked before reading any bytes.
        return read_artifact(root, ref)

    def compare(self, left, right):
        a, b = self.run(left), self.run(right)
        warnings = []
        if a['run']['dataset_sha256'] != b['run']['dataset_sha256']:
            warnings.append('数据集或论文版本不同，不能直接归因比较')
        if a['run']['configuration_sha256'] != b['run']['configuration_sha256']:
            warnings.append('运行配置不同，请结合具体配置解释差异')
        versions = lambda r: {(m['metric_id'],m['metric_version']) for c in r['cases'] for m in c['metrics']}
        if versions(a) != versions(b): warnings.append('指标版本或可用指标不同')
        judge = lambda r: {s['configuration'] for c in r['cases'] for s in c['scores'][-1:]}
        if judge(a) != judge(b): warnings.append('Judge 配置/阈值不同，质量分数不可直接比较')
        before = {c['attempt']['case_id']:c for c in a['cases']}
        after = {c['attempt']['case_id']:c for c in b['cases']}
        rows = []
        for key in sorted(before.keys() | after.keys()):
            x,y = before.get(key),after.get(key)
            delta = {}
            for field in ('elapsed_ms','total_tokens'):
                v,w = (x or {}).get('report') or {}, (y or {}).get('report') or {}
                delta[field] = w.get(field,0)-v.get(field,0) if field in v and field in w and (field!='total_tokens' or v.get('usage_complete') and w.get('usage_complete')) else None
            rows.append(dict(case_id=key, before=(x or {}).get('verdict'), after=(y or {}).get('verdict'), delta=delta))
        metrics=[]
        old={m['metric_id']:m for m in a['metrics']}
        for m in b['metrics']:
            x=old.get(m['metric_id']) or {}
            metrics.append(dict(metric_id=m['metric_id'],unit=m['unit'],before=x.get('mean'),after=m['mean'],
                before_samples=x.get('scored',0),after_samples=m['scored'],
                delta=m['mean']-x['mean'] if m['mean'] is not None and x.get('mean') is not None else None))
        return dict(baseline=left,current=right,warnings=warnings,cases=rows,metrics=metrics)

    def export(self, run_id, format):
        report = self.run(run_id)
        if format == 'json': return json.dumps(report, ensure_ascii=False, indent=2), 'application/json'
        if format == 'html':
            return '<!doctype html><meta charset="utf-8"><title>评测报告</title><h1>Agent 评测报告</h1><pre>' + html.escape(json.dumps(report,ensure_ascii=False,indent=2)) + '</pre>', 'text/html; charset=utf-8'
        if format != 'csv': raise ValueError('Unknown export format')
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['case_id','metric','target','status','value','reason'])
        def safe(value):
            text = '' if value is None else str(value)
            return "'"+text if text.lstrip().startswith(('=','+','-','@','\t','\r')) else text
        for case in report['cases']:
            for m in case['metrics']:
                writer.writerow([safe(v) for v in (case['attempt']['case_id'],m['metric_id'],m['target_id'],m['status'],m['value'],m['reason'])])
        return '\ufeff'+stream.getvalue(), 'text/csv; charset=utf-8'
