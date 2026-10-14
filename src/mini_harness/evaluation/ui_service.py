"""Local evaluation UI coordinator. Explicit consent is scoped to one operation."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from .dataset import load_dataset, bind_papers, fingerprint, MAX_DATASET_BYTES
from .models import EvaluationDataset
from .report import Reports
from .runner import EvaluationWorker
from .store import EvaluationStore


class EvaluationUI:
    def __init__(self, root, runtime=None, client_factory=None):
        from ..client import DeepSeekClient
        self.store = EvaluationStore(root)
        self.reports = Reports(self.store)
        self.runtime = runtime
        self.client_factory = client_factory or DeepSeekClient
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='evaluation-ui')
        self.lock = threading.RLock()
        self.pending = {}
        self.active = None
        self.worker = None
        self.stop = threading.Event()
        self.error = None
        with self.store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS human_reviews (review_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, verdict TEXT NOT NULL, note TEXT NOT NULL, created_at REAL NOT NULL)')

    def config(self):
        from .scoring import JudgeConfig
        path = self.store.root / 'judge.json'
        judge = JudgeConfig.model_validate_json(path.read_text(encoding='utf-8-sig')).model_dump(mode='json') if path.exists() else None
        return dict(model=getattr(self.client_factory,'MODEL','configured-client'),
                    endpoint=getattr(self.client_factory,'BASE_URL','configured-client'), judge=judge,
                    judge_key_ready=bool(judge and os.environ.get(judge['api_key_env'])))

    def state(self):
        with self.lock:
            config=self.config()
            return dict(active=self.active, error=self.error, runs=self.reports.history(), config=config, config_sha256=fingerprint(config))

    def validate(self, data):
        name, text = data.get('filename'), data.get('content')
        if not isinstance(name,str) or Path(name).suffix.lower() not in ('.json','.jsonl') or not isinstance(text,str):
            raise ValueError('请选择 JSON/JSONL 数据集文件')
        if len(text.encode('utf-8')) > MAX_DATASET_BYTES: raise ValueError('数据集最多 8 MiB')
        with tempfile.TemporaryDirectory(dir=self.store.root) as folder:
            file = 'dataset'+Path(name).suffix.lower()
            Path(folder,file).write_text(text,encoding='utf-8')
            dataset = load_dataset(folder,file)
        if dataset.papers and self.runtime is None: raise ValueError('论文服务未启用')
        bind_papers(dataset,self.runtime.catalog if self.runtime else None)
        identity = fingerprint(dataset.model_dump(mode='json'))
        with self.lock:
            if len(self.pending)>=8: self.pending.pop(next(iter(self.pending)))
            self.pending[identity]=dataset
        return dict(id=identity,dataset=dataset.model_dump(mode='json'), config=self.config(), config_sha256=fingerprint(self.config()), warnings=[
            '没有参考答案或 chunk 标注的指标将显示缺失；不会填零。',
            '同步模型请求采用协作取消，正在执行的请求需等待返回或超时。',
            '语义评分未设阈值时不判定通过，后续需人工校准。'])

    def _consent(self, data, judge, *, model=True):
        if model and data.get('allow_model_api') is not True: raise ValueError('需要明确授权本次问题和论文证据发送到模型 API')
        if judge and data.get('allow_judge_api') is not True: raise ValueError('需要明确授权本次问题、答案、参考答案和证据发送到 Judge API')
        if data.get('config_sha256') != fingerprint(self.config()): raise ValueError('配置已变更，请重新校验/查看配置后授权')

    def _dispatch(self, run_id, judge, rescore=False):
        self.active = dict(run_id=run_id, action='rescore' if rescore else 'execute')
        self.error = None
        self.stop.clear()
        worker = EvaluationWorker(self.store,self.client_factory,self.runtime,allow_judge_api=bool(judge))
        self.worker = worker
        def perform():
            try:
                if rescore:
                    from .scoring import SemanticScorer
                    with self.store.worker_lock():
                        self.store.recover()
                        scorer = SemanticScorer(self.store,judge,allow_api=True)
                        latest = {a['case_id']:a for a in self.store.attempts(run_id)}
                        for attempt in latest.values():
                            if self.stop.is_set(): break
                            if attempt['status']=='completed': scorer.score(attempt,cancelled=self.stop.is_set)
                else: worker.work(run_id=run_id)
            except Exception as error:
                # No key, question or remote response in the global UI error banner.
                self.error = f'评测未启动或中断: {type(error).__name__}；请检查运行状态，其他进程可能持有评测锁。'
            finally:
                with self.lock: self.active=None; self.worker=None
        self.pool.submit(perform)

    def start(self, data):
        with self.lock:
            if self.active: raise RuntimeError('已有评测或重评分正在执行')
            dataset = self.pending.get(data.get('dataset_id'))
            if dataset is None: raise ValueError('请先校验所选文件')
            mode = data.get('mode','live_rules')
            if mode not in ('live_rules','live_judge'): raise ValueError('未知执行模式')
            judge = self.config()['judge'] if mode=='live_judge' else None
            if mode=='live_judge' and (not judge or not self.config()['judge_key_ready']): raise ValueError('请在评测目录配置 judge.json，并在服务环境变量配置其 API key')
            self._consent(data,judge)
            bind_papers(dataset,self.runtime.catalog if self.runtime else None)
            config = dict(profile='paper-readonly-v1', model=self.config()['model'], endpoint=self.config()['endpoint'])
            if self.runtime: config['catalog']=str(Path(self.runtime.config.catalog_path).resolve())
            if judge: config['judge']=judge
            run_id=self.store.enqueue(dataset,config,mode=mode)
            self._dispatch(run_id,judge)
            return dict(run_id=run_id)

    def action(self, action, data):
        if action=='validate': return self.validate(data)
        if action=='start': return self.start(data)
        run_id=data.get('run_id')
        run=self.store.get_run(run_id)
        with self.lock:
            if action=='cancel':
                self.store.cancel(run_id)
                if self.active and self.active['run_id']==run_id:
                    self.stop.set()
                    if self.worker: self.worker.close()
                return dict(cancelled=True)
            if action in ('retry','rescore'):
                if self.active: raise RuntimeError('已有评测正在执行')
                if run['status'] in ('running','cancelling'): raise ValueError('运行仍活跃；若服务异常退出，请先执行恢复状态')
                judge = self.config()['judge'] if action=='rescore' else json.loads(run['configuration']).get('judge')
                if action=='rescore' and not judge: raise ValueError('请先配置 judge.json')
                if judge and judge != self.config()['judge']: raise ValueError('历史 judge 配置已变更，请新建运行或单独重评分')
                self._consent(data,judge,model=action!='rescore')
                if action=='retry' and run['status']!='queued': self.store.retry(run_id)
                self._dispatch(run_id,judge,rescore=action=='rescore')
                return dict(run_id=run_id)
            if action=='recover':
                if self.active: raise RuntimeError('仍有评测正在执行')
                with self.store.worker_lock(): self.store.recover()
                return dict(recovered=True)
            if action=='artifact':
                return self.reports.artifact(run_id,data.get('attempt_id'),data.get('ref'),data.get('score_id'))
            if action=='review':
                attempt=self.reports.attempt(run_id,data.get('attempt_id'))
                verdict,note=data.get('verdict'),data.get('note','')
                if verdict not in ('passed','failed','incomplete') or not isinstance(note,str) or len(note)>8000: raise ValueError('复核内容无效')
                with self.store.connection() as db:
                    db.execute('INSERT INTO human_reviews VALUES (?,?,?,?,?,?)',(uuid4().hex,run_id,attempt['attempt_id'],verdict,note,time.time()))
                return dict(saved=True)
            if action=='draft':
                ids=data.get('case_ids')
                dataset=self.store.dataset(run_id)
                if not isinstance(ids,list) or not ids or any(not isinstance(i,str) for i in ids) or not set(ids)<={c.case_id for c in dataset.cases}: raise ValueError('请选择有效的 case')
                draft=dataset.model_dump(mode='json')
                draft['dataset_id']=uuid4().hex
                draft['cases']=[dict(c,split='regression') for c in draft['cases'] if c['case_id'] in ids]
                return EvaluationDataset.model_validate(draft).model_dump(mode='json')
        raise ValueError('未知评测操作')

    def close(self):
        self.stop.set()
        if self.worker: self.worker.close()
        self.pool.shutdown(wait=True)
