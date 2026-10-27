"""Bounded read-only local status; no payloads, credentials or remote calls."""
import json
import re
import sqlite3
from pathlib import Path
from ..client import Tool
from ..cordis import depends
from .contracts import MemoryErrorCode


def safe_id(value):
    return value if value is None or (isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value)) else '[redacted]'


def job_status(path, target_id, scope, limit=20):
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('limit must be between 1 and 50')
    path = Path(path)
    if not path.exists():
        return dict(status='not_initialized', counts={}, jobs=[])
    try:
        db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)
        try:
            db.row_factory = sqlite3.Row
            db.execute('BEGIN')
            args = (target_id, scope.user_id, scope.project_id)
            where = 'target_id=? AND user_id=? AND project_id=?'
            counts = dict(db.execute('SELECT state, COUNT(*) FROM memory_jobs WHERE '+where+' GROUP BY state', args).fetchall())
            rows = db.execute('SELECT job_id,state,remote_task_id,error_code,retry_count,next_run_at,lease_expires_at,updated_at FROM memory_jobs WHERE '+where+' ORDER BY updated_at DESC,job_id LIMIT ?', (*args,limit)).fetchall()
            codes = {code.value for code in MemoryErrorCode}
            jobs = [dict(job_id=safe_id(r['job_id']), state=r['state'],
                remote_task_id=safe_id(r['remote_task_id']),
                error_code=r['error_code'] if r['error_code'] in codes else 'other' if r['error_code'] else None,
                retry_count=r['retry_count'], next_run_at=r['next_run_at'],
                lease_expires_at=r['lease_expires_at'], updated_at=r['updated_at']) for r in rows]
            return dict(status='ok', counts=counts, jobs=jobs, total=sum(counts.values()),
                        truncated=sum(counts.values())>len(jobs))
        finally:
            db.close()
    except sqlite3.Error:
        return dict(status='unavailable', error_code='local_status_read_failed')


@depends('tools', 'session')
def memory_status_plugin(ctx, config):
    read_config, sync_config = config
    def status(args):
        if not isinstance(args,dict) or set(args)-{'limit'}:
            raise ValueError('Only limit is accepted')
        limit = args.get('limit',20)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError('limit must be between 1 and 50')
        result = dict(read_enabled=read_config is not None, sync_enabled=sync_config is not None,
                      source='local_snapshot_not_remote_confirmation')
        if sync_config is not None:
            result['queue'] = job_status(sync_config.db_path,sync_config.target_id,sync_config.scope,limit)
            sync = ctx.require('memory_sync')
            result['runner'] = dict(scan_failed=sync.scan_error is not None,
                                    execution_failed=sync.runner.last_error is not None)
        latest = next((e for e in reversed(ctx.session.events) if e.type=='memory/observation'), None)
        # Explicit whitelist also protects reads of imported session logs.
        keys = {'elapsed_ms','cache_hit','hit_count','injected_count','injected_tokens',
                'budget_tokens','input_tokens','input_limit','evidence_tokens'}
        result['latest_projection'] = None if latest is None else {
            k:v for k,v in latest.data.items() if k in keys and type(v) in (int,float,bool)}
        recall = next((e for e in reversed(ctx.session.events) if e.type=='memory/recall'), None)
        result['latest_recall'] = None
        if recall is not None:
            elapsed = recall.data.get('elapsed_ms')
            result['latest_recall'] = dict(elapsed_ms=elapsed if type(elapsed) in (int,float) else None,
                error_code=recall.data.get('error_code') if recall.data.get('error_code') in {c.value for c in MemoryErrorCode} else None,
                observed_at=recall.ts)
        tool_status = result['latest_projection']
        if tool_status is not None:
            tool_status['observed_at'] = latest.ts
            reasons = {'input_overflow','scope_blocked','no_budget','no_query','projected'}
            tool_status['reason'] = latest.data.get('reason') if latest.data.get('reason') in reasons else 'other' 
        return json.dumps(result,ensure_ascii=False)
    tool = Tool('memory_status','查看本地记忆队列及最近召回预算指标；不查询远端、不重试任务，完成状态以本地记录为准。',
        {'type':'object','properties':{'limit':{'type':'integer','minimum':1,'maximum':50}},'additionalProperties':False},status)
    ctx.effect(lambda: ctx.tools.register(tool))
