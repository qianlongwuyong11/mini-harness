"""Local numeric telemetry and bounded UI previews of existing tool events."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_sink = ContextVar('mini_harness_metric_sink',default=None)

@contextmanager
def metric_scope(session, **identity):
    token=_sink.set((session, identity))
    try: yield
    finally: _sink.reset(token)

def record_metric(kind,data):
    sink=_sink.get()
    if sink is not None:
        session, identity = sink
        session.append(kind, {**data, **identity})


def turn_metrics(events):
    """Turn start event IDs remain unique even after an Agent is recreated."""
    turns=[];current=None
    for event in events:
        d=event.data
        if event.type=='turn/start':
            current=dict(id=event.id,number=len(turns)+1,started_at=event.ts,status='running',
                instrumented=d.get('metrics_version',0)>=1,model_calls=[],tools=[],rag=[],runtime=[],evidence=[],citation_repairs=0,retries=0,retry_wait_ms=0,citation_count=0,question='')
            turns.append(current)
        if current is None:continue
        current['end_event_id']=event.id
        if event.type=='user/message' and not current['question']:current['question']=(d.get('content') or '')[:160]
        elif event.type=='metrics/model':current['model_calls'].append(dict(d))
        elif event.type=='metrics/tool':current['tools'].append(dict(d))
        elif event.type=='rag/trace':current['rag'].append(dict(d))
        elif event.type=='rag/runtime':current['runtime'].append(dict(d))
        elif event.type=='rag/evidence':current['evidence'].append(dict(d))
        elif event.type=='rag/repair':current['citation_repairs']+=1
        elif event.type=='llm/retry':
            current['retries']+=1;current['retry_wait_ms']+=d.get('delay_ms',0)
        elif event.type=='rag/citations':current['citation_count']+=len(d.get('ids',()))
        elif event.type=='turn/end':
            current['status']=d.get('reason','completed')
            current['elapsed_ms']=d.get('elapsed_ms',max(0,(event.ts-current['started_at'])*1000))
            current=None
    for row in turns:
        _summarize(row)
    return turns


def _summarize(row):
    calls=row['model_calls'];tools=row['tools']
    row['elapsed_ms']=row.get('elapsed_ms',max(0,(time.time()-row['started_at'])*1000))
    row['model_count']=len(calls);row['model_ms']=sum(c['elapsed_ms'] for c in calls)
    row['model_errors']=sum(c['status']=='error' for c in calls)
    row['tool_count']=len(tools);row['tool_errors']=sum(c['status']=='error' for c in tools)
    row['extra_model_calls']=len(calls)-len({c['step'] for c in calls})
    row['usage_reported_calls']=sum(c.get('usage') is not None for c in calls)
    totals={}
    for key in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens'):
        values=[c['usage'][key] for c in calls if c.get('usage') is not None and key in c['usage']]
        totals[key]=sum(values) if values else None
    row['tokens']=totals
    row['usage_complete']=bool(calls) and all(c.get('usage') is not None and all(k in c['usage'] for k in ('prompt_tokens','completion_tokens','total_tokens')) for c in calls)


def step_metrics(events):
    """Project step boundaries; stable event IDs survive Agent restart and retries.

    Tool previews are bounded; original outputs remain in the existing JSONL log.
    RAG events inherit call_id from the tool scope, with sequential legacy fallback.
    """
    rows = []
    current = None
    turn_id = None
    turn_number = 0
    instrumented = False
    active_call = None
    calls = {}
    now = time.time()

    def finish_incomplete():
        if current is not None and current['status'] == 'running':
            current['status'] = 'interrupted'
            current['outcome'] = 'interrupted'

    for event in events:
        data = event.data
        if event.type == 'turn/start':
            finish_incomplete()
            current = None
            turn_id = event.id
            turn_number += 1
            instrumented = data.get('metrics_version', 0) >= 1
        elif event.type == 'step/start':
            finish_incomplete()
            current = dict(id=event.id, turn_id=turn_id, turn_number=turn_number,
                number=data.get('step', 1), started_at=event.ts, end_event_id=event.id,
                status='running', outcome='running', instrumented=instrumented,
                model_calls=[], tools=[], rag=[], runtime=[], evidence=[],
                citation_repairs=0, retries=0, retry_wait_ms=0, citation_count=0,
                tool_calls=[], error_type=None, timing_source='event_timestamps')
            rows.append(current)
            calls = {}
            active_call = None
        if current is None:
            continue
        kind = event.type
        # Replacement tool messages belong to model context pruning, not execution.
        if data.get('surface_op'):
            continue
        current['end_event_id'] = event.id
        if kind == 'tool/call':
            active_call = data['call_id']
            arguments = data.get('arguments', '')
            call = dict(call_id=active_call, name=data.get('name', ''), status='running',
                arguments_preview=arguments[:1200], arguments_chars=len(arguments),
                result_preview=None, result_chars=None, result_event_id=None)
            calls[active_call] = call
            current['tool_calls'].append(call)
        elif kind == 'tool/result':
            call = calls.get(data.get('call_id'))
            if call is not None:
                content = data.get('content') or ''
                call.update(status='error' if data.get('is_error') else 'ok',
                    result_preview=content[:1200], result_chars=len(content), result_event_id=event.id)
        elif kind == 'metrics/model':
            current['model_calls'].append(dict(data))
        elif kind == 'metrics/tool':
            current['tools'].append(dict(data))
            call = calls.get(data.get('call_id'))
            if call is not None:
                call.update(status=data['status'], elapsed_ms=data['elapsed_ms'], error_type=data.get('error_type'))
            active_call = None
        elif kind in ('rag/trace', 'rag/runtime', 'rag/evidence'):
            target = {'rag/trace':'rag','rag/runtime':'runtime','rag/evidence':'evidence'}[kind]
            current[target].append({**data, 'call_id': data.get('call_id', active_call)})
        elif kind == 'rag/repair':
            current['citation_repairs'] += 1
        elif kind == 'rag/citations':
            current['citation_count'] += len(data.get('ids', ()))
        elif kind == 'llm/retry':
            current['retries'] += 1
            current['retry_wait_ms'] += data.get('delay_ms', 0)
        elif kind == 'step/end':
            current.update(status=data.get('status', 'completed'), outcome=data.get('outcome', 'unknown'),
                error_type=data.get('error_type'),
                elapsed_ms=data.get('elapsed_ms', max(0, (event.ts-current['started_at'])*1000)),
                timing_source='monotonic' if 'elapsed_ms' in data else 'event_timestamps')
            if data.get('recovered') or data.get('reason') in ('interrupted', 'crashed'):
                current.update(status='interrupted', outcome='interrupted')
            current = None
        elif kind == 'turn/end':
            finish_incomplete()
            if current is not None:
                current['elapsed_ms'] = max(0, (event.ts-current['started_at'])*1000)
            current = None
    for row in rows:
        if row['status'] == 'running':
            row['elapsed_ms'] = max(0, (now-row['started_at'])*1000)
        elif 'elapsed_ms' not in row:
            row['elapsed_ms'] = 0
        _summarize(row)
        if row['status'] == 'completed' and row['model_count'] and row['model_errors'] == row['model_count']:
            row['status'] = 'error'
        if row['status'] == 'completed' and row['tool_errors']:
            row['status'] = 'completed_with_errors'
    return rows
