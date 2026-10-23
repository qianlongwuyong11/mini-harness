"""P1-12 手动批次实验入口：预览默认，--submit 显式写入；无自动恢复或重试。"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from ..persistence import JsonlStore
from .capture import capture_messages
from .contracts import MemoryBackendError, MemoryScope, TaskState
from .openviking_backend import OpenVikingBackend


def prepare_batch(path, start, end):
    session = JsonlStore(path).read_snapshot()
    messages = capture_messages(session.session_id, session.events, start=start, end=end)
    selected = session.events[start:end]
    # 实验只接受完整成功轮次，避免半轮输出或恢复合成收尾被误提交。
    active = False
    for event in selected:
        if event.type == 'turn/start':
            if active:
                raise ValueError('Nested turn boundary')
            active = True
        elif event.type == 'turn/end':
            if not active or event.data.get('reason') != 'completed':
                raise ValueError('Only successfully completed turns can be submitted')
            active = False
        elif not active:
            raise ValueError('Range must contain complete turns')
    if active or not selected or selected[0].type != 'turn/start' or selected[-1].type != 'turn/end':
        raise ValueError('Range must start and end at completed turn boundaries')
    if not 1 <= len(messages) <= 100:
        raise ValueError('Select a range containing 1 to 100 eligible messages')
    payload = json.dumps([asdict(m) for m in messages], ensure_ascii=False, sort_keys=True)
    metadata = dict(session_id=session.session_id, start=start, end=end,
                    message_count=len(messages), source_message_ids=[m.source_message_id for m in messages],
                    payload_sha256=hashlib.sha256(payload.encode()).hexdigest())
    return metadata, messages


def submit_batch(backend, scope, metadata, messages, record, *, query, max_polls=90, interval=2):
    """record 在每次远端写入之前落盘；写请求绝不自动重试。"""
    if max_polls < 1 or interval < 0 or not query.strip():
        raise ValueError('Invalid polling or query configuration')
    stage = 'append'
    try:
        record(dict(stage='append_started'))
        appended = backend.append(scope, metadata['session_id'], messages)
        record(dict(stage='appended', result=asdict(appended)))
        stage = 'commit'
        record(dict(stage='commit_started'))
        committed = backend.commit(scope, metadata['session_id'])
        record(dict(stage='committed', result=asdict(committed)))
        if committed.status != 'accepted':
            return 'noop'
        stage = 'poll'
        for attempt in range(max_polls):
            task = backend.get_task(scope, committed.task_id)
            record(dict(stage='task', result=asdict(task)))
            if task.state == TaskState.COMPLETED:
                stage = 'verify'
                hits = backend.recall(scope, query, limit=5)
                documents = [asdict(backend.read(scope, hit.uri)) for hit in hits]
                record(dict(stage='verification', query=query,
                            hits=[asdict(h) for h in hits], documents=documents))
                return 'completed' if hits else 'completed_without_hits'
            if task.state in {TaskState.FAILED, TaskState.CANCELLED}:
                return task.state.value
            if attempt + 1 < max_polls:
                time.sleep(interval)
        return 'poll_limit_reached'
    except MemoryBackendError as error:
        record(dict(stage='error', operation=stage, code=error.code.value))
        return error.code.value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session_file', type=Path)
    parser.add_argument('--start', required=True, type=int)
    parser.add_argument('--end', required=True, type=int)
    parser.add_argument('--user', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--account', required=True)
    parser.add_argument('--remote-user', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:1933')
    parser.add_argument('--query', required=True)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    scope = MemoryScope(args.user, args.project)
    metadata, messages = prepare_batch(args.session_file, args.start, args.end)
    # 独占创建避免覆盖尝试记录；换报告路径并不具备去重保证。
    with args.report.open('x', encoding='utf-8') as report:
        def record(item):
            report.write(json.dumps(item, ensure_ascii=False) + '\n')
            report.flush()
            os.fsync(report.fileno())
        record(dict(stage='prepared', mode='submit' if args.submit else 'preview',
                    session_file=str(args.session_file.resolve()), scope=asdict(scope),
                    account=args.account, remote_user=args.remote_user, base_url=args.base_url,
                    **metadata))
        if not args.submit:
            record(dict(stage='preview', messages=[asdict(m) for m in messages]))
            print('preview: inspect the report before submitting')
            return 0
        with OpenVikingBackend(base_url=args.base_url, scope=scope, account_id=args.account,
                               remote_user_id=args.remote_user, timeout=20) as backend:
            outcome = submit_batch(backend, scope, metadata, messages, record, query=args.query)
        record(dict(stage='finished', outcome=outcome))
        print(outcome)
        return 0 if outcome == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
