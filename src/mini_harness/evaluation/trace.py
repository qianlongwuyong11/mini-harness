"""Local content-addressed trace artifacts. No SDK or Pydantic dependency."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

_wire_sink = ContextVar("model_wire_sink", default=None)
_SECRET = re.compile(r"api[_-]?key|authorization|password|secret|access[_-]?token", re.I)

def redact(value, secrets=()):
    if isinstance(value, dict):
        return {k: ('[omitted: reasoning]' if k == 'reasoning_content' else
                    '[redacted]' if _SECRET.search(k) else redact(v,secrets)) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [redact(v,secrets) for v in value]
    if isinstance(value,str):
        # Structured JSON tool arguments are often embedded in a string.
        if value.lstrip().startswith(('{','[')):
            try: return json.dumps(redact(json.loads(value),secrets),ensure_ascii=False)
            except (ValueError,TypeError): pass
        for secret in secrets:
            if secret and len(secret)>=4: value=value.replace(secret,'[redacted]')
        value=re.sub(r'(?i)Bearer\s+[A-Za-z0-9_.\-]+','Bearer [redacted]',value)
        value=re.sub(r'(?i)((?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*)[^\s,;]+',r'\1[redacted]',value)
        return value
    return value

class TraceStore:
    def __init__(self, root=None, secrets=()):
        self.root=Path(root or '.mini-harness/traces').resolve()
        self.secrets=tuple(secrets)+tuple(v for k,v in os.environ.items() if _SECRET.search(k))
    def put(self, value):
        raw=json.dumps(redact(value,self.secrets),ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8')
        digest=sha256(raw).hexdigest();relative=f'{digest[:2]}/{digest}.json'
        dest=self.root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(dir=dest.parent,suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(raw);f.flush();os.fsync(f.fileno())
            os.replace(name,dest)
        finally:
            if os.path.exists(name):os.unlink(name)
        return dict(relative_path=relative,sha256=digest,size_bytes=len(raw),media_type='application/json',redacted=True)

def visible_message(message):
    data=asdict(message)
    data.pop('reasoning_content',None)
    return data

@contextmanager
def wire_capture(callback):
    token=_wire_sink.set(callback)
    try: yield
    finally:_wire_sink.reset(token)

def capture_wire_payload(payload):
    sink=_wire_sink.get()
    if sink is not None:sink(payload)
