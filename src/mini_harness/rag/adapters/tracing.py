"""LlamaIndex callback handler retaining bounded numeric spans only."""
from time import perf_counter
from uuid import uuid4
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import EventPayload

class LocalTraceHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__([],[])
        self.trace_id=uuid4().hex
        self.starts={};self.spans=[]
    def start_trace(self,trace_id=None):pass
    def end_trace(self,trace_id=None,trace_map=None):pass
    def on_event_start(self,event_type,payload=None,event_id='',parent_id='',**kwargs):
        self.starts[event_id]=(perf_counter(),event_type.value,parent_id,(payload or {}).get('stage',event_type.value))
        return event_id
    def on_event_end(self,event_type,payload=None,event_id='',**kwargs):
        row=self.starts.pop(event_id,None)
        if row is None:return
        start,kind,parent,name=row
        if len(self.spans)<256:
            self.spans.append(dict(id=event_id,parent_id=parent,name=name,event_type=kind,
                elapsed_ms=(perf_counter()-start)*1000,status='error' if (payload or {}).get(EventPayload.EXCEPTION) else 'ok'))
