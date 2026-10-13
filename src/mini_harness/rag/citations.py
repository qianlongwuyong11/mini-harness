"""Per-turn evidence ledger, bounded raw excerpts and deterministic citation checks."""
from dataclasses import replace
import json
import re
import secrets
from ..meter import estimate_tokens

CITATION = re.compile(r'\[cite:([^\]\s]+)\]')
FALLBACK = '未找到足够且可验证的论文依据，暂时无法回答。请缩小问题范围或补读相关章节。'


class EvidenceLedger:
    def __init__(self, runtime, budget=6000):
        if budget < 256:
            raise ValueError('Evidence budget must be at least 256 tokens')
        self.runtime, self.budget = runtime, budget
        self.begin()

    def begin(self):
        self.items = {}
        self.used = False
        self._seen = set()
        self.last_decisions = []

    def payload(self):
        return json.dumps([e.to_dict() for e in self.items.values()],ensure_ascii=False)

    def add(self, evidence):
        self.used = True
        accepted = []
        self.last_decisions = []
        for original in evidence:
            decision=dict(chunk_id=original.chunk_id,paper_id=original.paper_id,version=original.version,
                source_kind=original.source_kind.value,section_path=list(original.section_path),
                page_spans=[dict(start=p.start,end=p.end) for p in original.page_spans])
            self.last_decisions.append(decision)
            try:
                self.runtime.validate(original)
            except Exception:
                decision['reason']='invalid_evidence'
                raise
            key = (original.paper_id,original.version,original.source_kind,original.excerpt,
                   original.section_path,original.page_spans)
            if key in self._seen:
                decision['reason']='duplicate'
                continue
            # Avoid wholly repeated adjacent excerpts without discarding independent passages.
            if any(original.paper_id == e.paper_id and original.version == e.version
                   and original.source_kind == e.source_kind and original.section_path == e.section_path
                   and original.page_spans == e.page_spans and original.excerpt in e.excerpt
                   for e in self.items.values()):
                decision['reason']='covered'
                continue
            identity = secrets.token_hex(16)
            candidate = replace(original,evidence_id=identity)
            # Keep whole excerpts: never silently cut a table/formula to meet the budget.
            packed = json.dumps([e.to_dict() for e in (*self.items.values(),candidate)],ensure_ascii=False)
            if estimate_tokens(packed) > self.budget:
                decision['reason']='budget'
                continue
            decision.update(reason='accepted',evidence_id=identity)
            self.items[identity] = candidate
            self._seen.add(key)
            accepted.append(candidate)
        return tuple(accepted)

    def get_evidence(self, identity):
        evidence = self.items.get(identity)
        if evidence:
            self.runtime.validate(evidence)
        return evidence

    def validate_answer(self, text):
        if self.used and re.search(r'(?:全文|论文(?:中)?|本文档).{0,6}(?:没有|未).{0,4}(?:记录|提及|披露|给出)',text):
            raise ValueError('Retrieved excerpts cannot establish global absence')
        ids = CITATION.findall(text)
        if '[cite:' in CITATION.sub('',text):
            raise ValueError('Malformed citation')
        for identity in ids:
            if self.get_evidence(identity) is None:
                raise ValueError('Unknown or previous-turn citation')
        if self.used and not ids and text != FALLBACK:
            raise ValueError('Paper answer has no current-turn citation')
        return tuple(dict.fromkeys(ids))
