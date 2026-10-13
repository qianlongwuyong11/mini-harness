"""Offline retrieval metrics through LlamaIndex RetrieverEvaluator.

Replay the exact captured ranked IDs so evaluation never repeats ES or embedding calls.
The installed metrics reject empty retrieval; defined empty-result scores are zero.
"""
import statistics
from llama_index.core.evaluation.retrieval.evaluator import RetrieverEvaluator
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore,TextNode

class RankedResultsRetriever(BaseRetriever):
    def __init__(self,ids):
        super().__init__(callback_manager=CallbackManager([]))
        self.ids=ids
    def _retrieve(self,query_bundle):
        return [NodeWithScore(node=TextNode(id_=i,text=''),score=1/(n+1)) for n,i in enumerate(self.ids)]
    async def _aretrieve(self,query_bundle):
        return self._retrieve(query_bundle)

def evaluate_rankings(rows):
    rows=list(rows)
    if not rows:raise ValueError('Empty evaluation split')
    result={}
    for k in (1,3,5):
        scores=[]
        for ranked,gold in rows:
            expected=list(dict.fromkeys(gold))
            if not expected:raise ValueError('Missing gold evidence')
            ids=list(dict.fromkeys(ranked))[:k]
            if not ids:
                scores.append(dict(recall=0.,hit_rate=0.,mrr=0.))
                continue
            evaluator=RetrieverEvaluator.from_metric_names(['recall','hit_rate','mrr'],retriever=RankedResultsRetriever(ids))
            scores.append(evaluator.evaluate(query='captured ranking',expected_ids=expected).metric_vals_dict)
        for name,key in (('recall','recall'),('hit','hit_rate'),('mrr','mrr')):
            result[f'{name}@{k}']=statistics.mean(s[key] for s in scores)
    return result
