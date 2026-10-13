"""Regenerate versioned JSON schemas and metric catalog from Python contracts."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mini_harness.evaluation.models import EvaluationDataset,MetricResult,RunRecord,TraceIdentity,ArtifactReference
from mini_harness.evaluation.registry import metric_catalog

def main():
    target=ROOT/'docs/evaluation/v1';target.mkdir(parents=True,exist_ok=True)
    for name,model in [('dataset',EvaluationDataset),('metric-result',MetricResult),('run',RunRecord),('trace-identity',TraceIdentity),('artifact',ArtifactReference)]:
        schema=model.model_json_schema()
        schema['$schema']='https://json-schema.org/draft/2020-12/schema'
        (target/(name+'.schema.json')).write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (target/'metrics.json').write_text(json.dumps(metric_catalog(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Exported 5 schemas and {len(metric_catalog())} metric definitions')
if __name__=='__main__':main()
