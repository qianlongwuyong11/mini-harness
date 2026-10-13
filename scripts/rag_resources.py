"""Measure a fresh isolated import; local files and local ES only."""
import argparse,json,sys,threading,time,uuid
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mini_harness.rag.runtime import PaperRuntime,RagConfig
from pypdf import PdfReader


def main():
    cli=argparse.ArgumentParser();cli.add_argument('pdf',type=Path);cli.add_argument('--output',type=Path,default=ROOT/'docs/rag-resource-report.json');args=cli.parse_args()
    run=uuid.uuid4().hex[:10];root=ROOT/'.mini-harness/rag/acceptance'/run
    root.mkdir(parents=True);process=psutil.Process();samples=[];stop=threading.Event();started=time.perf_counter();phase=['startup'];stage_seconds={}
    def monitor():
        previous=time.perf_counter()
        while not stop.is_set():
            now=time.perf_counter();stage_seconds[phase[0]]=stage_seconds.get(phase[0],0)+now-previous;previous=now
            procs=[process]+process.children(recursive=True);rss=0
            for p in procs:
                try:rss+=p.memory_info().rss
                except psutil.Error:pass
            samples.append(dict(t=now-started,rss=rss,available=psutil.virtual_memory().available))
            stop.wait(.2)
    thread=threading.Thread(target=monitor);thread.start();runtime=None;swap_before=psutil.swap_memory()._asdict()
    try:
        runtime=PaperRuntime(RagConfig(catalog_path=str(root/'catalog.sqlite3'),root=str(root/'ingestion'),index_name='mini-harness-acceptance-'+run))
        t=time.perf_counter();runtime._load();load=time.perf_counter()-t
        job=runtime.enqueue(args.pdf)
        original=runtime.catalog.advance_job
        def advance(expected,stage):
            phase[0]=stage.value
            return original(expected,stage)
        runtime.catalog.advance_job=advance
        phase[0]='parsing';t=time.perf_counter();result=runtime._worker.run_once(job_id=job.job_id);duration=time.perf_counter()-t
        if result is None or result.status.value!='ready':raise RuntimeError('Import failed; inspect isolated catalog')
        v=runtime.catalog.get_active_version(job.paper_id);pages=len(PdfReader(args.pdf).pages)
        response=runtime._es.client.get('/'+runtime._worker.index.index_name+'/_stats/store,docs,translog');response.raise_for_status()
        report=dict(status=result.status.value,pages=pages,chunks=v.chunk_count,process_model_load_seconds=load,import_seconds=duration,amortized_seconds_per_page=duration/pages,stage_seconds=stage_seconds,index_stats=response.json()['_all'],isolated_run=run,cpu_only=True,notes=['Model files already cached; fresh catalog and artifact directory','RSS includes worker and child processes, excludes ES; system available memory includes all apps','Per-page figure is amortized whole-document import, not individual page latency','Swap counters on Windows may be zero or unavailable; no no-paging guarantee'])
    finally:
        stop.set();thread.join()
        if runtime:runtime.close()
    report.update(peak_process_tree_rss_bytes=max(s['rss'] for s in samples),minimum_system_available_bytes=min(s['available'] for s in samples),physical_memory_bytes=psutil.virtual_memory().total,swap_before=swap_before,swap_after=psutil.swap_memory()._asdict(),sample_count=len(samples),artifact_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
    report['resource_gates']=dict(import_ready=report['status']=='ready',available_memory_ge_2gib=report['minimum_system_available_bytes']>=2*1024**3)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
