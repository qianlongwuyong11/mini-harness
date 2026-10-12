"""Local UI boundary: bounded PDF intake, durable status and ID-only source access."""
from pathlib import Path
import json
import os
import tempfile
import threading
import time
from urllib.parse import unquote

from pypdf import PdfReader
from .models import IngestionStatus

MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_PAGES = 300
_PDF_RENDER_LOCK = threading.Lock()


class PaperUI:
    def __init__(self, runtime, *, monitor=True):
        self.runtime = runtime
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._status = dict(es='checking',embedding='not_loaded',worker='idle',degraded=False,error=None)
        self._thread = None
        if monitor:
            self._thread = threading.Thread(target=self._monitor,name='paper-ui-monitor',daemon=False)
            self._thread.start()

    def _monitor(self):
        from .es_service import ESService,ESConfig
        while not self._stop.is_set():
            status = dict(es='unavailable',embedding='ready' if self.runtime._embedding else 'not_loaded',
                          worker='idle',degraded=False,error=None)
            client = None
            try:
                client = ESService(ESConfig.load(Path(self.runtime.config.es_config)))
                client.health()
                status['es']='ready'
            except Exception:
                status['error']='ES 尚未就绪，请检查本地 ES 服务与连接配置。'
            finally:
                if client: client.close()
            try:
                jobs=self.runtime.catalog.list_unfinished_jobs()
                if jobs:
                    status['worker']='processing' if any(j.status is IngestionStatus.RUNNING for j in jobs) else 'queued'
                with self._lock: self._status=status
                if (jobs and status['es']=='ready' and not self._stop.is_set()
                        and not (self.runtime._thread and self.runtime._thread.is_alive())):
                    self.runtime.start_worker()
            except Exception:
                with self._lock:
                    self._status.update(worker='error',error='导入服务初始化失败，请检查本地模型文件和可选依赖。')
            self._stop.wait(3)

    def snapshot(self):
        c=self.runtime.catalog
        with c._connection() as db:
            papers=db.execute('SELECT paper_id FROM papers WHERE deleted=0 ORDER BY paper_id').fetchall()
            jobs=[json.loads(raw) for (raw,) in db.execute('SELECT data FROM jobs')]
        result=[]
        for (pid,) in papers:
            paper=c.get_paper(pid)
            if paper is None: continue
            version=c.get_active_version(pid)
            relevant=sorted((j for j in jobs if j['paper_id']==pid),key=lambda j:j['updated_at'],reverse=True)
            job=relevant[0] if relevant else None
            result.append(dict(paper_id=pid,title=paper.title or pid,version=version.version if version else None,
                ready=version is not None,job=None if job is None else {key:job[key] for key in (
                'job_id','status','stage','progress_current','progress_total','attempts','error_message')}))
        with self._lock: status=dict(self._status)
        status['embedding']='ready' if self.runtime._embedding else 'not_loaded'
        return dict(papers=result,status=status,max_upload_bytes=MAX_PDF_BYTES)

    def upload(self, stream, size, name, content_type):
        if not 0 < size <= MAX_PDF_BYTES:
            raise ValueError('PDF 必须大于 0 且不超过 30 MB')
        name=unquote(name)
        if not name or len(name)>200 or any(c in name for c in '/\\:\x00\r\n') or not name.lower().endswith('.pdf'):
            raise ValueError('仅接受合法 PDF 文件名')
        if content_type.split(';')[0].strip() not in ('application/pdf','application/octet-stream'):
            raise ValueError('仅接受 PDF 文件')
        directory=Path(self.runtime.config.root).resolve()/'uploads'
        directory.mkdir(parents=True,exist_ok=True)
        fd,path=tempfile.mkstemp(suffix='.pdf',dir=directory)
        try:
            with os.fdopen(fd,'wb') as output:
                remaining=size
                while remaining:
                    chunk=stream.read(min(1024*1024,remaining))
                    if not chunk: raise ValueError('上传中断，文件不完整')
                    output.write(chunk)
                    remaining-=len(chunk)
                output.flush()
                os.fsync(output.fileno())
            with open(path,'rb') as source:
                if source.read(5)!=b'%PDF-': raise ValueError('文件内容不是 PDF')
                source.seek(0)
                try:
                    reader=PdfReader(source)
                    if reader.is_encrypted: raise ValueError('暂不支持加密 PDF')
                    if not 1<=len(reader.pages)<=MAX_PAGES: raise ValueError('PDF 页数须为 1–300')
                except ValueError: raise
                except Exception as exc: raise ValueError('PDF 损坏或无法读取') from exc
            job=self.runtime.enqueue(path,title=name[:-4])
            return dict(job_id=job.job_id,paper_id=job.paper_id,status=job.status.value)
        finally:
            Path(path).unlink(missing_ok=True)

    def action(self, action, identifier):
        if not isinstance(identifier,str) or not identifier: raise ValueError('缺少任务或论文 ID')
        catalog=self.runtime.catalog
        if action=='retry':
            job=catalog.retry_job(identifier)
            return dict(job_id=job.job_id,status=job.status.value)
        if action=='cancel':
            job=catalog.cancel_job(identifier)
            return dict(job_id=job.job_id,status=job.status.value)
        if action=='delete':
            if catalog.get_paper(identifier) is None: raise ValueError('论文不存在')
            catalog.soft_delete_paper(identifier)
            return dict(deleted=True)
        raise ValueError('未知论文操作')

    def source(self, paper_id, version, page):
        self.runtime._active(paper_id,version)
        paper=self.runtime.catalog.get_paper(paper_id)
        path=Path(paper.source_uri)
        if not path.is_file() or path.suffix.lower()!='.pdf': raise ValueError('原始 PDF 不可用')
        if path.stat().st_size>MAX_PDF_BYTES: raise ValueError('原文件超过预览大小限制')
        if not 1 <= page <= len(PdfReader(path).pages): raise ValueError('页码越界')
        return path.read_bytes()

    def page_image(self, paper_id, version, page):
        import io
        import pypdfium2 as pdfium
        data=self.source(paper_id,version,page)
        with _PDF_RENDER_LOCK, pdfium.PdfDocument(data) as document:
            pdf_page=document[page-1]
            try:
                scale=min(1.5,1600/max(pdf_page.get_size()))
                bitmap=pdf_page.render(scale=scale)
                try:
                    output=io.BytesIO()
                    bitmap.to_pil().save(output,format='PNG')
                    return output.getvalue()
                finally: bitmap.close()
            finally: pdf_page.close()

    def close(self):
        self._stop.set()
        if self._thread: self._thread.join()
        self.runtime.close()
