"""Windows ZIP setup and owned foreground launcher. Run from repository root."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.mini-harness/elasticsearch'
VERSION = '9.5.3'
HOME = BASE / f'elasticsearch-{VERSION}'
CONFIG = BASE / 'connection.json'
sys.path.insert(0, str(ROOT / 'src'))
from mini_harness.rag.es_service import ESConfig, ESService


def command(name, *args):
    result = subprocess.run([str(HOME / 'bin' / (name + '.bat')), *args],
        cwd=HOME, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{name} failed (output withheld to protect credentials)')


def install():
    if CONFIG.exists():
        raise RuntimeError('Already configured; refusing to overwrite credentials/data')
    BASE.mkdir(parents=True, exist_ok=True)
    archive = BASE / 'elasticsearch.zip'
    url = f'https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-{VERSION}-windows-x86_64.zip'
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    expected = urllib.request.urlopen(url + '.sha512', timeout=60).read().decode().split()[0]
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha512').hexdigest()
    if actual != expected:
        raise RuntimeError('Official SHA512 mismatch; delete archive and retry')
    print('Official checksum verified; extracting', flush=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(BASE)
    certs = HOME / 'config/certs'
    certs.mkdir(exist_ok=True)
    command('elasticsearch-certutil', 'ca', '--silent', '--pem', '--out', str(BASE / 'ca.zip'))
    with zipfile.ZipFile(BASE / 'ca.zip') as zipped:
        zipped.extractall(certs)
    command('elasticsearch-certutil', 'cert', '--silent', '--pem', '--name', 'localhost',
        '--dns', 'localhost', '--ip', '127.0.0.1,::1',
        '--ca-cert', str(certs / 'ca/ca.crt'), '--ca-key', str(certs / 'ca/ca.key'),
        '--out', str(BASE / 'node.zip'))
    with zipfile.ZipFile(BASE / 'node.zip') as zipped:
        zipped.extractall(certs)
    for name in ('data', 'logs'):
        (BASE / name).mkdir(exist_ok=True)
    settings = {
        'cluster.name': 'mini-harness-rag', 'node.name': 'mini-harness-local',
        'network.host': '127.0.0.1', 'http.port': 9200, 'transport.port': 9300,
        'discovery.type': 'single-node', 'path.data': str(BASE / 'data'),
        'path.logs': str(BASE / 'logs'), 'xpack.security.enabled': True,
        'xpack.security.autoconfiguration.enabled': False,
        'xpack.security.http.ssl.enabled': True,
        'xpack.security.http.ssl.key': 'certs/localhost/localhost.key',
        'xpack.security.http.ssl.certificate': 'certs/localhost/localhost.crt',
        'xpack.security.http.ssl.certificate_authorities': ['certs/ca/ca.crt'],
        'xpack.ml.enabled': False,
    }
    (HOME / 'config/elasticsearch.yml').write_text(
        '\n'.join(f'{k}: {json.dumps(v)}' for k,v in settings.items()) + '\n', encoding='utf-8')
    (HOME / 'config/jvm.options.d/mini-harness.options').write_text('-Xms2g\n-Xmx2g\n', encoding='utf-8')
    password = secrets.token_urlsafe(32)
    command('elasticsearch-users', 'useradd', 'mini_harness', '-p', password, '-r', 'superuser')
    CONFIG.write_text(json.dumps(dict(url='https://127.0.0.1:9200',
        ca_file=str(certs / 'ca/ca.crt'), username='mini_harness', password=password), indent=2), encoding='utf-8')
    print('Installed; credentials are in ignored connection.json (local development administrator).')


def serve(smoke=False):
    service = ESService(ESConfig.load(CONFIG))
    process = None
    log = None
    try:
        with socket.socket() as sock:
            occupied = sock.connect_ex(('127.0.0.1',9200)) == 0
        if occupied:
            service.health()
            if smoke:
                raise RuntimeError('Smoke restart test requires its own instance')
            print('Reusing healthy existing ES; no process ownership acquired')
            return
        log = (BASE / 'launcher.log').open('ab')
        def start():
            child = subprocess.Popen([str(HOME / 'bin/elasticsearch.bat')], cwd=HOME,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            return child
        def stop(child):
            if child.poll() is None:
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'], check=True, capture_output=True)
                child.wait(timeout=30)
        def ready(child):
            for _ in range(180):
                if child.poll() is not None:
                    raise RuntimeError('ES exited; inspect launcher.log')
                try:
                    return service.health()
                except Exception:
                    time.sleep(1)
            raise TimeoutError('ES did not become healthy; inspect launcher.log')
        process = start()
        print(json.dumps(ready(process)), flush=True)
        if smoke:
            import httpx
            import ssl
            cfg = ESConfig.load(CONFIG)
            with httpx.Client(verify=ssl.create_default_context(cafile=cfg.ca_file), trust_env=False) as anonymous:
                assert anonymous.get(cfg.url + '/_cluster/health').status_code == 401
            nodes = service.client.get('/_nodes/http,transport')
            nodes.raise_for_status()
            for node in nodes.json()['nodes'].values():
                for kind in ('http', 'transport'):
                    assert all(address.startswith('127.0.0.1:') for address in node[kind]['bound_address'])
            print('PASS: anonymous access rejected; HTTP and transport bound only to loopback', flush=True)
            index = 'mini-harness-smoke-' + secrets.token_hex(6)
            try:
                response = service.client.put('/' + index, json={'settings':{'number_of_shards':1,'number_of_replicas':0}})
                response.raise_for_status()
                service.client.put(f'/{index}/_doc/1', params={'refresh':'wait_for'},json={'probe':'persistent'}).raise_for_status()
                stop(process)
                process = start()
                ready(process)
                response = service.client.get(f'/{index}/_doc/1')
                response.raise_for_status()
                assert response.json()['_source']['probe'] == 'persistent'
                print('PASS: TLS, authentication, health, write, forced-stop restart and persistence', flush=True)
            finally:
                service.client.delete('/' + index).raise_for_status()
        else:
            print('Ctrl+C stops only this launcher-owned instance.', flush=True)
            while process.poll() is None:
                time.sleep(1)
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],check=True,capture_output=True)
            process.wait(timeout=30)
        if log:
            log.close()
        service.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['install','serve','health','smoke'])
    args = parser.parse_args()
    try:
        if args.action == 'install': install()
        elif args.action == 'health':
            service = ESService(ESConfig.load(CONFIG))
            try: print(json.dumps(service.health()))
            finally: service.close()
        else: serve(args.action == 'smoke')
    except KeyboardInterrupt:
        pass
