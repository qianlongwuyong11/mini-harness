"""Authenticated local Elasticsearch client; no paper indexing in this step."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import ssl
from urllib.parse import urlsplit
import httpx

@dataclass(frozen=True)
class ESConfig:
    url: str
    ca_file: str
    username: str
    password: str = field(repr=False)

    def __post_init__(self):
        u = urlsplit(self.url)
        if (u.scheme != 'https' or u.hostname not in ('127.0.0.1', 'localhost', '::1')
                or u.username or u.password or u.path not in ('', '/') or u.query or u.fragment):
            raise ValueError('Expected HTTPS loopback URL without credentials or path')
        if not self.username or not self.password:
            raise ValueError('Credentials required')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if os.environ.get('MINI_HARNESS_ES_PASSWORD'):
            data['password'] = os.environ['MINI_HARNESS_ES_PASSWORD']
        return cls(**data)

class ESService:
    def __init__(self, config):
        self.client = httpx.Client(base_url=config.url,
            verify=ssl.create_default_context(cafile=config.ca_file),
            auth=(config.username, config.password), timeout=10, trust_env=False)

    def health(self):
        response = self.client.get('/_cluster/health', params={'wait_for_status':'yellow','timeout':'5s'})
        response.raise_for_status()
        data = response.json()
        if data.get('cluster_name') != 'mini-harness-rag':
            raise RuntimeError('Unexpected Elasticsearch cluster')
        if data.get('timed_out') or data.get('status') not in ('green', 'yellow'):
            raise RuntimeError('Elasticsearch not ready')
        return data

    def close(self):
        self.client.close()
