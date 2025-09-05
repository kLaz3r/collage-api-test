import asyncio
import json
import types
import uuid
from typing import Tuple, List

import pytest
from fastapi.testclient import TestClient


class FakeAsyncRedis:
    def __init__(self):
        self._store = {}

    @classmethod
    def from_url(cls, *args, **kwargs):  # pragma: no cover - factory compatibility
        return cls()

    async def ping(self):
        return True

    async def close(self):  # pragma: no cover - compatibility
        return None

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self._store[key] = value
        return True

    async def delete(self, key: str):
        self._store.pop(key, None)
        return 1

    async def exists(self, key: str):
        return 1 if key in self._store else 0

    async def scan(self, cursor: int = 0, match: str = "*", count: int = 100):
        # Very simple scan implementation
        keys = [k for k in self._store.keys() if k.startswith(match.replace("*", ""))]
        return 0, keys

    async def mget(self, keys: List[str]):
        return [self._store.get(k) for k in keys]


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    import server

    # Substitute AsyncRedis class used by startup with our fake
    monkeypatch.setattr(server, "AsyncRedis", FakeAsyncRedis)

    # Ensure global redis_client is our fake instance
    fake = FakeAsyncRedis()

    async def _get_redis_override():
        return fake

    monkeypatch.setattr(server, "_get_redis", _get_redis_override)
    monkeypatch.setattr(server, "redis_client", fake)

    # Disable sending real Celery tasks
    class DummySender:
        def send_task(self, *args, **kwargs):
            return None

    monkeypatch.setattr(server, "celery_app", DummySender())

    # Build client after monkeypatches; this triggers startup using our fakes
    test_client = TestClient(server.app)
    return test_client


