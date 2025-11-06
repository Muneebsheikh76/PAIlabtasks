import json
import requests
import types

from app import app


class DummyResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('status %s' % self.status_code)

    def json(self):
        return self._data


def test_index_page():
    client = app.test_client()
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'Random Joke App' in resp.data


def test_api_joke_monkeypatch(monkeypatch):
    fake = DummyResponse({'id': 1, 'type': 'general', 'setup': 'Why?', 'punchline': 'Because.'})

    def fake_get(*args, **kwargs):
        return fake

    monkeypatch.setattr('requests.get', fake_get)

    client = app.test_client()
    resp = client.get('/api/joke')
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert 'joke' in data
    assert 'Why?' in data['joke']
