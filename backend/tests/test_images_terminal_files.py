import base64
import io

import pytest
from fastapi import HTTPException
from open_webui.routers import images as images_module
from open_webui.routers.images import EditImageForm
from open_webui.utils import terminals as terminals_module
from PIL import Image

from tests.test_images_openrouter import _install_fake_io, _openrouter_config


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new('RGB', (1, 1), 'red').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _install_fake_terminal_fetch(monkeypatch, expected_path: str, b64: str, calls: list):
    """Install a fake open_webui.utils.terminals.fetch_terminal_file_bytes."""
    from urllib.parse import quote

    async def fake_fetch(request, user, metadata, file_path: str):
        calls.append(file_path)
        if file_path != f'/files/view?path={quote(expected_path, safe="")}':
            raise HTTPException(status_code=404, detail='File not found')
        return base64.b64decode(b64), 'image/png'

    monkeypatch.setattr(terminals_module, 'fetch_terminal_file_bytes', fake_fetch)


@pytest.fixture
def fake_io(monkeypatch):
    cfg = _openrouter_config()
    session = _SessionStub()
    return _install_fake_io(monkeypatch, cfg, session)


class _Resp:
    status = 200
    ok = True

    async def json(self, content_type=None):
        return {'data': [{'b64_json': _png_b64()}]}


class _PostCtx:
    async def __aenter__(self):
        return _Resp()

    async def __aexit__(self, *exc):
        return False


class _SessionStub:
    def __init__(self):
        pass

    def post(self, **kwargs):
        return _PostCtx()


METADATA = {
    'terminal_id': 'term-1',
    'chat_id': 'chat-1',
    'session_id': 'session-1',
}


@pytest.mark.asyncio
async def test_image_edits_resolves_terminal_filesystem_path(fake_io, monkeypatch):
    """A filesystem attachment path (absolute) resolves via the terminal server."""
    b64 = _png_b64()
    calls = []
    _install_fake_terminal_fetch(monkeypatch, '/home/user/image (1).jpg', b64, calls)

    result = await images_module.image_edits(
        request=None,
        form_data=EditImageForm(image='/home/user/image (1).jpg', prompt='make it blue'),
        metadata=dict(METADATA),
        user=None,
    )
    assert calls == ['/files/view?path=%2Fhome%2Fuser%2Fimage%20%281%29.jpg']
    assert len(result) == 1


@pytest.mark.asyncio
async def test_image_edits_terminal_path_lookup_failure_raises_actionable_error(fake_io, monkeypatch):
    """A terminal path the terminal cannot serve yields an actionable 404."""

    async def fake_fetch(path, request=None, metadata=None, user=None):
        raise HTTPException(status_code=404, detail='File not found')

    monkeypatch.setattr(terminals_module, 'fetch_terminal_file_bytes', fake_fetch)

    with pytest.raises(HTTPException) as exc_info:
        await images_module.image_edits(
            request=None,
            form_data=EditImageForm(image='/home/user/missing.jpg', prompt='make it blue'),
            metadata=dict(METADATA),
            user=None,
        )
    assert exc_info.value.status_code == 404
    detail = str(exc_info.value.detail)
    assert '/home/user/missing.jpg' in detail
    assert 'terminal' in detail.lower()


@pytest.mark.asyncio
async def test_image_edits_terminal_path_without_terminal_id_raises_actionable_error(fake_io):
    """A filesystem-like path with no terminal_id in metadata cannot be fetched."""
    with pytest.raises(HTTPException) as exc_info:
        await images_module.image_edits(
            request=None,
            form_data=EditImageForm(image='/home/user/image (1).jpg', prompt='make it blue'),
            metadata={'chat_id': 'chat-1'},
            user=None,
        )
    assert exc_info.value.status_code == 404
    assert 'terminal' in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_image_edits_bare_filename_not_treated_as_terminal_path(fake_io):
    """Bare filenames keep the legacy behavior: treated as an Open WebUI file id."""
    with pytest.raises(HTTPException) as exc_info:
        await images_module.image_edits(
            request=None,
            form_data=EditImageForm(image='image (1).jpg', prompt='make it blue'),
            metadata=dict(METADATA),
            user=None,
        )
    assert exc_info.value.status_code == 404
    detail = str(exc_info.value.detail)
    assert 'image (1).jpg' in detail
    assert '/api/v1/files/' in detail
