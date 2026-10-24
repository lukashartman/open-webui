import base64
import io
from types import SimpleNamespace

import aiohttp
import pytest
from fastapi import HTTPException
from open_webui.routers import images as images_module
from open_webui.routers.images import (
    CreateImageForm,
    EditImageForm,
    build_openrouter_edit_payload,
    build_openrouter_generation_payload,
    build_openrouter_headers,
    decode_openrouter_image,
    openrouter_request_size,
)
from PIL import Image


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new('RGB', (1, 1), 'red').save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def test_build_openrouter_headers():
    headers = build_openrouter_headers('sk-test')
    assert headers['Authorization'] == 'Bearer sk-test'
    assert headers['Content-Type'] == 'application/json'
    assert headers['X-Title'] == 'Open WebUI Image Generation'


def test_openrouter_request_size():
    assert openrouter_request_size('1024x1024') == '1024x1024'
    assert openrouter_request_size('auto') is None
    assert openrouter_request_size('2K') is None
    assert openrouter_request_size('') is None
    assert openrouter_request_size(None) is None


def test_build_openrouter_generation_payload():
    payload = build_openrouter_generation_payload('openai/gpt-image-1', 'a cat', 1, '1024x1024', {'quality': 'high'})
    assert payload == {
        'model': 'openai/gpt-image-1',
        'prompt': 'a cat',
        'n': 1,
        'size': '1024x1024',
        'quality': 'high',
    }


def test_generation_payload_omits_invalid_size():
    payload = build_openrouter_generation_payload('m', 'p', 1, 'auto', None)
    assert 'size' not in payload


def test_generation_payload_params_string():
    payload = build_openrouter_generation_payload('m', 'p', 1, None, '{"aspect_ratio": "16:9"}')
    assert payload['aspect_ratio'] == '16:9'


def test_generation_payload_params_override_size():
    payload = build_openrouter_generation_payload('m', 'p', 1, None, {'size': '2K'})
    assert payload['size'] == '2K'


def test_edit_payload_single_image():
    payload = build_openrouter_edit_payload('m', 'p', None, None, None, 'data:image/png;base64,AAA')
    assert payload['input_references'] == [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}}]
    assert 'n' not in payload


def test_edit_payload_image_list():
    payload = build_openrouter_edit_payload(
        'm', 'p', 2, None, {}, ['data:image/png;base64,AAA', 'data:image/png;base64,BBB']
    )
    assert payload['n'] == 2
    assert payload['input_references'] == [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,BBB'}},
    ]


@pytest.mark.asyncio
async def test_decode_openrouter_image_uses_media_type():
    b64 = _png_b64()
    data, content_type = await decode_openrouter_image({'b64_json': b64, 'media_type': 'image/svg+xml'})
    assert content_type == 'image/svg+xml'
    assert data == base64.b64decode(b64)


@pytest.mark.asyncio
async def test_decode_openrouter_image_sniffs_when_media_type_missing():
    data, content_type = await decode_openrouter_image({'b64_json': _png_b64()})
    assert content_type == 'image/png'
    assert data[:8] == b'\x89PNG\r\n\x1a\n'


def test_openrouter_config_keys_are_registered():
    from open_webui.config import DEFAULT_CONFIG
    from open_webui.models.config import DICT_CONFIG_KEY_ALIASES
    from open_webui.routers.images import IMAGE_CONFIG_KEYS

    expected_storage_keys = {
        'IMAGES_OPENROUTER_API_BASE_URL': 'image_generation.openrouter.api_base_url',
        'IMAGES_OPENROUTER_API_KEY': 'image_generation.openrouter.api_key',
        'IMAGES_OPENROUTER_API_PARAMS': 'image_generation.openrouter.params',
        'IMAGES_EDIT_OPENROUTER_API_BASE_URL': 'images.edit.openrouter.api_base_url',
        'IMAGES_EDIT_OPENROUTER_API_KEY': 'images.edit.openrouter.api_key',
        'IMAGES_EDIT_OPENROUTER_API_PARAMS': 'images.edit.openrouter.params',
    }
    for field, storage_key in expected_storage_keys.items():
        assert IMAGE_CONFIG_KEYS[field] == storage_key
        assert storage_key in DEFAULT_CONFIG
    assert DICT_CONFIG_KEY_ALIASES['image_generation.openrouter.params'] == ('IMAGES_OPENROUTER_API_PARAMS',)
    assert DICT_CONFIG_KEY_ALIASES['images.edit.openrouter.params'] == ('IMAGES_EDIT_OPENROUTER_API_PARAMS',)


def _openrouter_config(**overrides):
    base = {
        'IMAGE_GENERATION_ENGINE': 'openrouter',
        'IMAGE_GENERATION_MODEL': 'openai/gpt-image-1',
        'IMAGE_SIZE': '1024x1024',
        'IMAGES_OPENROUTER_API_BASE_URL': 'https://openrouter.ai/api/v1',
        'IMAGES_OPENROUTER_API_KEY': 'sk-test',
        'IMAGES_OPENROUTER_API_PARAMS': {},
        'IMAGE_EDIT_ENGINE': 'openrouter',
        'IMAGE_EDIT_MODEL': 'openai/gpt-image-1',
        'IMAGE_EDIT_SIZE': '',
        'IMAGES_EDIT_OPENROUTER_API_BASE_URL': 'https://openrouter.ai/api/v1',
        'IMAGES_EDIT_OPENROUTER_API_KEY': 'sk-test',
        'IMAGES_EDIT_OPENROUTER_API_PARAMS': {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _PostContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _Response:
    def __init__(self, payload, status=200, text=None):
        self._payload = payload
        self.status = status
        self.ok = status < 400
        self._text = text if text is not None else str(payload)

    def raise_for_status(self):
        if not self.ok:
            raise aiohttp.ClientResponseError(request_info=None, history=(), status=self.status, message='Not Found')

    async def json(self, content_type=None):
        return self._payload

    async def text(self):
        return self._text


class _Session:
    def __init__(self, payload, status=200, text=None):
        self._payload = payload
        self._status = status
        self._text = text
        self.posts = []

    def post(self, url, json=None, headers=None, ssl=None):
        self.posts.append({'url': url, 'json': json, 'headers': headers})
        return _PostContext(_Response(self._payload, status=self._status, text=self._text))


@pytest.fixture
def fake_io(monkeypatch):
    cfg = _openrouter_config()
    session = _Session({'data': [{'b64_json': _png_b64(), 'media_type': 'image/png'}]})
    return _install_fake_io(monkeypatch, cfg, session)


def _install_fake_io(monkeypatch, cfg, session):
    async def fake_get_image_config():
        return cfg

    async def fake_get_session():
        return session

    async def fake_upload_image(request, image_data, content_type, metadata, user, db=None):
        return None, {'id': 'file-1', 'url': '/api/v1/files/file-1/content', 'name': 'x', 'content_type': content_type}

    monkeypatch.setattr(images_module, 'get_image_config', fake_get_image_config)
    monkeypatch.setattr(images_module, 'get_session', fake_get_session)
    monkeypatch.setattr(images_module, 'upload_image', fake_upload_image)
    return SimpleNamespace(config=cfg, session=session)


@pytest.fixture
def fake_io_error(monkeypatch):
    cfg = _openrouter_config()
    session = _Session(
        {'data': []},
        status=404,
        text='{"error":{"message":"Model not found"}}',
    )
    return _install_fake_io(monkeypatch, cfg, session)


@pytest.mark.asyncio
async def test_get_image_model_openrouter_returns_configured(fake_io):
    assert await images_module.get_image_model(request=None) == 'openai/gpt-image-1'


@pytest.mark.asyncio
async def test_get_image_model_openrouter_unset_raises(fake_io, monkeypatch):
    fake_io.config.IMAGE_GENERATION_MODEL = ''
    with pytest.raises(HTTPException) as exc_info:
        await images_module.get_image_model(request=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_image_generations_openrouter_posts_to_images_endpoint(fake_io):
    result = await images_module.image_generations(request=None, form_data=CreateImageForm(prompt='a cat'), user=None)
    assert len(fake_io.session.posts) == 1
    post = fake_io.session.posts[0]
    assert post['url'] == 'https://openrouter.ai/api/v1/images'
    assert post['headers']['Authorization'] == 'Bearer sk-test'
    assert post['headers']['X-Title'] == 'Open WebUI Image Generation'
    assert post['json'] == {'model': 'openai/gpt-image-1', 'prompt': 'a cat', 'n': 1, 'size': '1024x1024'}
    assert result == [{'id': 'file-1', 'url': '/api/v1/files/file-1/content', 'name': 'x', 'content_type': 'image/png'}]


@pytest.mark.asyncio
async def test_image_generations_openrouter_requires_model(fake_io):
    fake_io.config.IMAGE_GENERATION_MODEL = ''
    with pytest.raises(HTTPException) as exc_info:
        await images_module.image_generations(request=None, form_data=CreateImageForm(prompt='a cat'), user=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_image_edits_openrouter_builds_input_references(fake_io):
    result = await images_module.image_edits(
        request=None,
        form_data=EditImageForm(image='data:image/png;base64,AAA', prompt='make it blue'),
        user=None,
    )
    assert len(result) == 1
    post = fake_io.session.posts[0]
    assert post['url'] == 'https://openrouter.ai/api/v1/images'
    assert post['json']['model'] == 'openai/gpt-image-1'
    assert post['json']['prompt'] == 'make it blue'
    assert post['json']['input_references'] == [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}}
    ]


@pytest.mark.asyncio
async def test_image_edits_openrouter_requires_model(fake_io):
    fake_io.config.IMAGE_EDIT_MODEL = ''
    with pytest.raises(HTTPException) as exc_info:
        await images_module.image_edits(
            request=None,
            form_data=EditImageForm(image='data:image/png;base64,AAA', prompt='make it blue'),
            user=None,
        )
    assert exc_info.value.status_code == 400


ERROR_MESSAGES_NOT_FOUND = "We could not find what you're looking for :/"


@pytest.mark.asyncio
async def test_image_edits_unresolvable_reference_raises_actionable_error(fake_io, monkeypatch):
    async def fake_get_file_content_by_id(file_id, user):
        raise HTTPException(status_code=404, detail=ERROR_MESSAGES_NOT_FOUND)

    monkeypatch.setattr(images_module, 'get_file_content_by_id', fake_get_file_content_by_id)

    with pytest.raises(HTTPException) as exc_info:
        await images_module.image_edits(
            request=None,
            form_data=EditImageForm(image='photo.jpg', prompt='make it blue'),
            user=None,
        )
    assert exc_info.value.status_code == 404
    detail = str(exc_info.value.detail)
    assert 'photo.jpg' in detail
    assert 'file' in detail
    assert 'http' in detail
    assert 'data:' in detail


@pytest.mark.asyncio
async def test_image_edits_extracts_file_id_from_embedded_file_path(fake_io, monkeypatch):
    seen = []

    async def fake_get_file_content_by_id(file_id, user):
        seen.append(file_id)
        raise HTTPException(status_code=404, detail=ERROR_MESSAGES_NOT_FOUND)

    monkeypatch.setattr(images_module, 'get_file_content_by_id', fake_get_file_content_by_id)

    with pytest.raises(HTTPException):
        await images_module.image_edits(
            request=None,
            form_data=EditImageForm(image='url="/api/v1/files/abc-123/content"', prompt='make it blue'),
            user=None,
        )
    assert seen == ['abc-123']


@pytest.mark.asyncio
async def test_openrouter_base_urls_are_normalized(fake_io):
    fake_io.config.IMAGES_OPENROUTER_API_BASE_URL = 'https://openrouter.ai/api/v1/'
    fake_io.config.IMAGES_EDIT_OPENROUTER_API_BASE_URL = 'https://openrouter.ai/api/v1/'

    await images_module.image_generations(request=None, form_data=CreateImageForm(prompt='a cat'), user=None)
    await images_module.image_edits(
        request=None,
        form_data=EditImageForm(image='data:image/png;base64,AAA', prompt='make it blue'),
        user=None,
    )
    assert [post['url'] for post in fake_io.session.posts] == [
        'https://openrouter.ai/api/v1/images',
        'https://openrouter.ai/api/v1/images',
    ]


@pytest.mark.asyncio
async def test_openrouter_upstream_error_body_is_surfaced(fake_io_error):
    with pytest.raises(HTTPException) as exc_info:
        await images_module.image_edits(
            request=None,
            form_data=EditImageForm(image='data:image/png;base64,AAA', prompt='make it blue'),
            user=None,
        )
    assert 'Model not found' in str(exc_info.value)
