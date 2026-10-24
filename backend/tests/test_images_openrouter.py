import base64
import io

import pytest
from open_webui.routers.images import (
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
    assert headers['HTTP-Referer'] == 'https://openwebui.com/'
    assert headers['X-Title'] == 'Open WebUI'


def test_openrouter_request_size():
    assert openrouter_request_size('1024x1024') == '1024x1024'
    assert openrouter_request_size('auto') is None
    assert openrouter_request_size('2K') is None
    assert openrouter_request_size('') is None
    assert openrouter_request_size(None) is None


def test_build_openrouter_generation_payload():
    payload = build_openrouter_generation_payload(
        'openai/gpt-image-1', 'a cat', 1, '1024x1024', {'quality': 'high'}
    )
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
    assert payload['input_references'] == [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}}
    ]
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
