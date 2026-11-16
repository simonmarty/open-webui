import pytest
from open_webui.main import app, sigv4_error_handler
from open_webui.utils.aws_sigv4 import SigV4ConfigError, SigV4CredentialsError, SigV4Error
from open_webui.utils.json_codec import JSONCodec
from starlette.requests import Request


def fake_request():
    return Request({'type': 'http', 'method': 'POST', 'path': '/openai/chat/completions', 'headers': []})


def detail_of(response):
    return JSONCodec.loads(response.body)['detail']


def test_handler_is_registered_for_every_sigv4_failure():
    assert SigV4Error in app.exception_handlers


@pytest.mark.asyncio
async def test_reports_a_missing_region_as_a_server_error():
    response = await sigv4_error_handler(fake_request(), SigV4ConfigError('An AWS region is required.'))

    assert response.status_code == 500
    assert 'An AWS region is required.' in detail_of(response)


@pytest.mark.asyncio
async def test_reports_unresolvable_credentials():
    response = await sigv4_error_handler(fake_request(), SigV4CredentialsError('No AWS credentials are available.'))

    assert response.status_code == 500
    assert 'No AWS credentials are available.' in detail_of(response)
