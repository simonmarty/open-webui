import datetime as dt
import re
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import boto3
import botocore.auth
import pytest
from botocore.credentials import Credentials
from open_webui.utils import aws_sigv4
from open_webui.utils.aws_sigv4 import (
    SigV4ConfigError,
    SigV4CredentialsError,
    SigV4Error,
    get_session,
    is_sigv4,
    sign_request,
)

PERMANENT_CREDENTIALS = Credentials(access_key='AKIAIOSFODNN7EXAMPLE', secret_key='wJalrXUtnFEMI/K7MDENG')
TEMPORARY_CREDENTIALS = Credentials(
    access_key='ASIAIOSFODNN7EXAMPLE',
    secret_key='wJalrXUtnFEMI/K7MDENG',
    token='FwoGZXIvYXdzEXAMPLETOKEN',
)

FROZEN_TIME = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC)

URL = 'https://bedrock-runtime.us-east-1.amazonaws.com/v1/chat/completions'
SIGV4_CONFIG = {'auth_type': 'sigv4', 'aws_region': 'us-east-1', 'aws_signing_service': 'bedrock'}


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(botocore.auth, 'get_current_datetime', lambda: FROZEN_TIME)


@pytest.fixture(autouse=True)
def fresh_session():
    get_session.cache_clear()
    yield
    get_session.cache_clear()


@pytest.fixture
def env_credentials(monkeypatch):
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'AKIAIOSFODNN7EXAMPLE')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'wJalrXUtnFEMI/K7MDENG')
    monkeypatch.delenv('AWS_SESSION_TOKEN', raising=False)


@contextmanager
def stub_session(credentials):
    session = SimpleNamespace(get_credentials=lambda: credentials)
    with patch.object(aws_sigv4, 'get_session', lambda: session):
        yield


def sign(**overrides):
    credentials = overrides.pop('credentials', PERMANENT_CREDENTIALS)
    config = {
        'auth_type': 'sigv4',
        'aws_region': overrides.pop('region', 'us-east-1'),
        'aws_signing_service': overrides.pop('service', 'bedrock'),
    }
    kwargs = {
        'method': 'POST',
        'url': URL,
        'headers': {'Content-Type': 'application/json'},
        'body': '{"model":"claude"}',
        'config': config,
    }
    kwargs.update(overrides)
    with stub_session(credentials):
        return sign_request(**kwargs)


def signature_of(headers):
    match = re.search(r'Signature=([0-9a-f]+)', headers['Authorization'])
    assert match, f'no signature in {headers["Authorization"]!r}'
    return match.group(1)


def signed_headers_of(headers):
    match = re.search(r'SignedHeaders=([^,]+)', headers['Authorization'])
    assert match, f'no SignedHeaders in {headers["Authorization"]!r}'
    return match.group(1).split(';')


class TestIsSigV4:
    def test_recognizes_sigv4_auth_type(self):
        assert is_sigv4({'auth_type': 'sigv4'}) is True

    def test_is_case_insensitive(self):
        assert is_sigv4({'auth_type': 'SigV4'}) is True

    def test_rejects_other_auth_types(self):
        assert is_sigv4({'auth_type': 'bearer'}) is False

    def test_rejects_missing_config(self):
        assert is_sigv4(None) is False
        assert is_sigv4({}) is False


class TestRegionAndService:
    def test_missing_region_is_an_error(self):
        with pytest.raises(SigV4ConfigError):
            sign(region='')

    def test_blank_region_is_an_error(self):
        with pytest.raises(SigV4ConfigError):
            sign(region='   ')

    def test_region_has_no_environment_fallback(self, monkeypatch):
        monkeypatch.setenv('AWS_REGION', 'us-west-2')
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-west-2')
        with pytest.raises(SigV4ConfigError):
            sign(region='')

    def test_missing_service_is_an_error(self):
        with pytest.raises(SigV4ConfigError):
            sign(service='')

    def test_blank_service_is_an_error(self):
        with pytest.raises(SigV4ConfigError):
            sign(service='  ')

    def test_region_appears_in_the_credential_scope(self):
        assert '/20260102/eu-west-1/bedrock/aws4_request' in sign(region='eu-west-1')['Authorization']

    def test_service_appears_in_the_credential_scope(self):
        assert '/us-east-1/mantle/aws4_request' in sign(service='mantle')['Authorization']


class TestSignedHeaders:
    def test_adds_sigv4_authorization_header(self):
        assert sign()['Authorization'].startswith('AWS4-HMAC-SHA256 ')

    def test_credential_scope_names_date_region_and_service(self):
        headers = sign()
        assert 'Credential=AKIAIOSFODNN7EXAMPLE/20260102/us-east-1/bedrock/aws4_request' in headers['Authorization']

    def test_signature_is_hex_sha256(self):
        assert re.fullmatch(r'[0-9a-f]{64}', signature_of(sign()))

    def test_adds_amz_date_header(self):
        assert sign()['X-Amz-Date'] == '20260102T030405Z'

    def test_signs_the_host_header(self):
        assert 'host' in signed_headers_of(sign())

    def test_preserves_caller_headers(self):
        headers = sign(headers={'Content-Type': 'application/json', 'X-OpenWebUI-User-Id': 'abc'})
        assert headers['Content-Type'] == 'application/json'
        assert headers['X-OpenWebUI-User-Id'] == 'abc'

    def test_does_not_mutate_the_caller_headers(self):
        original = {'Content-Type': 'application/json'}
        sign(headers=original)
        assert original == {'Content-Type': 'application/json'}

    def test_signature_covers_the_request_body(self):
        assert signature_of(sign(body='{"model":"a"}')) != signature_of(sign(body='{"model":"b"}'))

    def test_signature_covers_the_http_method(self):
        assert signature_of(sign(method='POST')) != signature_of(sign(method='GET'))

    def test_signature_covers_the_url_path(self):
        other = 'https://bedrock-runtime.us-east-1.amazonaws.com/v1/embeddings'
        assert signature_of(sign()) != signature_of(sign(url=other))

    def test_signature_covers_the_query_string(self):
        assert signature_of(sign(url=f'{URL}?a=1')) != signature_of(sign(url=f'{URL}?a=2'))

    def test_string_and_bytes_bodies_sign_identically(self):
        assert signature_of(sign(body='{"model":"claude"}')) == signature_of(sign(body=b'{"model":"claude"}'))

    def test_empty_body_is_signed(self):
        assert re.fullmatch(r'[0-9a-f]{64}', signature_of(sign(method='GET', body=None)))

    def test_unsignable_body_type_is_rejected(self):
        with pytest.raises(SigV4Error):
            sign(body={'model': 'claude'})

    def test_temporary_credentials_add_a_signed_session_token(self):
        headers = sign(credentials=TEMPORARY_CREDENTIALS)
        assert headers['X-Amz-Security-Token'] == TEMPORARY_CREDENTIALS.token
        assert 'x-amz-security-token' in signed_headers_of(headers)

    def test_permanent_credentials_omit_the_session_token(self):
        assert 'X-Amz-Security-Token' not in sign()

    def test_missing_credentials_raise(self):
        with pytest.raises(SigV4CredentialsError):
            sign(credentials=None)

    def test_a_stale_authorization_header_is_replaced(self):
        headers = sign(headers={'Content-Type': 'application/json', 'Authorization': 'Bearer leftover'})
        assert headers['Authorization'].startswith('AWS4-HMAC-SHA256 ')


class TestCredentialLifecycle:
    def test_reuses_one_session_across_requests(self, monkeypatch, env_credentials):
        sessions = []
        real_session = boto3.Session

        def counting_session(*args, **kwargs):
            sessions.append(1)
            return real_session(*args, **kwargs)

        monkeypatch.setattr(boto3, 'Session', counting_session)

        for _ in range(3):
            sign_request(method='GET', url=URL, headers={}, body=None, config=SIGV4_CONFIG)

        assert len(sessions) == 1

    def test_signs_using_credentials_from_the_default_chain(self, env_credentials):
        headers = sign_request(
            method='POST',
            url=URL,
            headers={'Content-Type': 'application/json'},
            body='{"model":"claude"}',
            config=SIGV4_CONFIG,
        )

        assert 'Credential=AKIAIOSFODNN7EXAMPLE/20260102/us-east-1/bedrock/aws4_request' in headers['Authorization']

    def test_missing_region_raises_before_resolving_credentials(self, env_credentials):
        with pytest.raises(SigV4ConfigError):
            sign_request(method='POST', url=URL, headers={}, body=None, config={'auth_type': 'sigv4'})

        assert get_session.cache_info().currsize == 0

    def test_unresolvable_credentials_raise(self):
        with stub_session(None):
            with pytest.raises(SigV4CredentialsError):
                sign_request(method='POST', url=URL, headers={}, body=None, config=SIGV4_CONFIG)
