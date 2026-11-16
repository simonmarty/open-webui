from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

log = logging.getLogger(__name__)


class SigV4Error(Exception):
    pass


class SigV4ConfigError(SigV4Error):
    pass


class SigV4CredentialsError(SigV4Error):
    pass


def is_sigv4(config: dict | None) -> bool:
    if not isinstance(config, dict):
        return False
    auth_type = config.get('auth_type')
    return isinstance(auth_type, str) and auth_type.strip().lower() == 'sigv4'


def _required_setting(config: dict | None, key: str, label: str) -> str:
    value = (config or {}).get(key)
    value = value.strip() if isinstance(value, str) else ''
    if not value:
        raise SigV4ConfigError(f'{label} is required for SigV4 connections. Set it on the connection.')
    return value


@lru_cache(maxsize=1)
def get_session() -> boto3.Session:
    return boto3.Session()


def sign_request(
    *,
    method: str,
    url: str,
    headers: dict | None,
    body: Any,
    config: dict | None,
) -> dict:
    region = _required_setting(config, 'aws_region', 'AWS region')
    service = _required_setting(config, 'aws_signing_service', 'AWS signing service')

    credentials = get_session().get_credentials()
    if credentials is None:
        raise SigV4CredentialsError('No AWS credentials are available to sign the request. ')

    if body is None:
        data = None
    elif isinstance(body, str):
        data = body.encode('utf-8')
    elif isinstance(body, (bytes, bytearray, memoryview)):
        data = bytes(body)
    else:
        raise SigV4Error(f'Cannot sign a request body of type {type(body).__name__}; expected str, bytes, or None.')

    outgoing = {key: value for key, value in (headers or {}).items() if value is not None}

    aws_request = AWSRequest(method=method.upper(), url=url, data=data, headers=outgoing)
    SigV4Auth(credentials.get_frozen_credentials(), service, region).add_auth(aws_request)

    return dict(aws_request.headers.items())
