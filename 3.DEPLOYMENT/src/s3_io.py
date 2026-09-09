"""S3 access. Default credential chain only.

Never pass aws_access_key_id= here. Explicit keys OVERRIDE and DISABLE IRSA, so the pod
would authenticate as whatever stale key it found instead of its scoped role — and the
failure is silent until an AccessDenied on a prefix the role could have reached.
"""
from __future__ import annotations

import io
import json
import os

import boto3
import pandas as pd

BUCKET = os.environ.get("S3_BUCKET", "qcomm-rfm")
REGION = os.environ.get("AWS_REGION", "ap-south-2")
ENV = os.environ.get("ENV_PREFIX", "dev")

_client = None


def client():
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=REGION)
    return _client


def key(rel: str) -> str:
    return f"{ENV}/{rel}"


def get_bytes(rel: str) -> bytes:
    return client().get_object(Bucket=BUCKET, Key=key(rel))["Body"].read()


def get_json(rel: str) -> dict:
    return json.loads(get_bytes(rel))


def get_parquet(rel: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(get_bytes(rel)))


def put_json(obj: dict, rel: str) -> None:
    client().put_object(Bucket=BUCKET, Key=key(rel),
                        Body=json.dumps(obj, indent=2, default=str).encode())


def put_bytes(data: bytes, rel: str) -> None:
    client().put_object(Bucket=BUCKET, Key=key(rel), Body=data)
