#!/usr/bin/env python3
"""
download_weights.py

Download model artifacts from S3 (single file or prefix).
"""
from urllib.parse import urlparse
import argparse
import os
import boto3
import re
import sys

def download_s3_prefix(s3, bucket, prefix, dest):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = os.path.relpath(key, prefix)
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            print("Downloading", key, "->", target)
            s3.download_file(bucket, key, target)


def ensure_versioned(key_or_prefix: str):
    """
    Ensure the given S3 key/prefix contains a version tag like /v1 or /v1.0.0/.
    If not present, abort unless ALLOW_UNVERSIONED env var is set to "1".
    """
    if os.environ.get("ALLOW_UNVERSIONED", "0") == "1":
        return
    # look for '/v' followed by digit (e.g. /v1 or /v1.0.0)
    if re.search(r"/v\d+(\.\d+)*(/|$)", key_or_prefix):
        return
    print("ERROR: S3 path does not contain a version tag (e.g., /v1.0.0/).")
    print("Set ALLOW_UNVERSIONED=1 to override, or use a versioned S3 prefix.")
    sys.exit(2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s3-uri", required=True)
    p.add_argument("--dest", required=True)
    args = p.parse_args()
    parsed = urlparse(args.s3_uri)
    if parsed.scheme != "s3":
        raise SystemExit("s3-uri must be s3://")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    # enforce versioned paths for reproducibility
    ensure_versioned(key)
    if args.s3_uri.endswith("/") or key == "":
        download_s3_prefix(s3, bucket, key, args.dest)
    else:
        os.makedirs(os.path.dirname(os.path.join(args.dest, os.path.basename(key))), exist_ok=True)
        target = os.path.join(args.dest, os.path.basename(key))
        print("Downloading single file", key, "->", target)
        s3.download_file(bucket, key, target)

if __name__ == "__main__":
    main()

