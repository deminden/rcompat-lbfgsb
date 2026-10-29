#!/usr/bin/env python3
"""Backward-compatible mixed-bound verification command."""
from verify_oracles import verify

if __name__ == "__main__":
    try:
        count = verify("mixed_bounds")
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(f"{count} mixed-bound cases; payload, recipe SHA256 and reference metadata verified")
