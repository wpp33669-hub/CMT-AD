#!/usr/bin/env python
"""Compatibility entry point for the refactored stronger CMKT pipeline."""

from main import parse_args, run


if __name__ == "__main__":
    run(parse_args())

