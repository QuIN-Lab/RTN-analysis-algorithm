#!/usr/bin/env bash

output_file=${1:-USAGE.md}

venv/bin/doctool document-cli main \
	--output-dir docs/command-output-example-images \
	--output-file "${output_file}"
