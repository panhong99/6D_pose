"""Shared version selection for training, evaluation and GUI replay."""
import os

# Explicit PP_VERSION takes priority. PP_V5 remains supported for old experiments.
VERSION = os.environ.get('PP_VERSION', 'v5' if os.environ.get('PP_V5') == '1' else 'v6')
if VERSION not in ('v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'):
    raise ValueError(f'Unsupported PP_VERSION: {VERSION}')
EXPERIMENT_NAME = f'm1013_2f85_rim_{VERSION}'
