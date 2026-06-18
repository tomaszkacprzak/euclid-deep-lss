#!/usr/bin/env bash
set -euo pipefail

# TensorFlow may only appear in documented TFRecord compatibility code.
# Keep this command in sync with README.md so contributors can run the exact
# same search locally. Notebook files are intentionally excluded.
TF_PATTERN='import tensorflow|from tensorflow|tf\.'
TF_ALLOWED_FILES_REGEX='^deep_lss/data/tfrecords\.py$'

# These legacy TensorFlow-adjacent imports should not appear in active code.
LEGACY_PATTERN='horovod\.tensorflow|tensorflow_probability|(^|[[:space:]])import deepsphere|from deepsphere'
ACTIVE_CODE_EXCLUDES=(-g '!deep_lss/deprecated/**' -g '!deep_lss/nets/legacy/**')

check_allowed_tensorflow_imports() {
    local matches disallowed
    matches=$(rg "$TF_PATTERN" deep_lss -g '!*.ipynb' || true)
    if [[ -z "$matches" ]]; then
        return 0
    fi

    disallowed=$(printf '%s\n' "$matches" | cut -d: -f1 | grep -Ev "$TF_ALLOWED_FILES_REGEX" || true)
    if [[ -n "$disallowed" ]]; then
        disallowed=$(printf '%s\n' "$matches" | grep -Ev "^($TF_ALLOWED_FILES_REGEX):" || true)
    fi
    if [[ -n "$disallowed" ]]; then
        cat >&2 <<'MSG'
TensorFlow usage is restricted to documented TFRecord compatibility modules.
Allowed files:
  - deep_lss/data/tfrecords.py

Disallowed matches:
MSG
        printf '%s\n' "$disallowed" >&2
        return 1
    fi
}

check_legacy_imports_absent_from_active_code() {
    local matches
    matches=$(rg "$LEGACY_PATTERN" deep_lss -g '!*.ipynb' "${ACTIVE_CODE_EXCLUDES[@]}" || true)
    if [[ -n "$matches" ]]; then
        cat >&2 <<'MSG'
Active code must not import horovod.tensorflow, tensorflow_probability, or the old TensorFlow deepsphere package.
Move active paths to PyTorch equivalents or quarantine them under an explicitly deprecated/legacy tree.

Disallowed active-code matches:
MSG
        printf '%s\n' "$matches" >&2
        return 1
    fi
}

check_allowed_tensorflow_imports
check_legacy_imports_absent_from_active_code
