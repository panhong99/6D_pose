#!/usr/bin/env bash
# Install Meta's official SAM2 into the selected Python environment.
# No checkpoints are downloaded by this script. Run --help for options.
set -euo pipefail

PYTHON_BIN="python"
METHOD="auto"
REPO_DIR="./sam2_src"
REVISION=""
BUILD_CUDA="0"
REPO_URL="https://github.com/facebookresearch/sam2.git"

usage() {
    cat <<'EOF'
Usage: bash install_sam2.sh [options]

  --method auto|git|editable  auto: reuse a working install, otherwise try pip
                             from official Git, then clone + editable fallback.
                             git: pip from Git only; editable: local clone only.
  --repo_dir PATH            Clone/reuse directory (default: ./sam2_src).
  --python EXECUTABLE        Python interpreter (default: python).
  --revision REF             Optional official Git commit, tag, or branch.
  --build_cuda               Build the optional CUDA postprocessing extension;
                             requires matching PyTorch, CUDA toolkit and nvcc.
  -h, --help                 Show this help without installing anything.

Default SAM2_BUILD_CUDA=0 allows CPU use and CUDA inference without compiling
the optional small-hole/sprinkle cleanup extension. Python >=3.10 is required.
Explicit --method, --revision or --build_cuda requests are not skipped merely
because SAM2 is already installed. Existing clone changes are never discarded.

Examples:
  bash install_sam2.sh
  bash install_sam2.sh --method editable --repo_dir ./third_party/sam2
  bash install_sam2.sh --python ./.venv/bin/python --revision main
EOF
}

die() { echo "Error: $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) need_value "$@"; METHOD="$2"; shift 2 ;;
        --repo_dir) need_value "$@"; REPO_DIR="$2"; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
        --revision) need_value "$@"; REVISION="$2"; shift 2 ;;
        --build_cuda) BUILD_CUDA="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

case "$METHOD" in auto|git|editable) ;; *) die "--method must be auto, git, or editable" ;; esac
[[ -z "$REVISION" || "$REVISION" != -* ]] || die "--revision must be a Git ref, not an option"
command -v "$PYTHON_BIN" >/dev/null || die "Python interpreter not found: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "SAM2 requires Python >=3.10")'
"$PYTHON_BIN" -m pip --version >/dev/null || die "pip is unavailable in $PYTHON_BIN"

check_install() {
    "$PYTHON_BIN" -c 'import sam2; from sam2.build_sam import build_sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; print("SAM2 ready:", sam2.__file__)'
    if [[ "$BUILD_CUDA" == 1 ]]; then
        "$PYTHON_BIN" -c 'from sam2 import _C; print("SAM2 CUDA extension:", _C.__file__)'
    fi
}

if [[ "$METHOD" == auto && -z "$REVISION" && "$BUILD_CUDA" == 0 ]] && check_install >/dev/null 2>&1; then
    check_install
    echo "Use --method git or --method editable to request a reinstall/update."
    exit 0
fi

command -v git >/dev/null || die "git is required; install it and rerun this script"
export SAM2_BUILD_CUDA="$BUILD_CUDA"
PIP_OPTIONS=()
if [[ "$BUILD_CUDA" == 1 ]]; then
    command -v nvcc >/dev/null || die "--build_cuda requires nvcc from a matching CUDA toolkit"
    "$PYTHON_BIN" -c 'import torch; print("Building with torch", torch.__version__, "CUDA", torch.version.cuda)'
    export SAM2_BUILD_ALLOW_ERRORS=0
    PIP_OPTIONS+=(--no-build-isolation --no-cache-dir)
fi

install_from_git() {
    local spec="git+${REPO_URL}"
    if [[ -n "$REVISION" ]]; then spec+="@${REVISION}"; fi
    echo "Installing official SAM2 from Git (SAM2_BUILD_CUDA=$SAM2_BUILD_CUDA)."
    "$PYTHON_BIN" -m pip install --upgrade "${PIP_OPTIONS[@]}" "$spec" || return $?
    check_install
}

install_editable() {
    if [[ ! -e "$REPO_DIR" ]]; then
        git clone -- "$REPO_URL" "$REPO_DIR"
    else
        [[ -d "$REPO_DIR/.git" ]] || die "Existing --repo_dir is not a Git clone: $REPO_DIR"
        local origin
        origin="$(git -C "$REPO_DIR" remote get-url origin)"
        case "$origin" in
            https://github.com/facebookresearch/sam2|https://github.com/facebookresearch/sam2.git|git@github.com:facebookresearch/sam2.git) ;;
            *) die "Existing clone does not use Meta's official SAM2 origin: $origin" ;;
        esac
        echo "Reusing local SAM2 clone at $REPO_DIR (no automatic reset or pull)."
    fi
    if [[ -n "$REVISION" ]]; then
        [[ -z "$(git -C "$REPO_DIR" status --porcelain)" ]] || die "Clone has local changes; use a clean --repo_dir for --revision"
        git -C "$REPO_DIR" fetch origin "$REVISION"
        git -C "$REPO_DIR" checkout --detach FETCH_HEAD
    fi
    echo "Installing editable SAM2 revision $(git -C "$REPO_DIR" rev-parse HEAD)."
    "$PYTHON_BIN" -m pip install "${PIP_OPTIONS[@]}" -e "$REPO_DIR"
    check_install
}

case "$METHOD" in
    git) install_from_git ;;
    editable) install_editable ;;
    auto)
        if ! install_from_git; then
            echo "Git pip installation/import failed; trying a local editable clone." >&2
            install_editable
        fi
        ;;
esac
echo "Installation complete. The pipeline downloads SAM2 weights on first use, or accepts --sam2_checkpoint."
