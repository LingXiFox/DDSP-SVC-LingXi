#!/bin/bash
# LingXi SVC bootstrap: isolated venv + pinned runtime + user launchers.
# No system Python pollution, no shell environment changes.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/environment/.venv"
LAUNCHER_DIR="$HOME/.local/bin"

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -U pip wheel
"$VENV/bin/python" -m pip install -e "$ROOT/app"

mkdir -p "$LAUNCHER_DIR"

write_launcher() {
    local name="$1" target="$2" err_title="$3" err_hint="$4"
    local dest="$LAUNCHER_DIR/$name"
    cat > "$dest" <<EOF
#!/bin/zsh

ROOT="$ROOT"
BIN="\$ROOT/environment/.venv/bin/$target"

if [[ ! -d "\$ROOT" ]]; then
    echo "LingXi SVC error: project volume/path is unavailable:"
    echo "\$ROOT"
    exit 1
fi

if [[ ! -x "\$BIN" ]]; then
    echo "LingXi SVC error: $err_title"
    echo "\$BIN"
    echo "$err_hint"
    exit 1
fi

exec "\$BIN" "\$@"
EOF
    chmod +x "$dest"
}

write_launcher "lingxi-svc" "lingxi-svc" \
    "runtime environment is missing or broken:" \
    "Run lingxi-selfcheck or reinstall using app/install.sh."
write_launcher "lingxi-selfcheck" "lingxi-selfcheck" \
    "self-check entrypoint is unavailable:" \
    "Reinstall using app/install.sh."

if ! grep -q 'HOME/.local/bin' "$HOME/.zshrc" 2>/dev/null; then
    printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.zshrc"
    echo "Added ~/.local/bin to PATH in ~/.zshrc (restart shell to apply)."
else
    echo "~/.local/bin already on PATH config."
fi

echo "Done. In a new shell:"
echo "  lingxi-selfcheck"
echo "  lingxi-svc vocal.wav"
