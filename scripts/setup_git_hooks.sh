#!/bin/bash
# Setup Git hooks for autonomous backlog management

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOKS_DIR="$REPO_ROOT/.git/hooks"

# Create prepare-commit-msg hook
cat > "$HOOKS_DIR/prepare-commit-msg" << 'EOF'
#!/bin/bash
# Enable rerere for merge conflict resolution
git config rerere.enabled true
git config rerere.autoupdate true
EOF

# Create pre-push hook
cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash
# Auto-rebase before push to avoid merge conflicts

set -e

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Skip for main branch
if [[ "$CURRENT_BRANCH" == "main" ]]; then
    exit 0
fi

echo "Auto-rebasing $CURRENT_BRANCH onto main..."

# Enable rerere
git config rerere.enabled true
git config rerere.autoupdate true

# Fetch latest main
git fetch origin main

# Attempt rebase
if ! git rebase origin/main; then
    echo "ERROR: Automatic rebase failed. Manual intervention required."
    echo "Run 'git rebase --abort' to cancel the rebase"
    exit 1
fi

echo "Rebase successful"
EOF

# Make hooks executable
chmod +x "$HOOKS_DIR/prepare-commit-msg"
chmod +x "$HOOKS_DIR/pre-push"

echo "Git hooks installed successfully"
echo "Rerere is now enabled for automatic conflict resolution"