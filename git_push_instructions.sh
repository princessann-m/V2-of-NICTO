#!/usr/bin/env bash
# Instructions to create a branch, commit, and push changes to GitHub.
# You must run these locally; do NOT share your GitHub token here.

set -euo pipefail

BRANCH=feature/mom-nicto-prototype
git checkout -b "$BRANCH"
git add .
git commit -m "Add MoM Phase1 prototype and nicto training scaffold"
git push --set-upstream origin "$BRANCH"

echo "Pushed branch $BRANCH. Create a Pull Request on GitHub to merge to main."

# Optional: use GitHub CLI to create PR (requires gh auth):
# gh pr create --fill --base main --head "$BRANCH"
