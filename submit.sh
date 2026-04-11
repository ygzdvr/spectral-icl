#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$SCRIPT_DIR/experiments"

shopt -s nullglob
experiment_scripts=("$EXPERIMENTS_DIR"/*.sh)

if [ ${#experiment_scripts[@]} -eq 0 ]; then
  echo "No experiment scripts found in $EXPERIMENTS_DIR"
  exit 1
fi

echo "Running ${#experiment_scripts[@]} experiment scripts from $EXPERIMENTS_DIR"

for script in "${experiment_scripts[@]}"; do
  echo
  echo "============================================================"
  echo "Running $(basename "$script")"
  echo "============================================================"
  bash "$script"
done

echo
echo "All experiment scripts completed."

