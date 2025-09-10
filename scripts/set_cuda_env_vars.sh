#!/bin/bash

# Set environment variables for conda environment activation
echo 'export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"' > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

# Set environment variables for conda environment deactivation
echo 'unset LIBRARY_PATH' > "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
echo 'unset LD_LIBRARY_PATH' >> "$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh"
