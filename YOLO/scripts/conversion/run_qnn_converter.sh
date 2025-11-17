#!/bin/bash

# QNN ONNX Converter Wrapper
# This script properly sets up the environment for QNN tools

# Deactivate conda
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate 2>/dev/null || true
fi

# Set QNN SDK root
export QAIRT_SDK_ROOT=~/snpe-sdk/2.40.0.251030
export QNN_SDK_ROOT=$QAIRT_SDK_ROOT
export SNPE_ROOT=$QAIRT_SDK_ROOT

# Source environment setup
cd $QAIRT_SDK_ROOT
source bin/envsetup.sh

# Run the command
"$@"
