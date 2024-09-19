#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Build the base image
docker build -t surgvu_cat2_base -f Dockerfile.base "$SCRIPTPATH"

# Build the final image using the base image
docker build -t surgvu_cat2 -f Dockerfile "$SCRIPTPATH"