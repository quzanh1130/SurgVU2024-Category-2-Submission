#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

bash build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="4g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create surgvu_cat2-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
    --memory="${MEM_LIMIT}" \
    --memory-swap="${MEM_LIMIT}" \
    --network="none" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    --shm-size="128m" \
    --pids-limit="256" \
    -v $SCRIPTPATH/test/:/input/ \
    -v surgvu_cat2-output-$VOLUME_SUFFIX:/output/ \
    surgvu_cat2

# Check if the output file was created
if docker run --rm \
        -v surgvu_cat2-output-$VOLUME_SUFFIX:/output/ \
        busybox test -f /output/surgical-step-classification.json; then
    # Pretty print the JSON output
    docker run --rm \
        -v surgvu_cat2-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/surgical-step-classification.json | python -m json.tool

    # Compare JSON output with expected output
    docker run --rm \
        -v surgvu_cat2-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        -v $SCRIPTPATH/:/tmp/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/surgical-step-classification.json')); f2 = json.load(open('/tmp/expected_output_detection.json')); sys.exit(f1 != f2);"

    if [ $? -eq 0 ]; then
        echo "Tests successfully passed..."
    else
        echo "Expected output was not found..."
    fi
else
    echo "Output file not found..."
fi

docker volume rm surgvu_cat2-output-$VOLUME_SUFFIX
