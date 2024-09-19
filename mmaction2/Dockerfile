# Use the base image built from Dockerfile.base
FROM surgvu_cat2_base:latest

# Clone and Install MMAction2
RUN git clone https://github.com/open-mmlab/mmaction2.git /home/algorithm/mmaction2
WORKDIR /home/algorithm/mmaction2
RUN mkdir -p /home/algorithm/mmaction2/data
RUN git checkout main
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .

# Copy the entire directory containing process.py to the container
COPY --chown=algorithm:algorithm . /opt/algorithm/

# Copy the checkpoint file into the container
COPY --chown=algorithm:algorithm work_dirs /home/algorithm/mmaction2/work_dirs

# Copy the configuration files into the container
COPY --chown=algorithm:algorithm configs /home/algorithm/mmaction2/configs

# Copy the expected output JSON into the container
COPY --chown=algorithm:algorithm expected_output_detection.json /opt/algorithm/

# Add /opt/algorithm to PYTHONPATH (optional)
ENV PYTHONPATH=/opt/algorithm

# Use JSON format for ENTRYPOINT to avoid OS signal issues
ENTRYPOINT ["python", "/opt/algorithm/process.py"]
