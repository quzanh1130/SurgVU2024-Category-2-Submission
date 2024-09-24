# SurgVU2024_Category-2-Submission


## Instruction Overview
This document provides a step-by-step guide to install the necessary libraries, download the required models and config files, and execute the method for the SurgVU2024 Category-2 submission. You will also find instructions for exporting the model in a container for submission.

## Table of Contents
1. [Clone Repository](#clone-repository)
2. [Install Libraries](#install)
3. [Download Model and Config](#model-and-config-file)
4. [Running Method](#running-method)
5. [Export container](#export-container)


## 1. Clone Repository
To begin, you need to clone the repository that contains the codebase for the project.

```bash
git clone https://github.com/quzanh1130/SurgVU2024-Category-2-Submission.git
cd SurgVU2024-Category-2-Submission
```

## 2. Install Libraries

Our code is based on [mmaction2](https://github.com/open-mmlab/mmaction2). To run our code, you must install mmaction2.

### 2.1 Install ffmpeg for Video Processing:

On Ubuntu or Debian
```bash
sudo apt update && sudo apt install ffmpeg
```

On Arch Linux
```bash
sudo pacman -S ffmpeg
```

On MacOS using Homebrew (https://brew.sh/)
```bash
brew install ffmpeg
```

On Windows using Chocolatey (https://chocolatey.org/)
```bash
choco install ffmpeg
```

On Windows using Scoop (https://scoop.sh/)
```bash
scoop install ffmpeg
```

### 2.2 Install Python Dependencies

**Using Anaconda**
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# Install pytorch at: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0

cd mmaction2
pip install -v -e .
```

## 3. Download Model and Config File

First you need to create folder to store model

```bash
cd mmaction2
mkdir work_dirs
cd work_dirs
mkdir slowfast
```

If the model file is missing at `mmaction2/work_dirs/slowfast/best_model.pth`, you can download it from the following Google Drive link:
- [Download Model](https://drive.google.com/file/d/12cy_RJnHYARxj452y8fTgH920MSZgzt6/view?usp=drive_link)

If the config file is missing at `mmaction2/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_rgb_1fps_video_final.py`, you can download it from the following Google Drive link:

- [Download Config](https://drive.google.com/file/d/1PcUHVhJlPDXVnHBM-7tW4hQbIpo-F5iN/view?usp=sharing)

## 4. Runing method
To run the model and test your input video, execute the following steps:

1. Navigate to the directory where the process_testing.py file is located.

2. Run the script using the following command:

```bash
python process_testing.py
```

3. After successful execution, the results will be saved in a JSON file at:
```bash
 mmaction2/output/surgical-step-classification.json
```

### 4.1 Customizing Config and Model
To use different configurations or model checkpoints, modify the `process_testing.py` file:

```python
#Config and checkpoint
config_path = 'configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_rgb_1fps_video_final.py'
checkpoint_path = 'work_dirs/slowfast/best_model.pth'
```

### 4.2 Testing a Different Video
To test a different input video, modify the `process_testing.py` file:

```python
if __name__ == "__main__":
    detector = SurgVUClassify()
    video_file = detector.input_path / "vid_1_short.mp4" # Change your test video
    detector.predict(str(video_file))
```

## 5. Export container

To prepare your submission, you will need to export the containerized application using Docker. If you don't have Docker installed, follow the links below for installation instructions:

1. [Install Docker on Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04)

2. [Install Docker on Window](https://www.simplilearn.com/tutorials/docker-tutorial/install-docker-on-windows)

### Export Container

Once Docker is installed, navigate to the `mmaction2` directory and run the following command to export the container:

```bash
cd mmaction2
bash export.sh
```

After successful execution, a tarball of the container will be created at:
```bash
mmaction2/surgtoolloc_det.tar.gz
```

This file is your submission container.