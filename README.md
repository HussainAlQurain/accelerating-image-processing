# Real-Time Image Processing on GPUs

This repository demonstrates real-time image processing (Gaussian blur and Sobel edge detection) using multiple GPU programming models:
- **CPU Baseline** (C++)
- **OpenACC** (C/C++)
- **CUDA C/C++** 
- **CUDA Python (Numba)**

The goal is to compare performance, ease of programming, and scalability for these methods.

---

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation and Setup](#installation-and-setup)
5. [Running the Code](#running-the-code)
   - [CPU Baseline](#cpu-baseline)
   - [OpenACC](#openacc)
   - [CUDA C/C++](#cuda-cc)
   - [CUDA Python (Numba)](#cuda-python-numba)
6. [Results and Analysis](#results-and-analysis)

---

## Project Objectives

1. **Compare** performance across CPU, OpenACC, CUDA Python, and CUDA C/C++ for a standard image-processing pipeline.
2. **Demonstrate** how different GPU programming approaches affect ease of development and runtime speed.
3. **Showcase** a reproducible setup with clear documentation, source code, and benchmark results.

---


- **cpu_baseline**: Contains `main_cpu.cpp` for CPU-only baseline.
- **openacc**: Contains `main_acc.cpp` with OpenACC directives.
- **cuda_c**: Contains `main_cuda.cu` with custom CUDA kernels.
- **cuda_python_numba**: Contains `numba_main.py` showing Python + Numba kernels.
- **docs**: Extended documentation, references, or diagrams.
- **results**: Contains `.qdrep` Nsight Systems profiles, CSV performance logs, and charts.

---

## Prerequisites

- **NVIDIA GPU** with a recent driver supporting CUDA.
- **CMake** or **Make** (optional) for building.  
- **NVIDIA HPC SDK** or `nvc++` compiler (for OpenACC).  
- **NVCC** (for CUDA C/C++).  
- **Python 3.8+** with `numba`, `opencv-python`, `numpy`, etc. (for Python approach).
- **Nsight Systems** (optional) if you want to reproduce profiling results.

---

## Installation and Setup

### You can use docker image to have a complete walkthrough of the project by using the following command:
```bash
docker pull hussain50/image-processing-project
```
Inside the image, you will find a folder named `image-processing` which contains a file named ```image-processing-project.ipynb```


1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/my-gpu-image-processing-project.git
   cd my-gpu-image-processing-project
   ```

2. **Install Dependencies:**

- For CPU and OpenACC code, install OpenCV dev libraries and the NVIDIA HPC SDK or another OpenACC-compatible compiler.
- For CUDA C/C++:
```bash
sudo apt-get install nvidia-cuda-toolkit
```
- For Python Numba:
```bash
pip install numba opencv-python numpy
```
3. **Check GPU Availability:**
```bash
nvidia-smi
```
Ensure your GPU is detected.

#### Before Running the Code ensure to include a video file and link it in the code, or use your camera directly by changing the line in main function cv::VideoCapture cap("sample.mp4"); to cv::VideoCapture cap(0);

- CPU Baseline:
```bash
cd cpu_baseline
# Example compile command:
g++ -o cpu_program main_cpu.cpp `pkg-config --cflags --libs opencv4`
./cpu_program
```
- OpenACC:
```bash
cd openacc
# Using nvc++ from NVIDIA HPC SDK
nvc++ -acc -ta=multicore -O2 main_acc.cpp -o acc_program `pkg-config --cflags --libs opencv4`
./acc_program
```
(Adjust -ta=tesla:cc80 for GPU offload if supported.)

- CUDA C/C++:
```bash
cd cuda_c
nvcc -O2 main_cuda.cu -o cuda_program `pkg-config --cflags --libs opencv4`
./cuda_program
```
- CUDA Python (Numba):
```bash
cd cuda_python_numba
python numba_main.py
```
(Ensure you’ve installed numba, opencv-python, numpy.)

## Results and Analysis

- FPS and Execution Times are stored in the console outputs.
- Nsight Systems profile files (.qdrep) located in results/:
    - cpu_profile_report.qdrep
    - acc_profile_report.qdrep
    - cuda_profile_report.qdrep
    - numba_profile_report.qdrep
- Sample Plot: See results/chart.png for a bar chart of performance comparisons.
#### Key Observations:

- CUDA C/C++ showed the best performance (~15x over CPU).
- OpenACC improved performance on CPU but less than pure CUDA.
- Numba provides moderate speedups with less code complexity than CUDA C/C++.
