#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cuda_runtime.h>

static __constant__ float d_kernel[25];  // For Gaussian kernel

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < 2 || y >= (rows - 2) || x < 2 || x >= (cols - 2)) return;

    float sum = 0.0f;
    // Kernel is 5x5
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int px = x + kx;
            int py = y + ky;
            sum += input[py * cols + px] * d_kernel[(ky+2)*5 + (kx+2)];
        }
    }
    output[y * cols + x] = (unsigned char)sum;
}

__global__ void sobelKernel(const unsigned char* input, unsigned char* output, int rows, int cols) {
    int gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    int gy[9] = {-1,-2,-1,0,0,0,1,2,1};

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < 1 || y >= (rows - 1) || x < 1 || x >= (cols - 1)) return;

    int sumX = 0;
    int sumY = 0;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = x + kx;
            int py = y + ky;
            unsigned char val = input[py * cols + px];
            sumX += val * gx[(ky+1)*3 + (kx+1)];
            sumY += val * gy[(ky+1)*3 + (kx+1)];
        }
    }
    int magnitude = (int)sqrtf((float)(sumX*sumX + sumY*sumY));
    if(magnitude > 255) magnitude = 255;
    if(magnitude < 0) magnitude = 0;
    output[y * cols + x] = (unsigned char)magnitude;
}

int main() {
    cv::VideoCapture cap("sample.mp4");
    if(!cap.isOpened()) {
        std::cerr << "Error: Cannot open video.\n";
        return -1;
    }

    // Precompute Gaussian kernel on host
    float h_kernel[25] = {
        1,4,6,4,1,
        4,16,24,16,4,
        6,24,36,24,6,
        4,16,24,16,4,
        1,4,6,4,1
    };
    float sumKernel = 0.0f;
    for (int i=0; i<25; i++) sumKernel += h_kernel[i];
    for (int i=0; i<25; i++) h_kernel[i] /= sumKernel;

    cudaMemcpyToSymbol(d_kernel, h_kernel, 25*sizeof(float));

    cv::Mat frame;
    // Warm-up
    for (int i = 0; i < 10; i++)
        cap >> frame;

    int num_frames = 200;
    auto start = std::chrono::high_resolution_clock::now();

    // Assuming all frames same size
    cap >> frame;
    if(frame.empty()) {
        std::cerr << "No frames in video.\n";
        return -1;
    }

    cv::Mat gray, blurImg, edgeImg;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    int rows = gray.rows;
    int cols = gray.cols;

    blurImg = gray.clone();
    edgeImg = gray.clone();

    // Allocate GPU buffers
    unsigned char *d_in, *d_intermediate, *d_out;
    cudaMalloc(&d_in, rows*cols);
    cudaMalloc(&d_intermediate, rows*cols);
    cudaMalloc(&d_out, rows*cols);

    dim3 block(16,16);
    dim3 grid((cols+block.x-1)/block.x,(rows+block.y-1)/block.y);

    // Process first frame (already captured)
    for (int i=0; i<num_frames; i++) {
        if (i > 0) {
            cap >> frame;
            if (frame.empty()) break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Transfer to GPU
        cudaMemcpy(d_in, gray.data, rows*cols, cudaMemcpyHostToDevice);

        // Gaussian blur on GPU
        gaussianBlurKernel<<<grid, block>>>(d_in, d_intermediate, rows, cols);
        cudaDeviceSynchronize();

        // Sobel on GPU
        sobelKernel<<<grid, block>>>(d_intermediate, d_out, rows, cols);
        cudaDeviceSynchronize();

        // Transfer back
        cudaMemcpy(edgeImg.data, d_out, rows*cols, cudaMemcpyDeviceToHost);

        // Threshold on CPU
        cv::threshold(edgeImg, edgeImg, 50, 255, cv::THRESH_BINARY);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double fps = num_frames / (elapsedMs / 1000.0);
    std::cout << "CUDA: Processed " << num_frames << " frames in " << elapsedMs << " ms. Approx FPS: " << fps << "\n";

    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);

    return 0;
}
