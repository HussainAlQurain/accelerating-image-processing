#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cmath>

// CPU-only Gaussian blur
void gaussianBlurCPU(const cv::Mat &input, cv::Mat &output) {
    float kernel[5][5] = {
        {1,4,6,4,1},
        {4,16,24,16,4},
        {6,24,36,24,6},
        {4,16,24,16,4},
        {1,4,6,4,1}
    };

    float sumKernel = 0.0f;
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            sumKernel += kernel[i][j];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            kernel[i][j] /= sumKernel;

    for (int y = 2; y < input.rows - 2; y++) {
        for (int x = 2; x < input.cols - 2; x++) {
            float sum = 0.0f;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    sum += input.at<uchar>(py, px)*kernel[ky+2][kx+2];
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(sum);
        }
    }
}

// CPU-only Sobel edge detection
void sobelEdgeCPU(const cv::Mat &input, cv::Mat &output) {
    int gx[3][3] = {
        {-1,0,1},
        {-2,0,2},
        {-1,0,1}
    };
    int gy[3][3] = {
        {-1,-2,-1},
        {0,0,0},
        {1,2,1}
    };

    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int sumX = 0;
            int sumY = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    uchar val = input.at<uchar>(py, px);
                    sumX += val * gx[ky+1][kx+1];
                    sumY += val * gy[ky+1][kx+1];
                }
            }
            int magnitude = std::min(255, std::max(0,(int)std::sqrt(sumX*sumX + sumY*sumY)));
            output.at<uchar>(y,x) = (uchar)magnitude;
        }
    }
}

int main() {
    cv::VideoCapture cap("sample.mp4");
    if(!cap.isOpened()) {
        std::cerr << "Error: Cannot open video.\n";
        return -1;
    }

    cv::Mat frame, gray, blurImg, edgeImg;
    // Warm-up
    for (int i = 0; i < 10; i++)
        cap >> frame;

    int num_frames = 200;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_frames; i++) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        blurImg = gray.clone();
        gaussianBlurCPU(gray, blurImg);

        edgeImg = blurImg.clone();
        sobelEdgeCPU(blurImg, edgeImg);
        cv::threshold(edgeImg, edgeImg, 50, 255, cv::THRESH_BINARY);

        // No imshow, no waitKey here
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double fps = num_frames / (elapsedMs / 1000.0);

    std::cout << "CPU-only: Processed " << num_frames << " frames in " << elapsedMs << " ms. Approx FPS: " << fps << "\n";

    return 0;
}
