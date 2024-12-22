import cv2
import numpy as np
import math
from numba import cuda, jit

# Define Gaussian kernel on host
h_kernel = np.array([1,4,6,4,1,
                     4,16,24,16,4,
                     6,24,36,24,6,
                     4,16,24,16,4,
                     1,4,6,4,1], dtype=np.float32)
sumKernel = h_kernel.sum()
h_kernel /= sumKernel

d_kernel = cuda.to_device(h_kernel)

@cuda.jit
def gaussianBlurKernel(input_image, output_image, rows, cols, kernel):
    x, y = cuda.grid(2)
    if y < 2 or y >= rows-2 or x < 2 or x >= cols-2:
        return
    sum_val = 0.0
    for ky in range(-2,3):
        for kx in range(-2,3):
            px = x + kx
            py = y + ky
            sum_val += input_image[py, px]*kernel[(ky+2)*5 + (kx+2)]
    output_image[y,x] = sum_val

@cuda.jit
def sobelKernel(input_image, output_image, rows, cols):
    gx = (-1,0,1,-2,0,2,-1,0,1)
    gy = (-1,-2,-1,0,0,0,1,2,1)
    x, y = cuda.grid(2)
    if y < 1 or y >= rows-1 or x < 1 or x >= cols-1:
        return

    sumX = 0
    sumY = 0
    idx = 0
    for ky in range(-1,2):
        for kx in range(-1,2):
            px = x + kx
            py = y + ky
            val = input_image[py, px]
            sumX += val * gx[idx]
            sumY += val * gy[idx]
            idx += 1
    magnitude = int(math.sqrt(sumX*sumX + sumY*sumY))
    if magnitude > 255: magnitude = 255
    if magnitude < 0: magnitude = 0
    output_image[y,x] = magnitude

cap = cv2.VideoCapture("sample.mp4")
if not cap.isOpened():
    print("Error: Cannot open video.")
else:
    # Warm-up
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break

    num_frames = 200
    ret, frame = cap.read()
    if not ret:
        print("No frames.")
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        blurImg = np.zeros_like(gray, dtype=np.uint8)
        edgeImg = np.zeros_like(gray, dtype=np.uint8)

        # Prepare GPU
        threadsperblock = (16,16)
        blockspergrid_x = (cols + threadsperblock[0] - 1)//threadsperblock[0]
        blockspergrid_y = (rows + threadsperblock[1] - 1)//threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        start = cv2.getTickCount()

        for i in range(num_frames):
            if i > 0:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            d_in = cuda.to_device(gray)
            d_mid = cuda.to_device(np.zeros_like(gray, dtype=np.float32))
            d_out = cuda.to_device(np.zeros_like(gray, dtype=np.float32))

            gaussianBlurKernel[blockspergrid, threadsperblock](d_in, d_mid, rows, cols, d_kernel)
            cuda.synchronize()
            sobelKernel[blockspergrid, threadsperblock](d_mid, d_out, rows, cols)
            cuda.synchronize()

            edge_data = d_out.copy_to_host().astype(np.uint8)
            _, edge_data = cv2.threshold(edge_data, 50, 255, cv2.THRESH_BINARY)
            # edge_data is your processed frame result

        end = cv2.getTickCount()
        elapsed = (end - start)/cv2.getTickFrequency()*1000.0
        fps = num_frames / (elapsed/1000.0)
        print(f"Numba CUDA: Processed {num_frames} frames in {elapsed:.2f} ms. Approx FPS: {fps:.2f}")
