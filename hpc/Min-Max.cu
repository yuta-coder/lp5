// reduction.cu
#include <iostream>
#include <cuda.h>
#include <limits.h>

using namespace std;

#define N 1024
#define THREADS 256

// 🔹 Kernel for reduction
__global__ void reduction(int *arr, int *sum, int *min_val, int *max_val) {
    __shared__ int sdata[THREADS];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = arr[i];
    __syncthreads();

    // 🔹 Reduction inside block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Sum
            sdata[tid] += sdata[tid + s];

            // Min
            if (sdata[tid + s] < sdata[tid])
                sdata[tid] = sdata[tid + s];

            // Max
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Store results from each block
    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
        atomicMin(min_val, sdata[0]);
        atomicMax(max_val, sdata[0]);
    }
}

int main() {
    int arr[N];

    // Initialize array
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000;
    }

    int *d_arr, *d_sum, *d_min, *d_max;

    int sum = 0;
    int min_val = INT_MAX;
    int max_val = INT_MIN;

    // Allocate device memory
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_sum, sizeof(int));
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min, &min_val, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &max_val, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = N / THREADS;
    reduction<<<blocks, THREADS>>>(d_arr, d_sum, d_min, d_max);

    // Copy results back
    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_val, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_val, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    double avg = (double)sum / N;

    cout << "Sum = " << sum << endl;
    cout << "Min = " << min_val << endl;
    cout << "Max = " << max_val << endl;
    cout << "Average = " << avg << endl;

    // Free memory
    cudaFree(d_arr);
    cudaFree(d_sum);
    cudaFree(d_min);
    cudaFree(d_max);

    return 0;
}



// how to run-
// nvcc min-max.cu -o min-max
// .\min-max.exe 





