#include <cstdio>
#include <ctime>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <chrono>
#include "cuda_runtime_api.h"

#define N 1
#define R 1
#define MAX 20

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
                blockIdx.y, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x]);
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, uint8_t* numbers) {
    /* curand works like rand - except that it takes a state as a parameter */
    uint32_t randInt = curand(&states[blockIdx.x]);
    numbers[4 * blockIdx.x]     = ((randInt & 0xFF000000UL) >> 24)  % MAX + 1;
    numbers[4 * blockIdx.x + 1] = ((randInt & 0x00FF0000UL) >> 16)  % MAX + 1;
    numbers[4 * blockIdx.x + 2] = ((randInt & 0x0000FF00UL) >> 8 )  % MAX + 1;
    numbers[4 * blockIdx.x + 3] = ((randInt & 0x000000FFUL)      )  % MAX + 1;
}

/* this GPU kernel takes an array of ints and adds 1 to the passcounter if they're greater than or equal to the given int */
__global__ void passcheck(unsigned long long int* passcounter, int8_t* numberstopass, const uint8_t* numbers) {
    if (numbers[blockIdx.x * blockDim.x + threadIdx.x] >= numberstopass[0]) { atomicAdd(passcounter, 1); }
}

void printLoadingBar(unsigned long long int rolled, unsigned long long int counterStop, double start_time ) {
    printf("Rolled: %lld%% ", (rolled * 100)/counterStop);
    auto end = std::chrono::system_clock::now().time_since_epoch().count();
    double diff = end - start_time;
    printf(": %i rolls per second \n", (int)(((double)rolled * 10000000) / diff));
}

// you must first call the cudaGetDeviceProperties() function, then pass
// the devProp structure returned to this function:
int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

int main() {
    cudaDeviceProp cudaDeviceProp;
    cudaGetDeviceProperties(&cudaDeviceProp, 0);
    int nrCores = getSPcores(cudaDeviceProp);
    std::cout << "ShaderCores: " << nrCores << "\n";

    // After how many rolls should you stop
    unsigned long long int counter = 0;
    unsigned long long int counterstop = (unsigned long long int)(INT32_MAX/512) * nrCores;

    // Cuda performance metrics
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

//    /* allocate an array of int8_t on the CPU and GPU */
//    uint8_t cpu_nums[ 4];
    uint8_t* gpu_nums;
    cudaMalloc((void **) &gpu_nums, nrCores * 4 * sizeof(uint8_t));

    /* allocate an array of int8_t on the CPU and GPU */
    unsigned long long int cpu_pass_counter[1];
    cpu_pass_counter[0] = 0;
    unsigned long long int* gpu_pass_counter;
    cudaMalloc((void **) &gpu_pass_counter, 1 * sizeof(unsigned long long int));
    cudaMemcpy(gpu_pass_counter, cpu_pass_counter, 1 * sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    /* allocate an array of int8_t on the CPU and GPU of numbers that should be checked against */
    int8_t cpu_num_to_pass[R];
    cpu_num_to_pass[0] = 20;
    int8_t* gpu_num_to_roll;
    cudaMalloc((void **) &gpu_num_to_roll, R * sizeof(int8_t));
    cudaMemcpy(gpu_num_to_roll, cpu_num_to_pass, R * sizeof(int8_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    /* allocate space on the GPU for the random states */
    curandState_t* states;
    cudaMalloc((void **) &states, nrCores * sizeof(curandState_t));

    /* invoke the GPU to initialize all of the random states */
    init<<<nrCores, 1>>>(time(nullptr), states);

    auto start_timer = std::chrono::system_clock::now();
    printLoadingBar(counter, counterstop, start_timer.time_since_epoch().count());
    cudaEventRecord(start);
    while (counter < counterstop) {
        /* invoke the kernel to get some random numbers */
        randoms<<<nrCores, 1>>>(states, gpu_nums);
        passcheck<<<nrCores, 4>>>(gpu_pass_counter, gpu_num_to_roll, gpu_nums);

        /* copy the random numbers back */
//        cudaMemcpy(cpu_nums, gpu_nums, nrCores * 4 * sizeof(int8_t), cudaMemcpyDeviceToHost);
//        cudaMemcpy(cpu_pass_counter, gpu_pass_counter, 1 * sizeof(int64_t), cudaMemcpyDeviceToHost);

        counter += nrCores*4;
        if ((counter % (nrCores * 4 * 10000)) == 0) {
            printLoadingBar(counter, counterstop, start_timer.time_since_epoch().count());
        }
    }
    cudaEventRecord(stop);
    printLoadingBar(counter, counterstop, start_timer.time_since_epoch().count());
    cudaMemcpy(cpu_pass_counter, gpu_pass_counter, 1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

    /* free memory from GPU */
    cudaFree(states);
    cudaFree(gpu_nums);
    cudaFree(gpu_pass_counter);
    cudaFree(gpu_num_to_roll);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Ran %lld simulations resulting in %lld passes taking %fs \n", counter, cpu_pass_counter[0], milliseconds/1000);
    printf("Averaged: %i rolls per second", (int)(counter/(milliseconds/1000)));

    return 0;
}

