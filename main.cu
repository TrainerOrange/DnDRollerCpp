#include <cstdio>
#include <ctime>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>

#define N 8704 * 2048
#define R 2
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
__global__ void randoms(curandState_t* states, int8_t* numbers) {
    /* curand works like rand - except that it takes a state as a parameter */
    int32_t randInt = curand(&states[blockIdx.x]);
    numbers[4 * blockIdx.x]     = ((randInt & 0xFF000000UL) >> 24)  % MAX + 1;
    numbers[4 * blockIdx.x + 1] = ((randInt & 0x00FF0000UL) >> 16)  % MAX + 1;
    numbers[4 * blockIdx.x + 2] = ((randInt & 0x0000FF00UL) >> 8 )  % MAX + 1;
    numbers[4 * blockIdx.x + 3] = ((randInt & 0x000000FFUL)      )  % MAX + 1;
}

/* this GPU kernel takes an array of ints and adds 1 to the passcounter if they're greater than or equal to the given int */
__global__ void passcheck(int64_t* passcounter, const int8_t* numberstopass, const int8_t* numbers) {
    if (numbers[blockIdx.x] == numberstopass[0] || numbers[blockIdx.x] == numberstopass[1])
        {
            passcounter[0] += 1;
//            printf("Number: %i \n", numbers[blockIdx.x]);
//            printf("counted! \n");
        }
}

int main() {
    int64_t counter = 0;
    auto start_time = clock();

    // After how many rolls should you stop
    int64_t counterstop = ((int64_t) LONG_MAX) * 4;

//    /* allocate an array of int8_t on the CPU and GPU */
//    int8_t cpu_nums[N * 4];
    int8_t* gpu_nums;
    cudaMalloc((void **) &gpu_nums, N * 4 * sizeof(int8_t));

    /* allocate an array of int8_t on the CPU and GPU */
    int64_t cpu_pass_counter[1];
    cpu_pass_counter[0] = 0;
    int64_t* gpu_pass_counter;
    cudaMalloc((void **) &gpu_pass_counter, 1 * sizeof(int64_t));
    cudaMemcpy(gpu_pass_counter, cpu_pass_counter, 1 * sizeof(int64_t), cudaMemcpyHostToDevice);

    /* allocate an array of int8_t on the CPU and GPU */
    int8_t cpu_num_to_roll[R];
    cpu_num_to_roll[0] = 19;
    cpu_num_to_roll[1] = 20;
    int8_t* gpu_num_to_roll;
    cudaMalloc((void **) &gpu_num_to_roll, R * sizeof(int8_t));
    cudaMemcpy(gpu_num_to_roll, cpu_num_to_roll, R * sizeof(int8_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    /* allocate space on the GPU for the random states */
    curandState_t* states;
    cudaMalloc((void **) &states, N * sizeof(curandState_t));

    while (counter < counterstop) {
        /* invoke the GPU to initialize all of the random states */
        init<<<N, 1>>>(counter, states);

        /* invoke the kernel to get some random numbers */
        randoms<<<N, 1>>>(states, gpu_nums);

        passcheck<<<N * 4, 1>>>(gpu_pass_counter, gpu_num_to_roll, gpu_nums);

        /* copy the random numbers back */
        // cudaMemcpy(cpu_nums, gpu_nums, N * 4 * sizeof(int8_t), cudaMemcpyDeviceToHost);

        counter += N*4;
    }
    cudaMemcpy(cpu_pass_counter, gpu_pass_counter, 1 * sizeof(int64_t), cudaMemcpyDeviceToHost);

    /* free memory from GPU */
    cudaFree(states);
    cudaFree(gpu_nums);
    cudaFree(gpu_pass_counter);
    cudaFree(gpu_num_to_roll);

    auto end_time = clock();
    printf("Ran %lld simulations resulting in %lld crits taking %fs", counter, cpu_pass_counter[0], (float)(end_time - start_time)/1000);

    return 0;
}