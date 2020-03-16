#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstring>
#include <ctime>

#define WARP_SIZE 32 // # of threads that are executed together (constant valid on most hardware)


#change1
// Allow timing of functions
clock_t start,end;

/* Add "scalar" to every element of the input array in parallel */
__global__ void
_cuda_add_scalar(int *in, int scalar, int n)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while(globalIdx < n)
    {
        in[globalIdx] = in[globalIdx] + scalar;
        globalIdx += blockDim.x * gridDim.x;
    }
}

// CPU entry point for kernel to add "scalar" to every element of the input array
void cuda_add_scalar(int * a, int scalar, int N) {
    // Get device properties to compute optimal launch bounds
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_SMs = prop.multiProcessorCount;
    // pad array with zeros to allow sum algorithm to work
    int * d_a;
    start = std::clock();
    // Copy array "b" to GPU 
    cudaMalloc( (void**) &d_a, N * sizeof(int) );
    cudaMemcpy( d_a, a, N * sizeof(int), cudaMemcpyHostToDevice );
    _cuda_add_scalar<<< num_SMs, 1024 >>>(d_a, scalar, N);
    // Read in results
    cudaMemcpy( a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost );
    end = std::clock();
    //verify
    for(int i = 0; i < N-1; ++i) {
    	if(a[i] + 1.0 != a[i+1]){
    		printf("a[i]: %d, a[i+1]: %d\n", a[i], a[i+1]);
    		exit(-1);
    	}
    }
    // Clean up
    cudaFree(d_a);
}

/* Use all GPU Streaming Multiprocessors to add elements in parallel.
   Requies that the number of elements is a multiple of #SMs * 1024 
   since the algorithm processes elements in chunks of this size. 
   This is taken care of in "cuda_parallel_sum which pads zeros. */
__global__ void
_cuda_parallel_sum(int *in, int num_elements, int *sum)
{
    //Holds intermediates in shared memory reduction
    __syncthreads();
    __shared__ int buffer[WARP_SIZE];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int temp;
    while(globalIdx < num_elements)
    {
    	// All threads in a block of 1024 take an element
        temp = in[globalIdx];
        // All warps in this block (32) compute the sum of all
        // threads in their warp
        for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
        {
             temp+= __shfl_xor(temp, delta);
        }
        // Write all 32 of these partial sums to shared memory
        if(lane == 0)
        {
            buffer[threadIdx.x / WARP_SIZE] = temp;
        }
        __syncthreads();
        // Add the remaining 32 partial sums using a single warp
        if(threadIdx.x < WARP_SIZE) 
        {
            temp = buffer[threadIdx.x];
            for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
            {  
                temp += __shfl_xor(temp, delta);
            }
        }
        // Add this block's sum to the total sum
        if(threadIdx.x == 0)
        {
            atomicAdd(sum, temp);
        }
        // Jump ahead 1024 * #SMs to the next region of numbers to sum
        globalIdx += blockDim.x * gridDim.x;
        __syncthreads();
    }

}

/* CPU entry function to create intermediate variables and do array padding
   and then call CUDA kernel to add all elements in an array.
 */
int cuda_parallel_sum(int * a, int N) {
    // Get device properties to compute optimal launch bounds
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_SMs = prop.multiProcessorCount;
    start = std::clock();
    // pad array with zeros to allow sum algorithm to work
    int batch_size = num_SMs * 1024;
    int padding = (batch_size - (N % batch_size)) % batch_size;
    // b is the new padded array
    int * b = new int[N + padding];
    memcpy(b, a, N * sizeof(int));
    memset(b + N, 0, padding * sizeof(int));

    // Copy array "b" to GPU 
    int *d_b;
    cudaMalloc( (void**) &d_b, (N + padding) * sizeof(int) );
    cudaMemcpy( d_b, b, (N + padding) * sizeof(int), cudaMemcpyHostToDevice );

    // Result
    int result = 0.0;
    int * d_result;
    cudaMalloc( (void**) &d_result, sizeof(int) );
    cudaMemcpy( d_result, &result, sizeof(int), cudaMemcpyHostToDevice );

    // Call kernel to get sum
    _cuda_parallel_sum<<< num_SMs, 1024 >>>(d_b, N + padding, d_result);
    // Read in results
    cudaMemcpy( &result, d_result, sizeof(int), cudaMemcpyDeviceToHost );
    end = std::clock();

    // Clean up
    cudaFree(d_result);
    cudaFree(d_b);
    free(b);

    return result;
}


/* Get pertienent info about connected GPUs */
void query_GPUs(int * nDevices) {
    cudaGetDeviceCount(nDevices);
    for (int i = 0; i < *nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    }
}

// Prints an array horzontally up to size elements
void print_array(int * in, int size) {
    for(int i = 0; i < size; ++i) {
        printf("%d ", in[i]);
    }
    printf("\n");
}

int cpu_sum(int * a, int N) {
	int sum = 0;
	start = std::clock();
	for(int i = 0; i < N; ++i) {
        sum += a[i];
    }
    end = std::clock();
    return sum;
}

void cpu_add(int * a, int scalar, int N) {
	start = std::clock();
    // Add one to every element of a
    for(int i = 0; i < N; ++i) {
        a[i] += scalar;
    }
    end = std::clock();
}
int main() {
    
    const int N = 20;
    printf("specified N: %d\n\n", N);
    int nDevices;
    // Query GPUs
    query_GPUs(&nDevices);
    // Select first GPU
    if(nDevices != 0) {
        cudaSetDevice(0);
    }
    else {
        printf("No CUDA devices found!\n");
        exit(-1);
    }

    // Declare CPU array
    int * a = new int[N];
    for(int i = 0; i < N; i++) {
        a[i] = i;
    }

    /* Perform GPU operations
    ___________________________ */
    printf("\n\n Computations using GPU\n\n");
    // If GPU Boost is enabled, the GPU might be in "idle" mode
    // so we run a kernel to get the clock speed to baseline levels.
    int warmup_the_gpu_sum = cuda_parallel_sum(a, N);
    // print sum before kernel launches
    int gpu_sum1 = cuda_parallel_sum(a, N);
    printf("sum: %d - total time: %ld\n", gpu_sum1, end - start);

    // Add one to every element of a
    cuda_add_scalar(a, 1.0, N);
    printf("Total time to add 1 to every element: %ld\n", end - start);

    //Print first 10 ints in resulting array
    print_array(a, 20);

    // Print new sum
    int gpu_sum2 = cuda_parallel_sum(a, N);
    printf("sum: %d - total time: %ld\n", gpu_sum2, end - start);
    

    /* Perform same operations on CPU
    ___________________________ */
    // Reset vector a
    for(int i = 0; i < N; ++i) {
        a[i] = i;
    }
    printf("\n\n Same Computations using CPU\n\n");
    int cpu_sum1 = cpu_sum(a, N);
    printf("sum: %d - total time: %ld\n", cpu_sum1, end - start);

    cpu_add(a, 1.0, N);
    printf("Total time to add 1 to every element: %ld\n", end - start);

    //Print first 10 ints in resulting array
    print_array(a, 20);

    // Print new sum
    int cpu_sum2 = cpu_sum(a, N);
    printf("sum: %d - total time: %ld\n", cpu_sum2, end - start);

    printf("\n\nGPU consistency test: %d\n ... %s\n", warmup_the_gpu_sum - gpu_sum1,
    	(warmup_the_gpu_sum - gpu_sum1 == 0.0 ? "SUCCESS" : "FAIL"));
    printf("Difference in first sums: %d\n ... %s\n", gpu_sum1 - cpu_sum1,
    	(gpu_sum1 - cpu_sum1 == 0 ? "SUCCESS" : "FAIL"));
    printf("Difference in second sums: %d\n ... %s\n", gpu_sum2 - cpu_sum2,
    (gpu_sum2 - cpu_sum2 == 0 ? "SUCCESS" : "FAIL"));

    // Clean up
    delete(a);
    return 0;
}
