#include<stdio.h>
#define N (2048*2048)
#define THREADS_PER_BLOCK 512
__global__ void dot( int *a, int *b, int *c ) {
__shared__ int temp[THREADS_PER_BLOCK];
int index = threadIdx.x + blockIdx.x * blockDim.x;
temp[threadIdx.x] = a[index] * b[index];
__syncthreads();
if( 0 == threadIdx.x ) {
int sum = 0;
for( int i = 0; i < THREADS_PER_BLOCK; i++ )
sum += temp[i];
atomicAdd( c , sum );
}
}

int main( void ) {
int *a, *b, *c,i; 
int *dev_a, *dev_b, *dev_c;
int size = N * sizeof( int ); 
cudaMalloc( (void**)&dev_a, size );
cudaMalloc( (void**)&dev_b, size );
cudaMalloc( (void**)&dev_c, sizeof( int ) );
a = (int *)malloc( size );
b = (int *)malloc( size );
c = (int *)malloc( sizeof( int ) );
int sumtest= 0;
for(i=0;i<N;i++)
   {
       a[i] = rand() % 10;
       b[i] = rand() % 10;
       sumtest += a[i]*b[i];
   }
cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );
dot<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
cudaMemcpy( c, dev_c, sizeof( int ) , cudaMemcpyDeviceToHost );
printf(" Using CPU:- %d\n ",sumtest);
printf("Using GPU:- %d",*c);
free( a );
free( b );
free( c );
cudaFree( dev_a );
cudaFree( dev_b );
cudaFree( dev_c );
return 0;
}

