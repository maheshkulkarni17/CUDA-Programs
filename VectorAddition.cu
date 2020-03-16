#include<stdio.h>
#include<time.h>
#define N (64*64)
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c ) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
c[index] = a[index] + b[index];
}

int main( void ) {
int *a, *b, *c;
int *dev_a, *dev_b, *dev_c;
int size = N * sizeof( int ); 
cudaMalloc( (void**)&dev_a, size );
cudaMalloc( (void**)&dev_b, size );
cudaMalloc( (void**)&dev_c, size );
a = (int*)malloc( size );
b = (int*)malloc( size );
c = (int*)malloc( size );

for(int i=0;i<N;i++)
{
a[i]=rand()%10;
b[i]=rand()%10;
}
cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice );
cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice );
add<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c );
cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost );

for(int i=0;i<N;i++)
{
  printf("%d ",c[i]);
}

free( a ); free( b ); free( c );
cudaFree(dev_a );
cudaFree(dev_b );
cudaFree(dev_c );
return 0;
}

