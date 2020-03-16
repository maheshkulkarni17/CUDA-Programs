#define N 3
#include <stdio.h>

__global__ void matrixMult (int *a, int *b, int *c, int width)
{
int i, sum = 0;
int col = threadIdx.x + blockDim.x * blockIdx.x;
int row = threadIdx.y + blockDim.y * blockIdx.y;
if(col < width && row < width)
for (i = 0; i< width; i++)
{
sum += a[row * width + i] * b[i * width + col];
}
c[row * width + col] = sum;
}

int main()
{
int a[N][N], b[N][N], c[N][N];
int *dev_a, *dev_b, *dev_c;
 int i=1;
for (int y = 0; y < N; y++)
                          {
for (int x = 0; x < N; x++)
a[y][x]=i++;
                          }
i=9;
   for (int y = 0; y < N; y++)
        {
       for (int x = 0; x < N; x++)
                b[y][x]=i--;
         }

int size = N * N * sizeof(int);
cudaMalloc((void **) &dev_a, size);
cudaMalloc((void **) &dev_b, size);
cudaMalloc((void **) &dev_c, size);
cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
dim3 dimGrid(1, 1);
dim3 dimBlock(N, N);
matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);
for (int y = 0; y < N; y++) {
for (int x = 0; x < N; x++) {
printf("%d \t", c[y][x]);
}
printf("\n");
}
return 0;

}
