#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include<ctime>

using namespace std;

#define BLOCK_SIZE  1024// ����� ���������  
#define N           1024// ����� �������



//�������� �������
__global__ void MatrixMul(float * a, float * b, int n, float * c)
{
	int   bx = blockIdx.x;     // block index
	int   by = blockIdx.y;
	int   tx = threadIdx.x;        // thread index
	int   ty = threadIdx.y;
	float sum = 0.0f;           // �����
	int   ia = n * BLOCK_SIZE * by + n * ty;   // a [i][0]
	int   ib = BLOCK_SIZE * bx + tx;

	// ��������
	for (int k = 0; k < n; k++)
		sum += a[ia + k] * b[ib + k*n];

	// �������� � ���� ���������
	int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	c[ic + n * ty + tx] = sum;
}


int main(int argc, char *  argv[])
{


	int numBytes = N * N * sizeof(float);

	// �������� �����
	float * a = new float[N*N];
	float * b = new float[N*N];
	float * c = new float[N*N];

	// ���������� �������
	for (int i = 0; i < N*N; i++)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}


	// �������� �� ������ ����� GPU
	float * adev = NULL;
	float * bdev = NULL;
	float * cdev = NULL;

	//�������� ���'��� ��� ������ �� ��������
	cudaMalloc((void**)&adev, numBytes);
	cudaMalloc((void**)&bdev, numBytes);
	cudaMalloc((void**)&cdev, numBytes);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(N / threads.x, N / threads.y);


	cudaEvent_t start, stop;
	float gpuTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// ������� �����
	cudaEventRecord(start, 0);

	// ������� ��� � ���'��� ���������
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);
	//����������

	// �������� �������
	MatrixMul << <blocks, threads >> > (adev, bdev, N, cdev);


	// ��������� ���������� � ����� �
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	// ����� �����
	cudaEventRecord(stop, 0);


	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);

	// ���������� �� ��������
	// ������� �����
	double begin = clock();

	for (int i = 0; i<N; i++) {
		for (int l = 0; l<N; l++) {
			float s = 0;
			for (int j = 0; j<N; j++)
				s += a[i*N + j] * b[j*N + l];

			c[i*N + l] = s;
		}
	}

	//����� �����
	double end = clock();
	double cpuTime = double(end - begin) / CLOCKS_PER_SEC*1000.0;
	printf("time spent executing by the CPU: %.10f milliseconds\n", cpuTime);


	////��������� �������
	//for (int i = 0; i < N*N; i++)
	//{

	//	if (i % N == 0)
	//		cout << "\n";
	//	cout << c[i] << " ";
	//}

	// ��������� �����
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);

	delete a;
	delete b;
	delete c;
	system("pause");
	return 0;
}