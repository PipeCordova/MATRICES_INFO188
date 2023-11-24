#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

const int BS = 16;  // Tamaño del bloque en la memoria compartida
// Kernel

__global__ void mikernel(float *A, float *B, float *C, int n){
    int tidx = (blockDim.x * blockIdx.x) + threadIdx.x;
    int tidy = (blockDim.y * blockIdx.y) + threadIdx.y;
    float acc = 0.0;
    for(int k=0; k < n; k++){
        acc += A[tidy * n + k] * B[k * n + tidx];
    }
    C[tidy * n + tidx] = acc;
}

__global__ void matrixMulShared(float *A, float *B, float *C, int n) {
    // Definir la memoria compartida
    __shared__ float sA[BS][BS];
    __shared__ float sB[BS][BS];

    int row = blockIdx.y * BS + threadIdx.y;
    int col = blockIdx.x * BS + threadIdx.x;

    float acc = 0.0;

    // Calcular el número de bloques necesarios
    int numBlocks = (n + BS - 1) / BS;

    for (int b = 0; b < numBlocks; ++b) {
        // Cargar bloques de A y B en memoria compartida
        if (row < n && (b * BS + threadIdx.x) < n) {
            sA[threadIdx.y][threadIdx.x] = A[row * n + b * BS + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < n && (b * BS + threadIdx.y) < n) {
            sB[threadIdx.y][threadIdx.x] = B[(b * BS + threadIdx.y) * n + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Sincronizar para asegurar que los bloques se han cargado completamente en memoria compartida
        __syncthreads();

        // Calcular producto parcial en el bloque
        for (int k = 0; k < BS; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Sincronizar antes de cargar el siguiente bloque
        __syncthreads();
    }

    // Escribir el resultado en la matriz de salida
    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

void init_mat(float *M, int n, float c){
	long nelem = (long)n*(long)n;
	#pragma omp parallel for 
	//printf("nelem = %lu\n", nelem);
	for(long i=0; i<nelem; ++i){
		//printf("it %lu \n", i); fflush(stdout);
		M[i] = (((int)c*i) % 5)/100.0f;
	}
}

void print_mat(float *M, int n, const char *msg){
	if(n > 32){ return; }
	for(int i=0;i<n;i++){
		for(int j=0; j<n; j++){
            long index = i*n + j;
			//printf("%.3f ", M[index]);
		}
		//printf("\n");
	}
}

void cpu(float *A, float *B, float *C, int n){
	#pragma omp parallel for 
	for(int i=0;i<n;i++){
		for(int j=0; j<n; j++){
            long index = i*n + j;
			float acc = 0.0f;
			for(long k=0; k<n; k++){
				acc += A[(long)i*n + k]*B[k*n +j];	
			}
			C[index] = acc;
		}
	}
}

int main(int argc, char **argv) {
    if(argc != 5){
		fprintf(stderr, "error ejecutar como ./prog nt n mode BSIZE\n");
		exit(EXIT_FAILURE);
	}
    int nt, n, m, bs;
    float *A,  *B,  *C;
    float *dA, *dB, *dC;
    // obtener argumentos
    nt = atoi(argv[1]); // numero de threads (en caso del modo CPU)
    n = atoi(argv[2]);  // lado de una matrix de n x n
    m = atoi(argv[3]);  // modo (0 -> CPU   1 -> GPU)
    bs = atoi(argv[4]); // lado de un bloque de bs x bs

    long nelem = (long)n * (long)n;
    printf("nt = %i   n = %i  m = %i  bs = %i\n", nt, n, m, bs);
    printf("elementos matrix 	= %lu elementos\n", nelem);
    printf("memoria usada 	= %f GBs\n", sizeof(float)*nelem/1e9);
    omp_set_num_threads(nt);

    // inicializar arreglos en Host (CPU)
    double t1 = omp_get_wtime();
    printf("inicializando...."); fflush(stdout);
    A = new float[nelem]; B = new float[nelem]; C = new float[nelem];
    init_mat(A, n, 1); print_mat(A, n, "MATRIX A"); 
    init_mat(B, n, 2); print_mat(B, n, "MATRIX B");
    init_mat(C, n, 0);
    double t2 = omp_get_wtime();
    printf("ok: %f secs\n", t2-t1); fflush(stdout);

    // allocar memoria en device  (GPU)
    cudaMalloc(&dA, sizeof(float) * nelem);
    cudaMalloc(&dB, sizeof(float) * nelem);
    cudaMalloc(&dC, sizeof(float) * nelem);

    // copiar de Host -> Device
    cudaMemcpy(dA, A, sizeof(float)*nelem, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float)*nelem, cudaMemcpyHostToDevice);

    dim3 block(bs, bs, 1);
    dim3 grid((n + bs -1) / bs, (n + bs -1) / bs, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    printf("calculando...."); fflush(stdout);
    cudaEventRecord(start);
    if (m) {
        printf("GPU\n"); fflush(stdout);
        // Utilizar el nuevo kernel con memoria compartida
        matrixMulShared<<<grid, block>>>(dA, dB, dC, n);
    } else {
        printf("CPU\n"); fflush(stdout);
        cpu(A, B, C, n);
    }
    cudaDeviceSynchronize(); cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    if (m) { cudaMemcpy(C, dC, sizeof(float) * nelem, cudaMemcpyDeviceToHost); }
    printf("ok: %f secs\n", milliseconds / 1000.0f); fflush(stdout);
    print_mat(C, n, "MATRIX C");
}
