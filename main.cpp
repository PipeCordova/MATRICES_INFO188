#include <iostream>
#include <omp.h>
#include <cstdlib> // para rand
#include <ctime> // para time

using namespace std;

void llenarMatriz(int n, int** matriz);
void multiplicar_matrices_bloques(int n, int nt, int** A, int** B, int** C, int tam_bloque);
void imprimirMatriz(int n, int ** matriz);


int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Uso: ./prog <n> <nt> <modo>" << endl;
        return EXIT_FAILURE;
    }

    int n = stoi(argv[1]);
    int nt = stoi(argv[2]);
    string modo = argv[3];

    // inicializar las matrices A, B y C
    int** A = new int*[n];
    int** B = new int*[n];
    int** C = new int*[n];

    for (int i = 0; i < n; i++) {
        A[i] = new int[n];
        B[i] = new int[n];
        C[i] = new int[n];
    }
    
    llenarMatriz(n, A);
    //cout << "Matriz A:" << endl;
    //imprimirMatriz(n, A);

    llenarMatriz(n, B);
    //cout << "Matriz B:" << endl;
    //imprimirMatriz(n, B);

    if (modo == "normal") {
        // multiplicacion de matrices sin bloque
        #pragma omp parallel for num_threads(nt)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    } else { // si no es normal es por bloque
        int tam_bloque = 32; // Tamaño del bloque, puedes ajustarlo según tus necesidades
        multiplicar_matrices_bloques(n, nt, A, B, C, tam_bloque);
    }

    //cout << "Matriz C:" << endl;
    //imprimirMatriz(n, C);

    // liberando la memoria
    for (int i = 0; i < n; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;

    return EXIT_SUCCESS;
}

/*
Se llena la matriz con numeros aleatorios entre 1 y 99
*/
void llenarMatriz(int n, int** matriz) {
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matriz[i][j] = rand() % 99 + 1;
        }
    }
}

void imprimirMatriz(int n, int ** matriz) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matriz[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void multiplicar_matrices_bloques(int n, int nt, int** A, int** B, int** C, int tam_bloque) {
    int i, j, k, ii, jj, kk;

    // utilizamos OpenMP para paralelizar la multiplicación de matrices por bloques
    #pragma omp parallel for private(i, j, k, ii, jj, kk) num_threads(nt)
    for (ii = 0; ii < n; ii += tam_bloque) {
        for (jj = 0; jj < n; jj += tam_bloque) {
            for (kk = 0; kk < n; kk += tam_bloque) {
                for (i = ii; i < min(ii + tam_bloque, n); i++) {
                    for (j = jj; j < min(jj + tam_bloque, n); j++) {
                        for (k = kk; k < min(kk + tam_bloque, n); k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

