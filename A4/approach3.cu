#include <iostream>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

// Function to read a sparse matrix stored in a binary file
void readMatrix(string& fileName, int& n, int& m, unsigned short *blocksMatrix, unsigned short *indexMatrix) {
    int k = 0;
    ifstream file(fileName, ios::binary);
    file.read(reinterpret_cast<char*>(&n), 4);
    file.read(reinterpret_cast<char*>(&m), 4);
    file.read(reinterpret_cast<char*>(&k), 4);

    int size = m * m;
    int rowIndex = 0, colIndex = 0;

    // Loop over all blocks in the file
    for (int j = 0; j < k; j++) {
        file.read(reinterpret_cast<char*>(&rowIndex), 4);
        file.read(reinterpret_cast<char*>(&colIndex), 4);
        indexMatrix[rowIndex * n / m + colIndex] = j + 1;
        file.read(reinterpret_cast<char*>(&blocksMatrix[j*size]), 2 * size);
    }
    file.close();
}

void writeMatrix(string file_name, int n, int m, unsigned int *c) {
    unsigned int MAX_VAL = 4294967295;
    ofstream file(file_name, ios::binary); int non_zero = 0;
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&non_zero), 4);
    for(int i = 0; i < n/m; i++){
        for(int j = 0; j < n/m; j++){
            int checksum = 0;
            for(int k = 0; k < m*m; k++){
                int relative_row = k / m, relative_col = k % m;
                int actual_row = relative_row + i*m;
                int actual_col = relative_col + j*m;
                int actual_index = actual_row * n + actual_col;
                checksum += c[actual_index];
                if(checksum > 0) break;
            }
            if(checksum){
                non_zero++;
                file.write(reinterpret_cast<const char*>(&i), 4);
                file.write(reinterpret_cast<const char*>(&j), 4);
                int data;
                for(int k = 0; k < m*m; k++){
                    int relative_row = k / m, relative_col = k % m;
                    int actual_row = relative_row + i*m;
                    int actual_col = relative_col + j*m;
                    int actual_index = actual_row * n + actual_col;
                    data = min(c[actual_index], MAX_VAL);
                    file.write(reinterpret_cast<const char*>(&data), 4);
                }
            }
        }
    }
    file.close();
    fstream update_k(file_name, ios::binary | ios::in | ios::out);
    update_k.write(reinterpret_cast<const char*>(&n), 4);
    update_k.write(reinterpret_cast<const char*>(&m), 4);
    update_k.write(reinterpret_cast<const char*>(&non_zero), 4);
    update_k.close();
}

__global__ void matrix_multiply_kernel(unsigned short *a, unsigned short *b, unsigned short *index_a, unsigned short *index_b, unsigned int *c, int n, int m) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * blockDim.y + threadIdx.y;
    int col = bx * blockDim.x + threadIdx.x;
    int base_a, base_b;
    unsigned int sum = 0;
    for (int i = 0; i < n/m; i++) {
        base_a = index_a[by * n/m + i] - 1; if(base_a == -1) continue;
        base_b = index_b[i * n/m + bx] - 1; if(base_b == -1) continue;
        for (int j = 0; j < m; j++) {
            sum += a[base_a*m*m + ty * m + j] * b[base_b*m*m + j * m + tx];
        }
    }
    c[row * n + col] = sum;
}

void multiplyMatrixWrapper(string input_matrix_file_1, string input_matrix_file_2, string output_matrix_file) {
    int n = 0, m = 0, k1 = 0, k2 = 0;
    cudaError_t err;
    auto initial_time = chrono::high_resolution_clock::now();
    auto start_time = chrono::high_resolution_clock::now(), end_time = chrono::high_resolution_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    // Read input matrix from file and store blocks in a vector of vectors
    ifstream file1(input_matrix_file_1, ios::binary);
    file1.read(reinterpret_cast<char*>(&n), 4);
    file1.read(reinterpret_cast<char*>(&m), 4);
    file1.read(reinterpret_cast<char*>(&k1), 4);
    file1.close();
    ifstream file2(input_matrix_file_2, ios::binary);
    file2.read(reinterpret_cast<char*>(&n), 4);
    file2.read(reinterpret_cast<char*>(&m), 4);
    file2.read(reinterpret_cast<char*>(&k2), 4);
    file2.close();

    unsigned short *h_a, *h_b, *index_a, *index_b; unsigned int *h_c;
    h_a = (unsigned short*) calloc(k1 * m * m, sizeof(unsigned short));
    h_b = (unsigned short*) calloc(k2 * m * m, sizeof(unsigned short));
    index_a = (unsigned short*) calloc((n / m * n / m), sizeof(unsigned short));
    index_b = (unsigned short*) calloc((n / m * n / m), sizeof(unsigned short));
    h_c = (unsigned int*) calloc(n * n, sizeof(unsigned int));

    readMatrix(input_matrix_file_1, n, m, h_a, index_a);
    readMatrix(input_matrix_file_2, n, m, h_b, index_b);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Read Files: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    unsigned short *d_a, *d_b, *d_index_a, *d_index_b; unsigned int *d_c;

    cudaMalloc(&d_a, k1 * m * m * sizeof(unsigned short));
    cudaMalloc(&d_b, k2 * m * m * sizeof(unsigned short));
    cudaMalloc(&d_c, n * n * sizeof(unsigned int));
    cudaMalloc(&d_index_a, n / m * n / m * sizeof(unsigned short));
    cudaMalloc(&d_index_b, n / m * n / m * sizeof(unsigned short));
    cudaMemcpy(d_a, h_a, k1 * m * m * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, k2 * m * m * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_a, index_a, n / m * n / m * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_b, index_b, n / m * n / m * sizeof(unsigned short), cudaMemcpyHostToDevice);
    free(h_a); free(h_b); free(index_a); free(index_b);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load A, B to GPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    // Multiply the matrices and write to output file
    dim3 block_size(m, m); dim3 grid_size(n / m, n / m);
    matrix_multiply_kernel<<<grid_size, block_size>>>(d_a, d_b, d_index_a, d_index_b, d_c, n, m);
    cudaDeviceSynchronize();
    err = cudaGetLastError(); cout << cudaGetErrorString(err) << endl;

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Multiply Matrices: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();
    cudaFree(d_a); cudaFree(d_b);

    cudaMemcpy(h_c, d_c, n * n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_c);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load C to CPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    // cout << "Write results to output file\n";
    writeMatrix(output_matrix_file, n, m, h_c);
    free(h_c);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Write C to File: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - initial_time).count();
    cout << "Total Time Taken: " << total_time << " milliseconds" << endl;
}

int main(int argc, char** argv) {
    string input_file_1 = "data/input2.1";   
    string input_file_2 = "data/input2.2";
    string my_output_file = "data/myoutput";

    // Check if input file and output file names are provided
    if (argc == 4) {
        input_file_1 = argv[1];
        input_file_2 = argv[2];
        my_output_file = argv[3];
    }

    // Call the wrapper function that calculates square matrix
    multiplyMatrixWrapper(input_file_1, input_file_2, my_output_file);

    return 0;
}