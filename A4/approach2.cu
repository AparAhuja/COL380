#include <iostream>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

// Function to read a sparse matrix stored in a binary file
void readMatrix(string& fileName, int& n, int& m, unsigned int *blocksMatrix) {
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

        // Read in the data of the block
        for(int i = 0; i < size; i++){
            unsigned int data = 0;
            int relative_row = i / m, relative_col = i % m;
            int actual_row = relative_row + rowIndex*m;
            int actual_col = relative_col + colIndex*m;
            int actual_index = actual_row * n + actual_col;
            file.read(reinterpret_cast<char*>(&data), 2);
            blocksMatrix[actual_index] = data;
        }
    }
    file.close();
}

void writeMatrix(string file_name, int n, int m, unsigned int *c) {
    ofstream file(file_name, ios::binary); int non_zero = 0;
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&non_zero), 4);
    for(int i = 0; i < n/m; i++){
        for(int j = 0; j < n/m; j++){
            int checksum = 0;
            for(int k = 0; k < m*m; k++){
                // rel-row = k/m ; rel-col = k%m ; base-row = i*m ; base-col = j*m
                // actual-row = base-row + rel-row ; actual-col = base-col + rel-col
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
                    data = c[actual_index];
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

__global__ void matrix_multiply_kernel(const unsigned int *a, const unsigned int *b, unsigned int *c, int n, int m) {
    __shared__ unsigned int s_a[8][8];
    __shared__ unsigned int s_b[8][8];
    unsigned int MAX_VAL = 4294967295;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = 0;

    for (int i = 0; i < n / m; ++i) {
        s_a[ty][tx] = a[row * n + i * m + tx];
        s_b[ty][tx] = b[(i * m + ty) * n + col];
        __syncthreads();

        for (int j = 0; j < m; ++j) {
            sum += s_a[ty][j] * s_b[j][tx];
        }
        __syncthreads();
    }
    c[row * n + col] = min(sum, MAX_VAL);
}

void multiplyMatrix(unsigned int *a, unsigned int *b, unsigned int *c, int n, int m) {
    dim3 block_size(m, m);
    dim3 grid_size(n / m, n / m);
    matrix_multiply_kernel<<<grid_size, block_size>>>(a, b, c, n, m);
}

void multiplyMatrixWrapper(string input_matrix_file_1, string input_matrix_file_2, string output_matrix_file) {
    int n = 0, m = 0;
    auto initial_time = chrono::high_resolution_clock::now();
    auto start_time = chrono::high_resolution_clock::now(), end_time = chrono::high_resolution_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    // Read input matrix from file and store blocks in a vector of vectors
    ifstream file(input_matrix_file_1, ios::binary);
    file.read(reinterpret_cast<char*>(&n), 4);
    file.read(reinterpret_cast<char*>(&m), 4);
    file.close();

    unsigned int *h_a, *h_b, *h_c;
    h_a = (unsigned int*) calloc(n * n, sizeof(unsigned int));
    h_b = (unsigned int*) calloc(n * n, sizeof(unsigned int));
    h_c = (unsigned int*) calloc(n * n, sizeof(unsigned int));
    
    readMatrix(input_matrix_file_1, n, m, h_a);
    readMatrix(input_matrix_file_2, n, m, h_b);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Read Files: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    unsigned int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * n * sizeof(unsigned int));
    cudaMalloc(&d_b, n * n * sizeof(unsigned int));
    cudaMalloc(&d_c, n * n * sizeof(unsigned int));
    cudaMemcpyAsync(d_a, h_a, n * n * sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b, h_b, n * n * sizeof(unsigned int), cudaMemcpyHostToDevice, stream2);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError(); cout << cudaGetErrorString(err) << endl;
    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load A, B to GPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    // Multiply the matrices and write to output file
    multiplyMatrix(d_a, d_b, d_c, n, m);
    cudaDeviceSynchronize();
    err = cudaGetLastError(); cout << cudaGetErrorString(err) << endl;
    
    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Multiply Matrices: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    cudaMemcpy(h_c, d_c, n * n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load C to CPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    // cout << "Write results to output file\n";
    writeMatrix(output_matrix_file, n, m, h_c);

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