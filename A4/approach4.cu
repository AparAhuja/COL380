#include <iostream>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

// Structure to store a block of the sparse matrix
struct Block {
    int row;          // Row index of the block
    int col;          // Column index of the block
    vector<unsigned short> data; // Data stored in the block
};

// Function to compare two blocks based on their column index
bool compareColumn(Block block1, Block block2){
    return block1.col < block2.col;
}

// Function to compare two blocks based on their row index
bool compareRow(Block block1, Block block2){
    return block1.row < block2.row;
}

// Function to read a sparse matrix stored in a binary file
void readMatrix(string& fileName, int& n, int& m, unsigned short *blocks, unsigned int *rowIndex, unsigned short *colIndex, bool rowWise) {
    vector<vector<Block>> blocksMatrix; int k = 0;
    ifstream file(fileName, ios::binary);
    file.read(reinterpret_cast<char*>(&n), 4);
    file.read(reinterpret_cast<char*>(&m), 4);
    file.read(reinterpret_cast<char*>(&k), 4);

    blocksMatrix.resize(n/m);
    int size = m * m;

    // Loop over all blocks in the file
    for (int i = 0; i < k; i++) {
        int row = 0, col = 0;
        file.read(reinterpret_cast<char*>(&row), 4);
        file.read(reinterpret_cast<char*>(&col), 4);
        Block block;
        block.data.resize(size);
        block.row = row;
        block.col = col;
        file.read(reinterpret_cast<char*>(block.data.data()), 2*size);
        if(rowWise)
            blocksMatrix[row].push_back(block);
        else
            blocksMatrix[col].push_back(block);
    }

    // Sort the blocks in each row
    for(int i = 0; i < n/m; i++){
        if(rowWise)
            sort(blocksMatrix[i].begin(), blocksMatrix[i].end(), compareColumn);
        else
            sort(blocksMatrix[i].begin(), blocksMatrix[i].end(), compareRow);
    }

    int cnt = 0; rowIndex[0] = 0;
    for(int i = 0; i < n/m; i++){
        for(int j = 0; j < blocksMatrix[i].size(); j++){
            if(rowWise) colIndex[cnt] = blocksMatrix[i][j].col;
            else colIndex[cnt] = blocksMatrix[i][j].row;
            copy(blocksMatrix[i][j].data.begin(), blocksMatrix[i][j].data.end(), blocks + cnt*size);
            cnt++;
        }
        rowIndex[i + 1] = cnt;
    }
    file.close();
}

void writeMatrix(string file_name, int n, int m, unsigned long long int *c) {
    ofstream file(file_name, ios::binary); int non_zero = 0;
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&non_zero), 4);
    for(int i = 0; i < n/m; i++){
        for(int j = 0; j < n/m; j++){
            int flag = 0;
            for(int k = 0; k < m*m; k++){
                int relative_row = k / m, relative_col = k % m;
                int actual_row = relative_row + i*m;
                int actual_col = relative_col + j*m;
                int actual_index = actual_row * n + actual_col;
                flag = c[actual_index];
                if(flag > 0) break;
            }
            if(flag){
                non_zero++;
                file.write(reinterpret_cast<const char*>(&i), 4);
                file.write(reinterpret_cast<const char*>(&j), 4);
                unsigned long long int data;
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

__global__ void matrix_multiply_kernel(unsigned short *a, unsigned short *b, unsigned int *d_index_a_row, unsigned int *d_index_b_row, unsigned short *d_index_a_col, unsigned short *d_index_b_col, unsigned long long int *c, int n, int m) {
    unsigned long long int MAX_VAL = 4294967295;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * blockDim.y + threadIdx.y;
    int col = bx * blockDim.x + threadIdx.x;
    unsigned long long int sum = 0;
    int start_a = d_index_a_row[by], end_a = d_index_a_row[by + 1];
    int start_b = d_index_b_row[bx], end_b = d_index_b_row[bx + 1];
    while(start_a < end_a && start_b < end_b){
        if(d_index_a_col[start_a] == d_index_b_col[start_b]){
            for (int j = 0; j < m; j++) {
                sum += a[start_a*m*m + ty * m + j] * b[start_b*m*m + j * m + tx];
            }start_a++;
        }
        else if(d_index_a_col[start_a] < d_index_b_col[start_b]) start_a++;
        else start_b++;
        if(sum > MAX_VAL) {
            sum = MAX_VAL;
            break;
        }
    }
    if(sum){
        c[row * n + col] = sum;
    }
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

    unsigned short *h_a, *h_b, *index_a_col, *index_b_col;
    unsigned int *index_a_row, *index_b_row;

    h_a = (unsigned short*) calloc(k1 * m * m, sizeof(unsigned short));
    h_b = (unsigned short*) calloc(k2 * m * m, sizeof(unsigned short));

    index_a_col = (unsigned short*) calloc(k1, sizeof(unsigned short));
    index_b_col = (unsigned short*) calloc(k2, sizeof(unsigned short));

    index_a_row = (unsigned int*) calloc(n / m + 1, sizeof(unsigned int));
    index_b_row = (unsigned int*) calloc(n / m + 1, sizeof(unsigned int));

    unsigned long long int *h_c;
    h_c = (unsigned long long int*) calloc(n * n, sizeof(unsigned long long int));

    readMatrix(input_matrix_file_1, n, m, h_a, index_a_row, index_a_col, true);
    readMatrix(input_matrix_file_2, n, m, h_b, index_b_row, index_b_col, false);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Read Files: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    unsigned short *d_a, *d_b, *d_index_a_col, *d_index_b_col;
    unsigned int *d_index_a_row, *d_index_b_row;
    cudaMalloc(&d_a, k1 * m * m * sizeof(unsigned short));
    cudaMalloc(&d_b, k2 * m * m * sizeof(unsigned short));
    cudaMalloc(&d_index_a_row, (n / m + 1) * sizeof(unsigned int));
    cudaMalloc(&d_index_b_row, (n / m + 1) * sizeof(unsigned int));
    cudaMalloc(&d_index_a_col, k1 * sizeof(unsigned short));
    cudaMalloc(&d_index_b_col, k2 * sizeof(unsigned short));
    cudaMemcpy(d_a, h_a, k1 * m * m * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, k2 * m * m * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_a_row, index_a_row, (n / m + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_b_row, index_b_row, (n / m + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_a_col, index_a_col, k1 * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_b_col, index_b_col, k2 * sizeof(unsigned short), cudaMemcpyHostToDevice);
    free(h_a); free(h_b); free(index_a_row); free(index_b_row); free(index_a_col); free(index_b_col);

    unsigned long long int *d_c;
    cudaMalloc(&d_c, n * n * sizeof(unsigned long long int));
    cudaDeviceSynchronize();
    err = cudaGetLastError(); cout << cudaGetErrorString(err) << endl;

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load A, B to GPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    // Multiply the matrices and write to output file
    dim3 block_size(m, m); dim3 grid_size(n / m, n / m);
    matrix_multiply_kernel<<<grid_size, block_size>>>(d_a, d_b, d_index_a_row, d_index_b_row, d_index_a_col, d_index_b_col, d_c, n, m);
    cudaDeviceSynchronize();
    err = cudaGetLastError(); cout << cudaGetErrorString(err) << endl;

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Multiply Matrices: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();
    cudaFree(d_a); cudaFree(d_b);

    cudaMemcpy(h_c, d_c, n * n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_c); 

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load C to CPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

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