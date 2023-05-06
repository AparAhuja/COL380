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
    // Vector to store the blocks of the matrix
    vector<vector<Block>> blocksMatrix; 

    // Number of blocks in the file
    int k = 0;

    // Open the binary file
    ifstream file(fileName, ios::binary);

    // Read the number of rows, columns, and blocks
    file.read(reinterpret_cast<char*>(&n), 4);
    file.read(reinterpret_cast<char*>(&m), 4);
    file.read(reinterpret_cast<char*>(&k), 4);

    // Resize the blocksMatrix vector to have one element per row/column
    blocksMatrix.resize(n/m);

    // Size of the blocks
    int size = m * m;

    // Loop over all blocks in the file
    for (int i = 0; i < k; i++) {

        // Read the row and column of the block
        int row = 0, col = 0;
        file.read(reinterpret_cast<char*>(&row), 4);
        file.read(reinterpret_cast<char*>(&col), 4);

        // Create a new block and set its row and column
        Block block;
        block.data.resize(size);
        block.row = row;
        block.col = col;

        // Read the data of the block from the file
        file.read(reinterpret_cast<char*>(block.data.data()), 2*size);

        // Store the block in the blocksMatrix
        if(rowWise)
            blocksMatrix[row].push_back(block);
        else
            blocksMatrix[col].push_back(block);
    }

    // Sort the blocks in each row/column
    for(int i = 0; i < n/m; i++){
        if(rowWise)
            sort(blocksMatrix[i].begin(), blocksMatrix[i].end(), compareColumn);
        else
            sort(blocksMatrix[i].begin(), blocksMatrix[i].end(), compareRow);
    }

    // Copy the data of the blocks into the output arrays
    int cnt = 0; 
    rowIndex[0] = 0;
    for(int i = 0; i < n/m; i++){
        for(int j = 0; j < blocksMatrix[i].size(); j++){
            if(rowWise) 
                colIndex[cnt] = blocksMatrix[i][j].col;
            else 
                colIndex[cnt] = blocksMatrix[i][j].row;
            copy(blocksMatrix[i][j].data.begin(), blocksMatrix[i][j].data.end(), blocks + cnt*size);
            cnt++;
        }
        rowIndex[i + 1] = cnt;
    }

    // Close the file
    file.close();
}

void writeMatrix(string file_name, int n, int m, unsigned int *c, bool *non_zero_list) {
    ofstream file(file_name, ios::binary);
    // Count the number of non-zero elements
    int non_zero = 0;
    for(int i = 0; i < n/m*n/m; i++) non_zero += non_zero_list[i];

    // Write the matrix size and number of non-zero elements to the file
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&non_zero), 4);

    // Loop over all blocks in the matrix and write the non-zero elements to the file
    for(int i = 0; i < n/m; i++){
        for(int j = 0; j < n/m; j++){
            if(non_zero_list[i*n/m + j]){ // Check if the block has non-zero elements
                // Write the block indices to the file
                file.write(reinterpret_cast<const char*>(&i), 4);
                file.write(reinterpret_cast<const char*>(&j), 4);
                // Write the non-zero elements of the block to the file
                for(int k = 0; k < m*m; k++){
                    file.write(reinterpret_cast<const char*>(&c[(i*n + j*m)*m + k]), 4);
                }
            }
        }
    }

    // Close the output binary file
    file.close();
}

__global__ void matrix_multiply_kernel(unsigned short *a, unsigned short *b, unsigned int *d_index_a_row, unsigned int *d_index_b_row, unsigned short *d_index_a_col, unsigned short *d_index_b_col, unsigned int *c, bool *non_zero, int n, int m) {
    __shared__ unsigned long long int temp[8][8][2];
    short tx = threadIdx.x;
    short ty = threadIdx.y % m;
    short tz = threadIdx.y / m;
    short bx = blockIdx.x;
    short by = blockIdx.y;
    unsigned long long int sum = 0;
    unsigned int start_a = d_index_a_row[by], end_a = d_index_a_row[by + 1];
    unsigned int start_b = d_index_b_row[bx], end_b = d_index_b_row[bx + 1];
    unsigned int temp1, temp2;
    if(end_a - start_a < end_b - start_b){
        if(tz) start_a = (start_a + end_a)/2;
        else end_a = (start_a + end_a)/2;
    }else{
        if(tz) start_b = (start_b + end_b)/2;
        else end_b = (start_b + end_b)/2;
    }
    while(start_a < end_a && start_b < end_b){
        if(d_index_a_col[start_a] == d_index_b_col[start_b]){
            temp1 = (start_a*m + ty) * m;
            temp2 = (start_b*m) * m + tx;
            sum += a[temp1] * b[temp2];
            sum += a[temp1 + 1] * b[temp2 + m];
            sum += a[temp1 + 2] * b[temp2 + 2*m];
            sum += a[temp1 + 3] * b[temp2 + 3*m];
            if(m == 8){
                sum += a[temp1 + 4] * b[temp2 + 4*m];
                sum += a[temp1 + 5] * b[temp2 + 5*m];
                sum += a[temp1 + 6] * b[temp2 + 6*m];
                sum += a[temp1 + 7] * b[temp2 + 7*m];
            }
            start_a++;
        }
        else if(d_index_a_col[start_a] < d_index_b_col[start_b]) start_a++;
        else start_b++;
        if(sum > 4294967295) {
            sum = 4294967295;
            break;
        }
    }
    temp[ty][tx][tz] = sum;
    __syncthreads();
    if(tz == 0 && ty < m && tx < m){
        if(temp[ty][tx][0] > 0 || temp[ty][tx][1] > 0){
            c[(by*n + bx*m)*m + ty*m + tx] = min((unsigned long long int)4294967295, temp[ty][tx][0] + temp[ty][tx][1]);
            non_zero[by*n/m + bx] = true;
        }
    }
}

void multiplyMatrixWrapper(string input_matrix_file_1, string input_matrix_file_2, string output_matrix_file) {
    int n = 0, m = 0, k1 = 0, k2 = 0;
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

    bool *non_zero, *d_non_zero;
    unsigned short *h_a, *h_b, *index_a_col, *index_b_col;
    unsigned int *index_a_row, *index_b_row;

    h_a = (unsigned short*) calloc(k1 * m * m, sizeof(unsigned short));
    h_b = (unsigned short*) calloc(k2 * m * m, sizeof(unsigned short));

    index_a_col = (unsigned short*) calloc(k1, sizeof(unsigned short));
    index_b_col = (unsigned short*) calloc(k2, sizeof(unsigned short));

    index_a_row = (unsigned int*) calloc(n / m + 1, sizeof(unsigned int));
    index_b_row = (unsigned int*) calloc(n / m + 1, sizeof(unsigned int));

    unsigned int *h_c;
    h_c = (unsigned int*) calloc(n * n, sizeof(unsigned int));

    readMatrix(input_matrix_file_1, n, m, h_a, index_a_row, index_a_col, true);
    readMatrix(input_matrix_file_2, n, m, h_b, index_b_row, index_b_col, false);

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Read Files: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    unsigned short *d_a, *d_b, *d_index_a_col, *d_index_b_col;
    unsigned int *d_index_a_row, *d_index_b_row;
    cudaMalloc(&d_non_zero, n/m * n/m * sizeof(bool));
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

    unsigned int *d_c;
    cudaMalloc(&d_c, n * n * sizeof(unsigned int));
    cudaDeviceSynchronize();

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load A, B to GPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    // Multiply the matrices and write to output file
    dim3 block_size(m, 2*m); dim3 grid_size(n / m, n / m);
    matrix_multiply_kernel<<<grid_size, block_size>>>(d_a, d_b, d_index_a_row, d_index_b_row, d_index_a_col, d_index_b_col, d_c, d_non_zero, n, m);
    cudaDeviceSynchronize();

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Multiply Matrices: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();
    
    non_zero = (bool*) calloc(n/m * n/m, sizeof(bool));
    cudaMemcpy(h_c, d_c, n * n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(non_zero, d_non_zero, n/m * n/m * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    end_time = chrono::high_resolution_clock::now();
    total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Load C to CPU: " << total_time << " milliseconds" << endl;
    start_time = chrono::high_resolution_clock::now();

    writeMatrix(output_matrix_file, n, m, h_c, non_zero);
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