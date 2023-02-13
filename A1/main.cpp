#include <iostream>
#include <algorithm>
#include <string.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <omp.h>
#include "library.hpp"

using namespace std;
using namespace std::chrono;

// Structure to store a block of the sparse matrix
struct Block {
    int row;          // Row index of the block
    int col;          // Column index of the block
    vector<int> data; // Data stored in the block
};

// Function to compare two blocks based on their column index
bool compareColumn(Block block1, Block block2){
    return block1.col < block2.col;
}

// Function to read a sparse matrix stored in a binary file
void readMatrix(string& fileName, int& numRows, int& blockSize, vector<vector<Block>> &rowBlocks, int outMatrix = 0) {
    int k = 0;
    ifstream file(fileName, ios::binary);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&blockSize), 4);
    file.read(reinterpret_cast<char*>(&k), 4);
    
    rowBlocks.resize(numRows/blockSize);
    int size = blockSize * blockSize;
    int rowIndex = 0, colIndex = 0;

    // Loop over all blocks in the file
    for (int i = 0; i < k; i++) {
        file.read(reinterpret_cast<char*>(&rowIndex), 4);
        file.read(reinterpret_cast<char*>(&colIndex), 4);
        Block block, blockTranspose;
        block.data.resize(size);
        block.row = rowIndex;
        block.col = colIndex;
        if(rowIndex != colIndex){
            blockTranspose.data.resize(size);
            blockTranspose.row = colIndex;
            blockTranspose.col = rowIndex;
        }

        // Read in the data of the block
        for(int i = 0; i < size; i++){
            int data = 0;
            file.read(reinterpret_cast<char*>(&data), 1 + outMatrix);
            block.data[i] = data;
            if(rowIndex != colIndex)
                blockTranspose.data[blockSize*(i%blockSize) + i/blockSize] = data;
        }
        rowBlocks[rowIndex].push_back(block);
        if(rowIndex != colIndex){
            rowBlocks[colIndex].push_back(blockTranspose);
        }
    }

    // Sort the blocks in each row
    #pragma omp parallel
    #pragma omp single
    for(int i = 0; i < rowBlocks.size(); i++){
        #pragma omp task
        sort(rowBlocks[i].begin(), rowBlocks[i].end(), compareColumn);
    }
    file.close();
}

void compareMatrices(const vector<vector<Block>>& rowBlocks1, const vector<vector<Block>>& rowBlocks2) {
    // check if the size of the two matrices is equal
    if (rowBlocks1.size() != rowBlocks2.size()){
        cout << "FAILURE: Rows count mismatch. rowBlocks1 rows: " << rowBlocks1.size() << ", rowBlocks2 rows: " << rowBlocks2.size() << endl;
        return;
    }
    // loop through each row
    for (int i = 0; i < rowBlocks1.size(); i++) {
        // check if the number of blocks in the current row is equal for both matrices
        if (rowBlocks1[i].size() != rowBlocks2[i].size()){
            cout << "FAILURE: Blocks count mismatch in row " << i << ". rowBlocks1 blocks: " << rowBlocks1[i].size() << ", rowBlocks2 blocks: " << rowBlocks2[i].size() << endl;
            return;
        }
        // loop through each block in the current row
        for (int j = 0; j < rowBlocks1[i].size(); j++) {
            // check if the row, col and data values are equal for both blocks
            if (rowBlocks1[i][j].row != rowBlocks2[i][j].row ||
                rowBlocks1[i][j].col != rowBlocks2[i][j].col ||
                rowBlocks1[i][j].data != rowBlocks2[i][j].data){
                    cout << "FAILURE: Data mismatch in block " << j << " of row " << i << endl;
                    cout << "rowBlocks1 block: (" << rowBlocks1[i][j].row << ", " << rowBlocks1[i][j].col << ") = ";
                    for(int k = 0; k < rowBlocks1[i][j].data.size(); k++) cout << rowBlocks1[i][j].data[k] << " ";
                    cout << endl;
                    cout << "rowBlocks2 block: (" << rowBlocks2[i][j].row << ", " << rowBlocks2[i][j].col << ") = ";
                    for(int k = 0; k < rowBlocks2[i][j].data.size(); k++) cout << rowBlocks2[i][j].data[k] << " ";
                    cout << endl;
                    return;
            }
        }
    }
    cout << "SUCCESS! ";
    return;
}


void compareOutputFiles(string& file1, string& file2){
    // Open the two input files for binary reading
    ifstream in_file1(file1, ios::binary);
    ifstream in_file2(file2, ios::binary);

    // Read the values for n, m, and k from both files
    int n1 = 0, n2 = 0, m1 = 0, m2 = 0, k1 = 0, k2 = 0;

    // Read n values
    in_file1.read(reinterpret_cast<char*>(&n1), 4);
    in_file2.read(reinterpret_cast<char*>(&n2), 4);

    // Check if n values are equal, if not print an error message
    if(n1 != n2){
        cout << "compare_sparse_matrices(): n values not equal. n1 = " << n1 << ", n2 = " << n2 << ".\n";
        return;
    }

    // Read m values
    in_file1.read(reinterpret_cast<char*>(&m1), 4);
    in_file2.read(reinterpret_cast<char*>(&m2), 4);

    // Check if m values are equal, if not print an error message
    if(m1 != m2){
        cout << "compare_sparse_matrices(): m values not equal. m1 = " << m1 << ", m2 = " << m2 << ".\n";
        return;
    }

    // Read k values
    in_file1.read(reinterpret_cast<char*>(&k1), 4);
    in_file2.read(reinterpret_cast<char*>(&k2), 4);

    // Check if k values are equal, if not print an error message
    if(k1 != k2){
        cout << "compare_sparse_matrices(): k values not equal. k1 = " << k1 << ", k2 = " << k2 << ".\n";
        return;
    }

    // Calculate the block size
    int size = m1 * m1;
    int row1 = 0, col1 = 0, row2 = 0, col2 = 0;
    
    // Store the blocks in row-wise order
    vector<vector<Block>> rowBlocks1, rowBlocks2;
    rowBlocks1.resize(n1/m1);
    rowBlocks2.resize(n2/m2);

    // Read blocks from file 1 and store in `rowBlocks1`
    for (int i = 0; i < k1; i++) {
        in_file1.read(reinterpret_cast<char*>(&row1), 4);
        in_file1.read(reinterpret_cast<char*>(&col1), 4);
        Block block;
        block.data.resize(size);
        block.row = row1;
        block.col = col1;
        for(int i = 0; i < size; i++){
            int data1 = 0;
            in_file1.read(reinterpret_cast<char*>(&data1), 2);
            block.data[i] = data1;
        }
        rowBlocks1[row1].push_back(block);
    }
    // Read blocks from file 2 and store in `rowBlocks2`
    for (int i = 0; i < k2; i++) {
        in_file2.read(reinterpret_cast<char*>(&row2), 4);
        in_file2.read(reinterpret_cast<char*>(&col2), 4);
        Block block, block_T;
        block.data.resize(size);
        block.row = row2;
        block.col = col2;
        for(int i = 0; i < size; i++){
            int data2 = 0;
            in_file2.read(reinterpret_cast<char*>(&data2), 2);
            block.data[i] = data2;
        }
        rowBlocks2[row2].push_back(block);
    }

    // Sort the blocks in each row
    #pragma omp parallel
    #pragma omp single
    {
        for(int i = 0; i < rowBlocks1.size(); i++)
            #pragma omp task
            sort(rowBlocks1[i].begin(), rowBlocks1[i].end(), compareColumn);
        for(int i = 0; i < rowBlocks2.size(); i++)
            #pragma omp task
            sort(rowBlocks2[i].begin(), rowBlocks2[i].end(), compareColumn);
    }
    
    // Compare the matrices in the files
    compareMatrices(rowBlocks1, rowBlocks2);
    in_file1.close();
    in_file2.close();
}

// Function to calculate the result of block-matrix multiplacation
void matrix_mult(vector<int> &result, vector<int> &matrixA, vector<int>&matrixB, int m){
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < m; j++) {
                result[i*m + j] = Outer(Inner(matrixA[i*m + k], matrixB[k + j*m]), result[i*m + j]);
            }
        }
    }
}

void squareMatrix(string& file_name, vector<vector<Block>> &rowBlocks, int n, int m) {
    int k = 0;
    ofstream file(file_name, ios::binary);
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&k), 4);
    
    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // Parallelize the outer loop
    #pragma omp parallel
    #pragma omp single
    for (int i = 0; i < n/m; i++) {
        #pragma omp task
        {
            for(int j = i; j < n/m; j++){
                int colIndex = 0, max_rowIndex = rowBlocks[i].size(), max_colIndex = rowBlocks[j].size();
                if(max_colIndex == 0 || max_colIndex == 0) {
                    // Skip if either row or column does not have any data
                    continue;
                }
                // Flag to check if any multiplication was performed
                bool flag = false;
                Block b;
                b.row = i;
                b.col = j;
                b.data = vector<int> (m*m, 0);

                // Loop through the blocks in the row and column
                for(int rowIndex = 0; rowIndex < max_rowIndex; rowIndex++){
                    while(colIndex < max_colIndex && rowBlocks[j][colIndex].col < rowBlocks[i][rowIndex].col){
                        colIndex++;
                    }
                    if(colIndex == max_colIndex){
                        // No matching block found in the column
                        break;
                    }
                    if(rowBlocks[j][colIndex].col == rowBlocks[i][rowIndex].col){
                        flag = true;
                        // Perform matrix multiplication
                        matrix_mult(b.data, rowBlocks[i][rowIndex].data, rowBlocks[j][colIndex].data, m);
                    }
                }
                if(flag) {
                    bool is_submatrix_nonzero = false;
                    // Check if submatrix is non-zero
                    for (auto& data : b.data) {
                        if (data != 0) {
                            is_submatrix_nonzero = true;
                            break;
                        }
                    }
                    if(is_submatrix_nonzero){
                        #pragma omp task
                        {
                            omp_set_lock(&writelock);
                            k++;
                            file.write(reinterpret_cast<const char*>(&b.row), 4);
                            file.write(reinterpret_cast<const char*>(&b.col), 4);
                            for(auto& data : b.data){
                                data = min(data, MAX_VAL);
                                file.write(reinterpret_cast<const char*>(&data), 2);
                            }
                            omp_unset_lock(&writelock);
                        }
                    }
                }
            }
        }
    }
    file.close();
    fstream update_k(file_name, ios::binary | ios::in | ios::out);
    update_k.write(reinterpret_cast<const char*>(&n), 4);
    update_k.write(reinterpret_cast<const char*>(&m), 4);
    update_k.write(reinterpret_cast<const char*>(&k), 4);
    update_k.close();
}

void squareMatrixWrapper(string input_matrix_file, string output_matrix_file, string expected_output_file) {
    int num_rows = 0, num_cols = 0;
    auto start_time = high_resolution_clock::now();  // record start time

    // Read input matrix from file and store blocks in a vector of vectors
    vector<vector<Block>> row_blocks;
    readMatrix(input_matrix_file, num_rows, num_cols, row_blocks);

    // Compute the square matrix and store it in the output file
    squareMatrix(output_matrix_file, row_blocks, num_rows, num_cols);

    auto end_time = high_resolution_clock::now();  // record end time
    auto total_time = duration_cast<milliseconds>(end_time - start_time).count();

    // Compare the output matrix to the expected output
    compareOutputFiles(output_matrix_file, expected_output_file);

    cout << "Total Time Taken: " << total_time << "ms" << endl;
}


int main(int argc, char** argv) {
    int num_threads = 8;
    string input_file = "data/input2";
    string my_output_file = "data/myoutput";
    string real_output_file = "data/output2";

    // Check if input file and output file names are provided
    if (argc > 1) {
        input_file = argv[1];
        my_output_file = argv[2];
    }

    // Check if number of threads is provided
    if (argc > 3) {
        num_threads = atoi(argv[3]);
    }

    // Set the number of threads to be used
    omp_set_num_threads(num_threads);

    // Call the wrapper function that calculates square matrix
    squareMatrixWrapper(input_file, my_output_file, real_output_file);

    return 0;
}
