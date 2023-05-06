#include <iostream>
#include <algorithm>
#include <string.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <omp.h>

using namespace std;

const unsigned int MAX_VAL = 4294967295;

// Structure to store a block of the sparse matrix
struct Block {
    int row;          // Row index of the block
    int col;          // Column index of the block
    vector<unsigned short> data; // Data stored in the block
    vector<unsigned int> result; // Data stored in the block
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
void readMatrix(string& fileName, int& numRows, int& blockSize, vector<vector<Block>> &blocksMatrix, bool rowWise) {
    int k = 0;
    ifstream file(fileName, ios::binary);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&blockSize), 4);
    file.read(reinterpret_cast<char*>(&k), 4);

    blocksMatrix.resize(numRows/blockSize);
    int size = blockSize * blockSize;
    int rowIndex = 0, colIndex = 0;

    // Loop over all blocks in the file
    for (int i = 0; i < k; i++) {
        file.read(reinterpret_cast<char*>(&rowIndex), 4);
        file.read(reinterpret_cast<char*>(&colIndex), 4);
        Block block;
        block.data.resize(size);
        block.row = rowIndex;
        block.col = colIndex;
        file.read(reinterpret_cast<char*>(block.data.data()), 2*size);
        
        if(rowWise)
            blocksMatrix[rowIndex].push_back(block);
        else
            blocksMatrix[colIndex].push_back(block);
    }

    // Sort the blocks in each row
    for(int i = 0; i < blocksMatrix.size(); i++){
        if(rowWise)
            sort(blocksMatrix[i].begin(), blocksMatrix[i].end(), compareColumn);
        else
            sort(blocksMatrix[i].begin(), blocksMatrix[i].end(), compareRow);
    }
    file.close();
}

// Function to calculate the result of block-matrix multiplacation
void matrix_mult(vector<unsigned int> &result, vector<unsigned short> &matrixA, vector<unsigned short>&matrixB, int m){
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < m; j++) {
                result[i*m + j] += matrixA[i*m + k] * matrixB[k*m + j];
            }
        }
    }
}

// Function to multiply two sparse matrices
void multiplyMatrix(string output_matrix_file, vector<vector<Block>> &rowBlocks, vector<vector<Block>> &colBlocks, int n, int m) {
    int k = 0;
    ofstream file(output_matrix_file, ios::binary);
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&k), 4);

    for (int i = 0; i < n/m; i++) {
        for(int j = 0; j < n/m; j++){
            int colIndex = 0, max_rowIndex = rowBlocks[i].size(), max_colIndex = colBlocks[j].size();
            if(max_colIndex == 0 || max_colIndex == 0) {
                // Skip if either row or column does not have any data
                continue;
            }
            // Flag to check if any multiplication was performed
            bool flag = false;
            Block b;
            b.row = i;
            b.col = j;
            b.result = vector<unsigned int> (m*m, 0);

            // Loop through the blocks in the row and column
            for(int rowIndex = 0; rowIndex < max_rowIndex; rowIndex++){
                while(colIndex < max_colIndex && colBlocks[j][colIndex].row < rowBlocks[i][rowIndex].col){
                    colIndex++;
                }
                if(colIndex == max_colIndex){
                    // No matching block found in the column
                    break;
                }
                if(colBlocks[j][colIndex].row == rowBlocks[i][rowIndex].col){
                    flag = true;
                    // Perform matrix multiplication
                    matrix_mult(b.result, rowBlocks[i][rowIndex].data, colBlocks[j][colIndex].data, m);
                }
            }
            if(flag) {
                bool is_submatrix_nonzero = false;
                // Check if submatrix is non-zero
                for (auto& data : b.result) {
                    if (data != 0) {
                        is_submatrix_nonzero = true;
                        break;
                    }
                }
                if(is_submatrix_nonzero){
                    k++;
                    file.write(reinterpret_cast<const char*>(&b.row), 4);
                    file.write(reinterpret_cast<const char*>(&b.col), 4);
                    for(auto& data : b.result){
                        data = min(data, MAX_VAL);
                        file.write(reinterpret_cast<const char*>(&data), 4);
                    }
                }
            }
        }
    }
    file.close();
    fstream update_k(output_matrix_file, ios::binary | ios::in | ios::out);
    update_k.write(reinterpret_cast<const char*>(&n), 4);
    update_k.write(reinterpret_cast<const char*>(&m), 4);
    update_k.write(reinterpret_cast<const char*>(&k), 4);
    update_k.close();
}

void printMartix(vector<vector<Block>> &blocksMatrix){
    for(auto x : blocksMatrix){
        for(auto& block : x){
            cout << "Row: " << block.row << " Col: " << block.col << endl;
            for(auto e : block.data)
                cout << e << " ";    
            cout << endl;
        }
    }
}

void multiplyMatrixWrapper(string input_matrix_file_1, string input_matrix_file_2, string output_matrix_file) {
    int num_rows = 0, blk_size = 0;
    auto start_time = chrono::high_resolution_clock::now();  // record start time

    // Read input matrix from file and store blocks in a vector of vectors
    vector<vector<Block>> row_blocks;
    readMatrix(input_matrix_file_1, num_rows, blk_size, row_blocks, true);
    vector<vector<Block>> col_blocks;
    readMatrix(input_matrix_file_2, num_rows, blk_size, col_blocks, false);

    // Multiply the matrices and write to output file
    multiplyMatrix(output_matrix_file, row_blocks, col_blocks, num_rows, blk_size);

    auto end_time = chrono::high_resolution_clock::now();  // record end time
    auto total_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "Total Time Taken: " << total_time << " milliseconds" << endl;
}

int main(int argc, char** argv) {
    string input_file_1 = "data/input1";
    string input_file_2 = "data/input2";
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
