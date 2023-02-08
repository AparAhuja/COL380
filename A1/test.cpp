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

struct Block {
    int row;
    int col;
    vector<int> data;
};

bool rowcmp(Block b1, Block b2){
    return b1.col < b2.col;
}

int file_size(string& file_name){
    ifstream in_file(file_name, ios::binary);
    in_file.seekg(0, ios::end);
    int filesize = in_file.tellg();
    return filesize;
}

void read_matrix(string& file_name, int& n, int& m, vector<vector<Block>> &rowBlocks, int outMatrix = 0) {
    int k = 0;
    ifstream file(file_name, ios::binary);
    file.read(reinterpret_cast<char*>(&n), 4);
    file.read(reinterpret_cast<char*>(&m), 4);
    file.read(reinterpret_cast<char*>(&k), 4);
    int actual_k = (file_size(file_name) - 12)/((1 + outMatrix)*m*m+8);
    if(actual_k != k) cout << "read_matrix(): k doesn't match. actual-k = " << actual_k << ", k = " << k << endl;
    rowBlocks.resize(n/m);
    int size = m * m, row = 0, col = 0;
    for (int i = 0; i < actual_k; i++) {
        file.read(reinterpret_cast<char*>(&row), 4);
        file.read(reinterpret_cast<char*>(&col), 4);
        Block block, block_T;
        block.data.resize(size);
        block.row = row;
        block.col = col;
        if(row != col){
            block_T.data.resize(size);
            block_T.row = col;
            block_T.col = row;
        }
        for(int i = 0; i < size; i++){
            int data = 0;
            file.read(reinterpret_cast<char*>(&data), 1 + outMatrix);
            block.data[i] = data;
            if(row != col) block_T.data[m*(i%m) + i/m] = data;
        }
        rowBlocks[row].push_back(block);
        if(row != col){
            rowBlocks[col].push_back(block_T);
        }
    }
    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < rowBlocks.size(); i++){
        sort(rowBlocks[i].begin(), rowBlocks[i].end(), rowcmp);
    }
    file.close();
}

void isSameMatrix(const vector<vector<Block>>& rowBlocks1, const vector<vector<Block>>& rowBlocks2) {
    if (rowBlocks1.size() != rowBlocks2.size()){
        cout << "FAILURE: size1 mismatch\n";
        return;
    }
    for (int i = 0; i < rowBlocks1.size(); i++) {
        if (rowBlocks1[i].size() != rowBlocks2[i].size()){
            cout << "FAILURE: size2 mismatch\n";
            return;
        }
        for (int j = 0; j < rowBlocks1[i].size(); j++) {
            if (rowBlocks1[i][j].row != rowBlocks2[i][j].row ||
                rowBlocks1[i][j].col != rowBlocks2[i][j].col ||
                rowBlocks1[i][j].data != rowBlocks2[i][j].data){
                    cout << "FAILURE: data mismatch\n";
                    return;
            }
        }
    }
    cout << "SUCCESS! ";
    return;
}

void compare_output(string& filename1, string& filename2){
    ifstream file1(filename1, ios::binary);
    ifstream file2(filename2, ios::binary);
    int n1 = 0, n2 = 0, m1 = 0, m2 = 0, k1 = 0, k2 = 0;
    file1.read(reinterpret_cast<char*>(&n1), 4);
    file2.read(reinterpret_cast<char*>(&n2), 4);
    if(n1 != n2){
        cout << "compare_output(): n not equal. n1 = " << n1 << ", n2 = " << n2 << ".\n";
        return;
    }
    file1.read(reinterpret_cast<char*>(&m1), 4);
    file2.read(reinterpret_cast<char*>(&m2), 4);
    if(m1 != m2){
        cout << "compare_output(): m not equal. m1 = " << m1 << ", m2 = " << m2 << ".\n";
        return;
    }
    file1.read(reinterpret_cast<char*>(&k1), 4);
    file2.read(reinterpret_cast<char*>(&k2), 4);
    if(k1 != k2){
        cout << "compare_output(): k not equal. k1 = " << k1 << ", k2 = " << k2 << ".\n";
        return;
    }
    int actual_k1 = (file_size(filename1) - 12)/(2*m1*m1+8);
    int actual_k2 = (file_size(filename2) - 12)/(2*m2*m2+8);
    k1 = actual_k1;
    k2 = actual_k2;
    int size = m1 * m1, row1 = 0, col1 = 0, row2 = 0, col2 = 0;
    vector<vector<Block>> rowBlocks1, rowBlocks2;
    rowBlocks1.resize(n1/m1);
    rowBlocks2.resize(n2/m2);
    for (int i = 0; i < k1; i++) {
        file1.read(reinterpret_cast<char*>(&row1), 4);
        file1.read(reinterpret_cast<char*>(&col1), 4);
        Block block;
        block.data.resize(size);
        block.row = row1;
        block.col = col1;
        for(int i = 0; i < size; i++){
            int data1 = 0;
            file1.read(reinterpret_cast<char*>(&data1), 2);
            block.data[i] = data1;
        }
        rowBlocks1[row1].push_back(block);
    }
    for (int i = 0; i < k2; i++) {
        file2.read(reinterpret_cast<char*>(&row2), 4);
        file2.read(reinterpret_cast<char*>(&col2), 4);
        Block block, block_T;
        block.data.resize(size);
        block.row = row2;
        block.col = col2;
        for(int i = 0; i < size; i++){
            int data2 = 0;
            file2.read(reinterpret_cast<char*>(&data2), 2);
            block.data[i] = data2;
        }
        rowBlocks2[row2].push_back(block);
    }
    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < rowBlocks1.size(); i++)
        sort(rowBlocks1[i].begin(), rowBlocks1[i].end(), rowcmp);
    #pragma omp parallel for schedule(dynamic) 
    for(int i = 0; i < rowBlocks2.size(); i++)
        sort(rowBlocks2[i].begin(), rowBlocks2[i].end(), rowcmp);
    isSameMatrix(rowBlocks1, rowBlocks2);
    file1.close();
    file2.close();
}

void matrix_mult(vector<int> &c, vector<int> &a, vector<int>&b, int m){
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < m; j++) {
                c[i*m + j] = Outer(Inner(a[i*m + k], b[k + j*m]), c[i*m + j]);
            }
        }
    }
}

void square_matrix(string& file_name, vector<vector<Block>> &rowBlocks, int n, int m) {
    ofstream file(file_name, ios::binary);
    int k = 0;
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&k), 4);
    int num_t = omp_get_num_threads();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n/m; i++) {
        for(int j = i; j < n/m; j++){
            int colIndex = 0, max_rowIndex = rowBlocks[i].size(), max_colIndex = rowBlocks[j].size();
            if(max_colIndex == 0 || max_colIndex == 0) continue;
            bool flag = false;
            Block b;
            b.row = i;
            b.col = j;
            b.data = vector<int> (m*m, 0);
            for(int rowIndex = 0; rowIndex < max_rowIndex; rowIndex++){
                while(colIndex < max_colIndex && rowBlocks[j][colIndex].col < rowBlocks[i][rowIndex].col){
                    colIndex++;
                }
                if(colIndex == max_colIndex){
                    break;
                }
                if(rowBlocks[j][colIndex].col == rowBlocks[i][rowIndex].col){
                    flag = true;
                    matrix_mult(b.data, rowBlocks[i][rowIndex].data, rowBlocks[j][colIndex].data, m);
                }
            }
            if(flag) {
                bool is_submatrix_nonzero = false;
                for (auto& data : b.data) {
                    if (data != 0) {
                        is_submatrix_nonzero = true; 
                        break;
                    }
                }
                if(is_submatrix_nonzero){
                    #pragma omp task 
                    {   
                        #pragma omp critical
                        {
                            if(b.row != b.col) k+=2;
                            else {k++;}
                            file.write(reinterpret_cast<const char*>(&b.row), 4);
                            file.write(reinterpret_cast<const char*>(&b.col), 4);
                            for(auto& data : b.data){
                                data = min(data, MAX_VAL);
                                file.write(reinterpret_cast<const char*>(&data), 2);
                            }
                        }
                    }
                }
            }
        }
    }
    file.close();
    fstream fileagain(file_name, ios::binary | ios::in | ios::out);
    fileagain.write(reinterpret_cast<const char*>(&n), 4);
    fileagain.write(reinterpret_cast<const char*>(&m), 4);
    fileagain.write(reinterpret_cast<const char*>(&k), 4);
    fileagain.close();
}

void final_code(string inputfile, string myoutputfile, string realoutputfile){
    int n = 0, m = 0;
    auto total_start = high_resolution_clock::now();
    vector<vector<Block>> rowBlocks;
    read_matrix(inputfile, n, m, rowBlocks);
    square_matrix(myoutputfile, rowBlocks, n, m);
    auto total_end = high_resolution_clock::now();
    auto total_time = duration_cast<milliseconds>(total_end - total_start).count();
    compare_output(myoutputfile, realoutputfile);
    cout << "Total Time Taken: " << total_time << "\n";
}

int main(int argc, char** argv) {
    int num_t = 8;
    string inputfile = "data/input2", myoutputfile = "data/myoutput", realoutputfile = "data/output2";
    if(argc > 1){
        inputfile = argv[1];
        myoutputfile = argv[2];
    }
    if(argc > 3)
        num_t = atoi(argv[3]);
    omp_set_num_threads(num_t);
    final_code(inputfile, myoutputfile, realoutputfile);
    return 0;
}