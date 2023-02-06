#include <fstream>
#include <vector>
#include <cassert>
#include <omp.h>
#include "library.hpp"

using namespace std; 

struct Block {
    int row;
    int col;
    vector<int> data;
};

bool rowcmp(Block b1, Block b2){
    return b1.col < b2.col;
}

bool colcmp(Block b1, Block b2){
    return b1.row < b2.row;
}

int file_size(const string& file_name){
    ifstream in_file(file_name, ios::binary);
    in_file.seekg(0, ios::end);
    int filesize = in_file.tellg();
    return filesize;
}

void read_matrix(const string& file_name, int& n, int& m, vector<vector<Block>> &rowBlocks, int outMatrix = 0) {
    int k;
    ifstream file(file_name, ios::binary);
    file.read(reinterpret_cast<char*>(&n), 4);
    file.read(reinterpret_cast<char*>(&m), 4);
    file.read(reinterpret_cast<char*>(&k), 4);
    cout << "n = " << n << ", m = " << m << ", k = " << k << endl;
    int actual_k = (file_size(file_name) - 12)/((1 + outMatrix)*m*m+8);
    if(actual_k != k) cout << "read_matrix(): k doesn't match. actual-k = " << actual_k << ", k = " << k << endl;
    rowBlocks.resize(n/m);
    int size = m * m, row, col;
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
            int data;
            file.read(reinterpret_cast<char*>(&data), 1 + outMatrix);
            block.data[i] = data;
            if(row != col) block_T.data[m*(i%m) + i/m] = data;
        }
        rowBlocks[row].push_back(block);
        if(row != col){
            rowBlocks[col].push_back(block_T);
        }
    }
    #pragma omp parallel for
    for(int i = 0; i < rowBlocks.size(); i++){
        sort(rowBlocks[i].begin(), rowBlocks[i].end(), rowcmp);
    }
    file.close();
}

void isSameMatrix(const vector<vector<Block>>& rowBlocks1, const vector<vector<Block>>& rowBlocks2) {
    if (rowBlocks1.size() != rowBlocks2.size()){
        cout << "size1 mismatch\n";
        return;
    }
    for (int i = 0; i < rowBlocks1.size(); i++) {
        if (rowBlocks1[i].size() != rowBlocks2[i].size()){
            cout << "size2 mismatch\n";
            return;
        }
        for (int j = 0; j < rowBlocks1[i].size(); j++) {
            if (rowBlocks1[i][j].row != rowBlocks2[i][j].row ||
                rowBlocks1[i][j].col != rowBlocks2[i][j].col ||
                rowBlocks1[i][j].data != rowBlocks2[i][j].data){
                    cout << "data mismatch\n";
                    return;
                }
        }
    }
    return;
}

void compare_output(const string& filename1, const string& filename2){
    ifstream file1(filename1, ios::binary);
    ifstream file2(filename2, ios::binary);
    int n1, n2, m1, m2, k1, k2;
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
        int actual_k1 = (file_size(filename1) - 12)/(2*m1*m1+8);
        int actual_k2 = (file_size(filename2) - 12)/(2*m2*m2+8);
        cout << "compare_output(): updating k1 to " << actual_k1 << " and k2 to " << actual_k2 << ".\n";
        k1 = actual_k1;
        k2 = actual_k2;
        // return;
    }
    int size = m1 * m1, row1, col1, row2, col2;
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
            int data1=0;
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
            int data2=0;
            file2.read(reinterpret_cast<char*>(&data2), 2);
            block.data[i] = data2;
        }
        rowBlocks2[row2].push_back(block);
    }
    for(auto& row: rowBlocks1)
        sort(row.begin(), row.end(), rowcmp);
    for(auto& row: rowBlocks2)
        sort(row.begin(), row.end(), rowcmp);
    isSameMatrix(rowBlocks1, rowBlocks2);
    file1.close();
    file2.close();
}

void write_matrix(const string& file_name, int n, int m, vector<vector<Block>> &rowBlocks, int outMatrix = 0) {
    ofstream file(file_name, ios::binary);
    int k = 0;
    cout << "n = " << n << ", m = " << m << ", ";
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&k), 4);
    for (auto& row : rowBlocks) {
        for (auto& block : row) {
            k++;
            // cout << block.row << " " << block.col << endl;
            // for(auto x : block.data) cout << x << " "; cout << endl;
            file.write(reinterpret_cast<const char*>(&block.row), 4);
            file.write(reinterpret_cast<const char*>(&block.col), 4);
            for(auto& data : block.data){
                data = min(data, MAX_VAL);
                file.write(reinterpret_cast<const char*>(&data), 1 + outMatrix);
            }
        }
    }
    file.close();
    fstream fileagain(file_name, ios::binary | ios::in | ios::out);
    cout << "k = " << k << endl;
    fileagain.write(reinterpret_cast<const char*>(&n), 4);
    fileagain.write(reinterpret_cast<const char*>(&m), 4);
    fileagain.write(reinterpret_cast<const char*>(&k), 4);
    fileagain.close();
}

void printMatrix(const vector<vector<Block>>& rowBlocks, int m) {
    int cnt = 0;
    for (int i = 0; i < rowBlocks.size(); i++) {
        bool flag = false;
        for (int j = 0; j < rowBlocks[i].size(); j++) {
            Block block = rowBlocks[i][j];
            bool is_submatrix_nonzero = false;
            for (auto& data : block.data) {
                if (data != 0) {
                    is_submatrix_nonzero = true; break;
                }
            }
            if(!is_submatrix_nonzero) continue;
            if(block.row > block.col) continue;
            flag = true;
            cout << "Block (" << block.row << ", " << block.col << "):\n";
            int count = 0;
            for (int k = 0; k < block.data.size(); k++) {
                cout << block.data[k] << " ";
                count++;
                if (count == m) {
                    cout << endl;
                    count = 0;
                }
            }
            cnt++;
            if(cnt == 2) return;
        }
        if (flag) {
            cout << "_____________________\n\n";
        }
    }
}

void elementWise_Inner(vector<int> &a, vector<int> &b, vector<int> &c){
    int size = a.size();
    c.resize(size);
    for(int i = 0; i < size; i++){
        c[i] = Inner(a[i], b[i]);
    }
}

void elementWise_Outer(vector<int> &a, vector<int> &b, vector<int> &c){
    int size = a.size();
    c.resize(size);
    // #pragma omp parallel for
    for(int i = 0; i < size; i++){
        c[i] = Outer(a[i], b[i]);
    }
}

void matrix_mult(vector<int> &c, vector<int> &a, vector<int>&b, int m){
    // #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < m; j++) {
                c[i*m + j] = Outer(Inner(a[i*m + k], b[k + j*m]), c[i*m + j]);
                // c[i*m + j] = Outer(Inner(a[i*m + k], b[k*m + j]), c[i*m + j]); // original
            }
        }
    }
}

void square_matrix(vector<vector<Block>> &rowBlocks, vector<vector<Block>> &result, int n, int m) {
    result.resize(n/m);
    #pragma omp parallel for
    for (int i = 0; i < n/m; i++) {
        vector<Block> blocks;
        for(int j = i; j < n/m; j++){
            bool flag = false;
            int colIndex = 0, max_rowIndex = rowBlocks[i].size(), max_colIndex = rowBlocks[j].size();
            if(max_colIndex == 0 || max_colIndex == 0) continue;
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
                    // cout << colBlocks[j][colIndex].row << rowBlocks[i][rowIndex].col << endl;
                    // cout << "$ "; for(auto x : rowBlocks[i][rowIndex].data) cout << x << " "; cout << endl;
                    vector<int> matData(m*m, 0);
                    flag = true;
                    matrix_mult(matData, rowBlocks[i][rowIndex].data, rowBlocks[j][colIndex].data, m);
                    // cout << "> "; for(auto x : matData) cout << x << " "; cout << endl;
                    elementWise_Outer(matData, b.data, b.data);
                }
            }
            if(flag) {
                bool is_submatrix_nonzero = false;
                for (auto& data : b.data) {
                    if (data != 0) {
                        is_submatrix_nonzero = true; break;
                    }
                }
                if(is_submatrix_nonzero)
                    // #pragma omp critical 
                    blocks.push_back(b);
            }
        }
        #pragma omp critical
        result.push_back(blocks);
    }
}

// original
// void square_matrix(vector<vector<Block>> &rowBlocks, vector<vector<Block>> &colBlocks, vector<vector<Block>> &result, int n, int m) {
//     result.resize(n/m);
//     #pragma omp parallel for
//     for (int i = 0; i < n/m; i++) {
//         // int tid = omp_get_thread_num();
//         // cout << tid << endl;
//         vector<Block> blocks;
//         // #pragma omp parallel for
//         for(int j = i; j < n/m; j++){
//             bool flag = false;
//             int colIndex = 0, max_rowIndex = rowBlocks[i].size(), max_colIndex = colBlocks[j].size();
//             if(max_colIndex == 0 || max_colIndex == 0) continue;
//             Block b;
//             b.row = i;
//             b.col = j;
//             b.data = vector<int> (m*m, 0);
//             for(int rowIndex = 0; rowIndex < max_rowIndex; rowIndex++){
//                 while(colIndex < max_colIndex && colBlocks[j][colIndex].row < rowBlocks[i][rowIndex].col){
//                     colIndex++;
//                 }
//                 if(colIndex == max_colIndex){
//                     break;
//                 }
//                 if(colBlocks[j][colIndex].row == rowBlocks[i][rowIndex].col){
//                     // cout << colBlocks[j][colIndex].row << rowBlocks[i][rowIndex].col << endl;
//                     // cout << "$ "; for(auto x : rowBlocks[i][rowIndex].data) cout << x << " "; cout << endl;
//                     vector<int> matData(m*m, 0);
//                     flag = true;
//                     matrix_mult(matData, rowBlocks[i][rowIndex].data, colBlocks[j][colIndex].data, m);
//                     // cout << "> "; for(auto x : matData) cout << x << " "; cout << endl;
//                     elementWise_Outer(matData, b.data, b.data);
//                 }
//             }
//             if(flag) {
//                 bool is_submatrix_nonzero = false;
//                 for (auto& data : b.data) {
//                     if (data != 0) {
//                         is_submatrix_nonzero = true; break;
//                     }
//                 }
//                 if(is_submatrix_nonzero)
//                     // #pragma omp critical 
//                     blocks.push_back(b);
//             }
//         }
//         #pragma omp critical
//         result.push_back(blocks);
//     }
// }

// void read_matrix(const string& file_name, int& n, int& m, vector<vector<Block>> &rowBlocks, vector<vector<Block>> &colBlocks, int outMatrix = 0) {
//     int k;
//     ifstream file(file_name, ios::binary);
//     file.read(reinterpret_cast<char*>(&n), 4);
//     file.read(reinterpret_cast<char*>(&m), 4);
//     file.read(reinterpret_cast<char*>(&k), 4);
//     cout << "n = " << n << ", m = " << m << ", k = " << k << endl;
//     int actual_k = (file_size(file_name) - 12)/((1 + outMatrix)*m*m+8);
//     if(actual_k != k) cout << "read_matrix(): k doesn't match. actual-k = " << actual_k << ", k = " << k << endl;
//     rowBlocks.resize(n/m);
//     colBlocks.resize(n/m);
//     int size = m * m, row, col;
//     // #pragma omp parallel for
//     for (int i = 0; i < actual_k; i++) {
//         file.read(reinterpret_cast<char*>(&row), 4);
//         file.read(reinterpret_cast<char*>(&col), 4);
//         Block block, block_T;
//         block.data.resize(size);
//         block.row = row;
//         block.col = col;
//         if(row != col){
//             block_T.data.resize(size);
//             block_T.row = col;
//             block_T.col = row;
//         }
//         // for(auto& data : block.data)
//         //     file.read(reinterpret_cast<char*>(&data), 1 + outMatrix);
//         for(int i = 0; i < size; i++){
//             int data;
//             file.read(reinterpret_cast<char*>(&data), 1 + outMatrix);
//             block.data[i] = data;
//             if(row != col) block_T.data[m*(i%m) + i/m] = data;
//         }
//         // cout << block.row << " " << block.col << endl;
//         // for(auto x : block.data) cout << x << " "; cout << endl;
//         rowBlocks[row].push_back(block);
//         colBlocks[col].push_back(block);
//         if(row != col){
//             rowBlocks[col].push_back(block_T);
//             colBlocks[row].push_back(block_T);
//         }
//     }
//     #pragma omp parallel for
//     for(int i = 0; i < rowBlocks.size(); i++){
//         sort(rowBlocks[i].begin(), rowBlocks[i].end(), rowcmp);
//     }
//     // for(auto& row: rowBlocks)
//     //     sort(row.begin(), row.end(), rowcmp);
//     #pragma omp parallel for
//     for(int i = 0; i < colBlocks.size(); i++){
//         sort(colBlocks[i].begin(), colBlocks[i].end(), colcmp);
//     }
//     // for(auto& col: colBlocks)
//     //     sort(col.begin(), col.end(), colcmp);
//     file.close();
// }