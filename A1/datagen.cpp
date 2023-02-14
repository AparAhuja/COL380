#include <iostream>
#include <string.h>
#include <fstream>
#include <vector>
#include <set>
#include <cstdlib>

using namespace std; 

int file_size(string& file_name){
    ifstream in_file(file_name, ios::binary);
    in_file.seekg(0, ios::end);
    int filesize = in_file.tellg();
    return filesize;
}

void dataGen(string file_name, int n, int m, int k){
    fstream file(file_name, ios::binary | ios::out);
    file.write(reinterpret_cast<const char*>(&n), 4);
    file.write(reinterpret_cast<const char*>(&m), 4);
    file.write(reinterpret_cast<const char*>(&k), 4);
    int num_rows = n/m;
    set< pair<int, int> > indices;
    while(indices.size() < k){
        int row = rand()%num_rows;
        int col = rand()%num_rows;
        col = row + col;
        row = min(row, col);
        col = col - row;
        indices.insert({row, col});
    }
    int val;
    for(auto &index: indices){
        file.write(reinterpret_cast<const char*>(&index.first), 4);
        file.write(reinterpret_cast<const char*>(&index.second), 4);
        for(int i = 0; i < m*m; i++) {
            val = rand()%256;
            file.write(reinterpret_cast<const char*>(&val), 1);
        }
    }
    file.close();
    int size = file_size(file_name);
    if(size != k*(m*m+8) + 12){
        cout << "ERROR: FILE SIZE INCORRECT. correct size = " << size << ". given size = "  << k*(m*m+8) + 12  << ".\n";
    }
}

int main(int argc, char** argv) {
    int n = 1500000, m = 25, logk = 20;
    if (argc > 1) {
        logk = atoi(argv[1]);
    }
    string file = "data/input_n_" + to_string(n) + "_m_" + to_string(m) + "_logk_" + to_string(logk);
    int k = (1 << logk);
    if(k > n) {
        cout << "ERROR: NUMBER OF BLOCKS INCORRECT (k > n)\n";
        return 0;
    }
    dataGen(file, n, m, k);
    return 0;
}
