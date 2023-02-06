#include <iostream>
#include "dataloader.h"
#include <algorithm>
#include <string.h>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void create_test_data(){
    const string file = "data/input3";
    int n = 4, m = 2; 

    vector<vector<Block>> rowBlocks(2);
    vector<int> v1 = {2, 2, 2 ,2}, v2 = {3, 3, 3, 3}, v3 = {1, 1, 1, 1};
    Block b1; b1.row = 0; b1.col = 0; b1.data = v1;
    Block b2; b2.row = 1; b2.col = 1; b2.data = v2;
    Block b3; b3.row = 0; b3.col = 1; b3.data = v3;
    rowBlocks[0].push_back(b1);
    rowBlocks[1].push_back(b2);
    rowBlocks[1].push_back(b3);

    write_matrix(file, n, m, rowBlocks);
    printMatrix(rowBlocks, m);
}

void check_test_data(){
    const string file = "data/output2";
    int n = -1, m = -1, outMatrix = 0;
    vector<vector<Block>> rowBlocks, colBlocks, result;
    if(file.find("output") != string::npos) outMatrix = 1;
    read_matrix(file, n, m, rowBlocks, outMatrix);
    // read_matrix(file, n, m, rowBlocks, colBlocks, outMatrix); // original
    printMatrix(rowBlocks, m);
}

void final_code(){
    const string inputfile = "data/input2", myoutputfile = "data/myoutput", realoutputfile = "data/output2";
    int n = -1, m = -1;
    auto total_start = high_resolution_clock::now();
    vector<vector<Block>> rowBlocks, colBlocks, result;
    cout << "Reading Matrix from Input File...\n";
    auto start = high_resolution_clock::now();
    read_matrix(inputfile, n, m, rowBlocks);
    // read_matrix(inputfile, n, m, rowBlocks, colBlocks); // original
    auto end = high_resolution_clock::now();
    auto time_taken = duration_cast<milliseconds>(end - start).count();
    cout << "Reading Matrix from Input File Complete. Time Taken: " << time_taken << "\n";
    // printMatrix(rowBlocks, m);
    cout << "Computing Square of Matrix...\n";
    start = high_resolution_clock::now();
    square_matrix(rowBlocks, result, n, m);
    // square_matrix(rowBlocks, colBlocks, result, n, m); // original
    end = high_resolution_clock::now();
    time_taken = duration_cast<milliseconds>(end - start).count();
    cout << "Computing Square of Matrix Complete. Time Taken: " << time_taken << "\n";
    // printMatrix(result, m);
    cout << "Writing the Result to Output File...\n";
    start = high_resolution_clock::now();
    write_matrix(myoutputfile, n, m, result, 1);
    end = high_resolution_clock::now();
    time_taken = duration_cast<milliseconds>(end - start).count();
    cout << "Writing the Result to Output File Complete. Time Taken: " << time_taken << "\n";
    // printMatrix(result, m);
    auto total_end = high_resolution_clock::now();
    auto total_time = duration_cast<milliseconds>(total_end - total_start).count();
    cout << "Total Time Taken: " << total_time << "\n";
    compare_output(myoutputfile, realoutputfile);
}

int main(int argc, char** argv) {
    int arg = 0;
    omp_set_num_threads(64);
    if(argc > 1)
        arg = atoi(argv[1]);
    if(arg == 0)
        final_code();
    else if(arg == 1)
        create_test_data();
    else if(arg == 2)
        check_test_data();
    return 0;
}