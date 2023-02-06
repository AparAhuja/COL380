#include <iostream>
#include <algorithm>
#include <fstream>
#include <string.h>
#include <omp.h>
using namespace std;

int main(){
    omp_set_num_threads(8);
    int n = 100;
    vector<int> v(n, 0);
    int i = 0; for(auto &x : v) {v = i; i++;}
    #pragma omp parallel for
    for(int i = 0; i < v.size(); i++){
        int tid = omp_get_thread_num(); cout << tid << endl;
        #pragma omp critical
        file << v[i] << "\n";
    }
    return 0;
}