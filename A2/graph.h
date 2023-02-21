#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

struct Graph {

    int n, m;
    ifstream inputfile, headerfile;
    vector<int> offsets;

    Graph(){
        n = 0;
        m = 0;
    }

    void read_graph(string inputpath, string headerpath){
        inputfile.open(inputpath, ios::binary);
        headerfile.open(headerpath, ios::binary);
        inputfile.read(reinterpret_cast<char*>(&n), 4);
        inputfile.read(reinterpret_cast<char*>(&m), 4);
        offsets.resize(n);
        for(auto &offset: offsets)
            headerfile.read(reinterpret_cast<char*>(&offset), 4);
        headerfile.close();
    }

    void print_graph(){
        cout << "n = " << n << endl;
        cout << "m = " << m << endl;
        for (int i = 0; i < n; i++) {
            inputfile.seekg(offsets[i]);
            int node_id, deg;
            inputfile.read(reinterpret_cast<char*>(&node_id), 4);
            inputfile.read(reinterpret_cast<char*>(&deg), 4);
            cout << "Node ID: " << node_id << ", Degree: " << deg << endl;
            for(int j = 0; j < deg; j++){
                int nbr = 0;
                inputfile.read(reinterpret_cast<char*>(&nbr), 4);
                cout << nbr << " ";
            }cout << endl;
        }
    }
};
