#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;

struct Graph {
    int n, m;
    vector<int> offsets;
    vector<int> degrees;

    Graph(){
        n = 0;
        m = 0;
    }

    void read_graph(string inputpath, string headerpath){
        ifstream headerfile, inputfile;
        inputfile.open(inputpath, ios::binary);
        headerfile.open(headerpath, ios::binary);
        inputfile.read(reinterpret_cast<char*>(&n), 4);
        inputfile.read(reinterpret_cast<char*>(&m), 4);
        offsets.resize(n);
        degrees.resize(n);
        for(int i = 0; i < n; i++){
            int degree = 0;
            headerfile.read(reinterpret_cast<char*>(&offsets[i]), 4);
            inputfile.seekg(offsets[i] + 4);
            inputfile.read(reinterpret_cast<char*>(&degree), 4);
            degrees[i] = degree;
        }
        headerfile.close();
        inputfile.close();
    }

    void print_graph(string inputpath){
        ifstream inputfile;
        inputfile.open(inputpath, ios::binary);
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
        inputfile.close();
    }
};
