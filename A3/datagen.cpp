#include <iostream>
#include <string.h>
#include <fstream>
#include <vector>
#include <set>

using namespace std; 

void dataGen(string graphfile_name, string textfile_name, string headerfile_name){
    fstream graphfile(graphfile_name, ios::binary | ios::out);
    fstream textfile(textfile_name, ios::out);
    fstream headerfile(headerfile_name, ios::binary | ios::out);
    
    int n = 10, curr_cntr = 0;
    set<pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {3, 4}, {2, 5}, {2, 9}, {8, 9}, {7, 8}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {4, 9}};
    int m = edges.size();

    graphfile.write(reinterpret_cast<const char*>(&n), 4); curr_cntr += 4;
    graphfile.write(reinterpret_cast<const char*>(&m), 4); curr_cntr += 4;
    textfile << n << " " << m << endl;

    vector<vector<int>> adj(n);
    for (auto edge : edges) {
        int u = edge.first;
        int v = edge.second;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    for (int i = 0; i < n; i++) {
        int deg = adj[i].size();
        headerfile.write(reinterpret_cast<const char*>(&curr_cntr), 4);
        graphfile.write(reinterpret_cast<const char*>(&i), 4); curr_cntr += 4;
        graphfile.write(reinterpret_cast<const char*>(&deg), 4); curr_cntr += 4;
        textfile << i << " " << deg << " ";
        for (int j = 0; j < deg; j++) {
            graphfile.write(reinterpret_cast<const char*>(&adj[i][j]), 4); curr_cntr += 4;
            textfile << adj[i][j] << " ";
        } textfile << endl;
    }

    graphfile.close();
    textfile.close();
    headerfile.close();
}

int main(int argc, char** argv) {
    string testNumber = "0";
    string myFolder = "mytest" + testNumber + "/";
    string graphfile = myFolder + "test-input-" + testNumber + ".gra";
    string textfile = myFolder + "test-input-" + testNumber + ".txt";
    string headerfile = myFolder + "test-header-" + testNumber + ".dat";
    dataGen(graphfile, textfile, headerfile);
    return 0;
}