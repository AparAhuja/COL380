#include <vector>
#include <string>
#include <iostream>

struct Task {
    Graph& graph;
    int startk, endk;
    bool verbose;

    Task(Graph& g, int startk, int endk, bool verbose): graph(g), startk(startk), endk(endk), verbose(verbose) {}

    void run(){

    }
        
};
