#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include "graph.cpp"
#include "task.cpp"
#include "parse.h"

int main(int argc, char **argv) {
    // Parse command line arguments
    parseInput(argc, argv);
    
    // Read in graph data
    Graph graph;
    graph.read_graph(inputpath, headerpath);

    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Perform task: Advertisement
    // TODO: call function to perform task and output results
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
