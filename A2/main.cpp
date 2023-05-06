#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <climits>
#include <algorithm>
#include <set>
#include <mpi.h>
#include <map>
#include <chrono>
#include "graph.h"
#include "task.h"
#include "parse.h"

using namespace chrono;

Graph graph;

bool cmp(int a, int b) {
    if (graph.degrees[a] != graph.degrees[b]) {
        return graph.degrees[a] < graph.degrees[b];
    }
    return a < b;
};

int root(int v, vector<int> &parent){
    if(parent[v] == v) return v;
    parent[v] = root(parent[v], parent);
    return parent[v];
}

void degsort(vector<int> &v){
    sort(v.begin(), v.end(), cmp);
}

int main(int argc, char **argv) {
    // Record the start time
    auto start = high_resolution_clock::now(), end = high_resolution_clock::now();

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Load world number and size
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Parse command line arguments
    Args args;
    args.parseInput(argc, argv);

    // Read in graph data
    if(world_rank == 0) graph.read_graph(args.inputpath, args.headerpath);

    // Send processor-map to other processes
    MPI_Bcast(&graph.n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    graph.offsets.resize(graph.n);
    graph.degrees.resize(graph.n);
    MPI_Bcast(graph.offsets.data(), graph.n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(graph.degrees.data(), graph.n, MPI_INT, 0, MPI_COMM_WORLD);

    // Define the adjacency list for each vertex in the graph
    map<int, set<int>> adjList;
    // Define the support set for each edge in the graph
    map<pair<int, int>, set<int>> supp;
    // Define the set of deletable vertices (vertices with degree < startk + 1)
    set<int> deletable;
    // Open the input file in binary mode
    ifstream inputFile(args.inputpath, ios::binary);
    // Iterate over vertices in the graph (in a round-robin fashion based on world rank and size)
    for(int vertex = world_rank; vertex < graph.n; vertex+=world_size){
        // Seek to the start of the vertex's neighbor list in the input file
        inputFile.seekg(graph.offsets[vertex] + 8);
        // Check if the degree of the current vertex is less than startk + 1
        if(graph.degrees[vertex] < args.startk + 1) deletable.insert(vertex);
        // Iterate over the neighbors of the current vertex
        for(int j = 0; j < graph.degrees[vertex]; j++){
            // Read the ID of the current neighbor from the input file
            int nbr = 0; inputFile.read(reinterpret_cast<char*>(&nbr), 4);
            // Add the neighbor ID to the adjacency list for the current vertex
            adjList[vertex].insert(nbr);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ofstream myfile(args.outputpath);
    // Create vectors to store the count and displacement arrays used in MPI_Alltoallv and the actual data sent
    vector<int> sendDispls(world_size), recvDispls(world_size), sendVertices;
    // Create a map that stores neighbors of vertices in the deletable set
    map<int, vector<int>> shareVertices;

    // Loop until all vertices that are deletable have been removed
    while(true){
        for(auto &v: deletable){
            graph.degrees[v] = -1; // Mark the vertex as deleted
            for(auto &nbr: adjList[v])
                shareVertices[nbr%world_size].push_back(nbr);
        }
        deletable.clear(); // Clear the deletable set for the next iteration

        // Create vectors to store the size of data to be sent/received 
        vector<int> sendSizes(world_size, 0), recvSizes(world_size, 0);

        // Fill the sendSizes and sendVertices vectors based on the shareVertices map
        for(auto &x: shareVertices){
            sendSizes[x.first] = x.second.size();
            sendVertices.insert(sendVertices.end(), x.second.begin(), x.second.end());
        } shareVertices.clear();

        // Exchange data between processes using MPI_Alltoallv
        MPI_Alltoall(sendSizes.data(), 1, MPI_INT, recvSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Calculate the total size of data to be received
        int recv_size_total = accumulate(recvSizes.begin(), recvSizes.end(), 0);

        // Create a vector to store the received data
        vector<int> recvVertices(recv_size_total);

        // Fill the send_displs and recv_displs arrays based on the size of data to be sent/received
        sendDispls[0] = 0; recvDispls[0] = 0;
        for (int i = 1; i < world_size; i++) {
            sendDispls[i] = sendDispls[i-1] + sendSizes[i-1];
            recvDispls[i] = recvDispls[i-1] + recvSizes[i-1];
        }

        // Exchange data between processes using MPI_Alltoallv
        MPI_Alltoallv(sendVertices.data(), sendSizes.data(), sendDispls.data(), MPI_INT, recvVertices.data(), recvSizes.data(), recvDispls.data(), MPI_INT, MPI_COMM_WORLD);
        sendVertices.clear();
        
        // Check if any received vertex should be marked as deletable and update the deletable set
        int flag = 1;
        for(auto &x: recvVertices){
            graph.degrees[x]--;
            if(graph.degrees[x] < args.startk + 1 && graph.degrees[x] > 0) {
                deletable.insert(x);
                flag = 0; 
            }
        }
        
        // Broadcast the flag to all processes and determine whether to continue iterating
        int flag_sum = 0;
        MPI_Allreduce(&flag, &flag_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(flag_sum == world_size) break;
    }

    // Reduce degrees to minimum value across processes
    MPI_Allreduce(MPI_IN_PLACE, graph.degrees.data(), graph.n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Remove nodes with negative degrees
    for(int i = world_rank; i < graph.n; i+=world_size){
        if(graph.degrees[i] < 0){
            adjList.erase(i);
        }
    }
    
    // Remove neighbors with negative degrees
    for(auto &x: adjList){
        int u = x.first;
        auto it = x.second.begin();
        while (it != x.second.end()) {
            int v = *it;
            if (graph.degrees[v] < 0) {
                it = x.second.erase(it);
            } else {
                if((graph.degrees[u] < graph.degrees[v]) || (graph.degrees[u] == graph.degrees[v] && u < v))
                    supp[{u, v}] = {};
                ++it;
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    int cnt = 0, cnt2 = 0; MPI_Status status;
    for(auto &x: adjList){
        int u_new = x.first;
        for (auto it = x.second.begin(); it != x.second.end(); ++it) {
            int v_new = *it;
            if((graph.degrees[u_new] > graph.degrees[v_new]) || (graph.degrees[u_new] == graph.degrees[v_new] && u_new > v_new))
                continue;
            for (auto jt = x.second.begin(); jt != x.second.end(); ++jt) {
                int w_new = *jt;
                int u = u_new, v = v_new, w = w_new;
                if((graph.degrees[u] > graph.degrees[w]) || (graph.degrees[u] == graph.degrees[w] && u > w))
                    continue;
                if((graph.degrees[v] > graph.degrees[w]) || (graph.degrees[v] == graph.degrees[w] && v > w))
                    continue;
                if(v == w) continue;
                int message[4] = {u, v, w, -1};
                int process = v % world_size;
                if(process == world_rank){
                    if(supp.find({v, w}) != supp.end()){
                        supp[{u, v}].insert(w);
                        supp[{u, w}].insert(v);
                        supp[{v, w}].insert(u);
                    }
                    continue;
                }
                MPI_Send(&message, 4, MPI_INT, process, 0, MPI_COMM_WORLD);
                int recv[4]; int flag;
                while(true){
                    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
                    if (!flag) break;
                    MPI_Recv(&recv, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                    if(recv[0] == -1){
                        cnt++; 
                    }
                    else if(recv[0] == -2){
                        cnt2++;
                    }
                    else if(recv[3] == -1){
                        if(supp.find({recv[1], recv[2]}) == supp.end()) continue;
                        recv[3] = 1; 
                        MPI_Send(&recv, 4, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                        supp[{recv[1], recv[2]}].insert(recv[0]);
                    }
                    else{
                        supp[{recv[0], recv[1]}].insert(recv[2]);
                        supp[{recv[0], recv[2]}].insert(recv[1]);
                    }break;
                }
            } 
        }
    }adjList.clear();
    
    int FINISH[4] = {-1, -1, -1, -1};
    for(int i = 0; i < world_size; i++){
        if(i != world_rank) MPI_Send(&FINISH, 4, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    
    while (true) {
        int recv[4];
        if(cnt == world_size - 1) break;
        MPI_Recv(&recv, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if(recv[0] == -1){
            cnt++; if(cnt == world_size - 1) break;
        }
        else if(recv[0] == -2){
            cnt2++;
        }
        else if(recv[3] == -1){
            if(supp.find({recv[1], recv[2]}) == supp.end()) continue;
            recv[3] = 1; MPI_Send(&recv, 4, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            supp[{recv[1], recv[2]}].insert(recv[0]);
        }
        else{
            supp[{recv[0], recv[1]}].insert(recv[2]);
            supp[{recv[0], recv[2]}].insert(recv[1]);
        }
    }

    FINISH[0] = -2; 
    for(int i = 0; i < world_size; i++){
        if(i != world_rank) MPI_Send(&FINISH, 4, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    while (true) {
        int recv[4];
        if(cnt2 == world_size - 1) break;
        MPI_Recv(&recv, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if(recv[0] == -1){ 
            cnt++; if(cnt == world_size - 1) break;
        }
        else if(recv[0] == -2){
            cnt2++;
        }
        else if(recv[3] == -1){
            if(supp.find({recv[1], recv[2]}) == supp.end()) continue;
            recv[3] = 1; MPI_Send(&recv, 4, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            supp[{recv[1], recv[2]}].insert(recv[0]);
        }
        else{
            supp[{recv[0], recv[1]}].insert(recv[2]);
            supp[{recv[0], recv[2]}].insert(recv[1]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for(int k = args.startk; k <= args.endk; k++){
        set<pair<int, int>> del;
        for(auto &x: supp){ 
            if(x.second.size() < k) del.insert(x.first);
        }

        // Create vectors to store the count and displacement arrays used in MPI_Alltoallv and the actual data sent
        vector<int> sendEdges;
        // Create a map that stores neighbors of vertices in the deletable set
        map<int, vector<int>> shareEdges;
        // Loop until all vertices that are deletable have been removed
        while(true){
            for(auto &e: del){
                for(auto &x: supp[e]){
                    int u = e.first, v = e.second, w = x;
                    int min_vertex, max_vertex;
                    min_vertex = u, max_vertex = w;
                    if((graph.degrees[u] > graph.degrees[w]) || (graph.degrees[u] == graph.degrees[w] && u > w)){
                        min_vertex = w; max_vertex = u;
                    }
                    shareEdges[min_vertex%world_size].push_back(min_vertex);
                    shareEdges[min_vertex%world_size].push_back(max_vertex);
                    shareEdges[min_vertex%world_size].push_back(v);
                    min_vertex = v, max_vertex = w;
                    if((graph.degrees[v] > graph.degrees[w]) || (graph.degrees[v] == graph.degrees[w] && v > w)){
                        min_vertex = w; max_vertex = v;
                    }
                    shareEdges[min_vertex%world_size].push_back(min_vertex);
                    shareEdges[min_vertex%world_size].push_back(max_vertex);
                    shareEdges[min_vertex%world_size].push_back(u);
                }
                supp.erase(e);

            }
            del.clear(); // Clear the deletable set for the next iteration

            // Create vectors to store the size of data to be sent/received 
            vector<int> sendSizes(world_size, 0), recvSizes(world_size, 0);

            // Fill the sendSizes and sendVertices vectors based on the shareEdges map
            for(auto &x: shareEdges){
                sendSizes[x.first] = x.second.size();
                sendEdges.insert(sendEdges.end(), x.second.begin(), x.second.end());
            } shareEdges.clear();

            // Exchange data between processes using MPI_Alltoallv
            MPI_Alltoall(sendSizes.data(), 1, MPI_INT, recvSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // Calculate the total size of data to be received
            int recv_size_total = accumulate(recvSizes.begin(), recvSizes.end(), 0);

            // Create a vector to store the received data
            vector<int> recvEdges(recv_size_total);

            // Fill the send_displs and recv_displs arrays based on the size of data to be sent/received
            sendDispls[0] = 0; recvDispls[0] = 0;
            for (int i = 1; i < world_size; i++) {
                sendDispls[i] = sendDispls[i-1] + sendSizes[i-1];
                recvDispls[i] = recvDispls[i-1] + recvSizes[i-1];
            }

            // Exchange data between processes using MPI_Alltoallv
            MPI_Alltoallv(sendEdges.data(), sendSizes.data(), sendDispls.data(), MPI_INT, recvEdges.data(), recvSizes.data(), recvDispls.data(), MPI_INT, MPI_COMM_WORLD);
            sendEdges.clear();
            
            // Check if any received vertex should be marked as deletable and update the deletable set
            int flag = 1;
            for(int i = 0; i < recvEdges.size(); i += 3){
                int u = recvEdges[i], v = recvEdges[i+1], delVertex = recvEdges[i+2];
                supp[{u, v}].erase(delVertex);
                if(supp[{u, v}].size() < k){
                    del.insert({u, v});
                    flag = 0;
                }
            }
            
            // Broadcast the flag to all processes and determine whether to continue iterating
            int flag_sum = 0;
            MPI_Allreduce(&flag, &flag_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if(flag_sum == world_size) break;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        for(auto &x: supp){
            adjList[x.first.first].insert(x.first.second);
            adjList[x.first.second].insert(x.first.first);
        }
        int size = supp.size();
        MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(world_rank == world_size - 1){
            if(size){
                myfile << 1 << endl;
            }else{
                while(k<=args.endk){
                    myfile << 0 << endl;
                    k++;
                }break;
            }
        }
        if(size == 0) break;
        if(args.verbose == 0) continue;
        vector<int> parent(graph.n), rank(graph.n, 0);
        for(int i = 0; i < graph.n; i++){
            parent[i] = i;
        }
        for(int i = 0; i < world_size; i++){
            if(i == world_rank){
                if(i) MPI_Recv(parent.data(), graph.n, MPI_INT, i-1, 0, MPI_COMM_WORLD, &status);
                for(auto &x: supp){
                    int u = x.first.first, v = x.first.second;
                    int root_u = root(u, parent), root_v = root(v, parent);
                    if(rank[root_u] < rank[root_v]){
                        parent[root_u] = root_v;
                    }
                    else if(rank[root_u] > rank[root_v]){
                        parent[root_v] = root_u;
                    }
                    else{
                        parent[root_u] = root_v;
                        rank[root_v]++;
                    }
                }
                if(world_rank < world_size - 1)
                    MPI_Send(parent.data(), graph.n, MPI_INT, i+1, 0, MPI_COMM_WORLD);
            }
        }
        map<int, vector<int>> components;
        if(world_rank == world_size - 1){
            for(int i = 0; i < graph.n; i++){
                if(graph.degrees[i] > 0)
                    components[root(i, parent)].push_back(i);
            }
            int number_of_components = 0;
            for(auto &x: components){
                if(x.second.size() < 2) continue;
                number_of_components++;
            }
            myfile << number_of_components << endl;
            for(auto &x: components){
                if(x.second.size() < 2) continue;
                for(int i = 0; i < x.second.size(); i++){
                    myfile << x.second[i];
                    if(i < x.second.size() - 1) myfile << " ";
                }
                myfile << endl;
            }
        }
    }

    // Finalize MPI
    MPI_Finalize();

    // Record the end time and print the time taken
    end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    // if(world_rank == 0) cout << "Total Time:" << duration.count() << endl;
    return 0;
}
