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
#include <omp.h>
#include "graph.h"
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
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_support);

    // Set the number of threads to be used
    // int num_threads = 4;
    // omp_set_num_threads(num_threads);

    // Load world number and size
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Parse command line arguments
    Args args;
    args.parseInput(argc, argv);
    if(args.taskid == 2){
        args.startk = args.endk;
    }

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
    vector<vector<int>> adjListCopy(graph.n/world_size + 1);
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
        if(graph.degrees[vertex] < args.startk + 1) {
            deletable.insert(vertex);
        }
        // Iterate over the neighbors of the current vertex
        for(int j = 0; j < graph.degrees[vertex]; j++){
            // Read the ID of the current neighbor from the input file
            int nbr = 0; inputFile.read(reinterpret_cast<char*>(&nbr), 4);
            // Add the neighbor ID to the adjacency list for the current vertex
            adjList[vertex].insert(nbr);
            adjListCopy[vertex/world_size].push_back(nbr);
        }
    }
    // adjListCopy = adjList;
    // if(world_rank == 0) cout << "adjList Loaded.\n";
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

    // if(world_rank == 0) cout << "Prefilter Complete.\n";
    MPI_Barrier(MPI_COMM_WORLD);

    int cnt = 0, cnt2 = 0;

    // #pragma omp parallel for schedule(dynamic)
    for(int vertex = world_rank; vertex < graph.n; vertex+=world_size){

        // Check if vertex is valid, continue if not
        if(graph.degrees[vertex] < 0) continue;

        MPI_Status status;
        int u_new = vertex;

        // Loop through each adjacent vertex of the current vertex
        for (auto it = adjList[vertex].begin(); it != adjList[vertex].end(); ++it) {
            int v_new = *it;

            // Check if edge is invalid, continue if so
            if((graph.degrees[u_new] > graph.degrees[v_new]) || (graph.degrees[u_new] == graph.degrees[v_new] && u_new > v_new)){
                continue;
            }

            // Loop through each adjacent vertex of the current vertex again
            for (auto jt = adjList[vertex].begin(); jt != adjList[vertex].end(); ++jt) {
                int w_new = *jt;
                int u = u_new, v = v_new, w = w_new;

                // Check if edge is invalid, continue if so
                if((graph.degrees[u] > graph.degrees[w]) || (graph.degrees[u] == graph.degrees[w] && u > w))
                    continue;
                if((graph.degrees[v] > graph.degrees[w]) || (graph.degrees[v] == graph.degrees[w] && v > w))
                    continue;
                if(v == w)
                    continue;

                // Create message to be sent to another process
                int message[4] = {u, v, w, -1};
                int process = v % world_size;

                // If destination process is current process, update support set directly
                if(process == world_rank){
                    // omp_set_lock(&supplock);
                    if(supp.find({v, w}) != supp.end()){
                        supp[{u, v}].insert(w);
                        supp[{u, w}].insert(v);
                        supp[{v, w}].insert(u);
                    }
                    // omp_unset_lock(&supplock);
                    continue;
                }

                // Send message to destination process
                MPI_Send(&message, 4, MPI_INT, process, 0, MPI_COMM_WORLD);

                int recv[4]; int flag;
                // int tid = omp_get_thread_num();
                // Process responses other processes
                // if(tid == 0)
                    while(true){
                        // Probe to check if there are messages in the buffer, and load them
                        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
                        if (flag)
                            MPI_Recv(&recv, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

                        // Break the loop if buffer is empty
                        if (!flag) break;

                        // If response message indicates FINISH-1, update count of FINISH-1
                        if(recv[0] == -1){
                            cnt++; 
                        }
                        // If response message indicates FINISH-2, update count of FINISH-2
                        else if(recv[0] == -2){
                            cnt2++;
                        }
                        // If response message indicates query, check and respond back
                        else if(recv[3] == -1){
                            bool ifsend = false;
                            // Check if edge exists in supp and update supp
                            // omp_set_lock(&supplock);
                            if(supp.find({recv[1], recv[2]}) != supp.end()){
                                recv[3] = 1; 
                                ifsend = true;
                                supp[{recv[1], recv[2]}].insert(recv[0]);
                            }
                            // omp_unset_lock(&supplock);
                            // Send back a message with the triangle detected
                            if(ifsend){
                                MPI_Send(&recv, 4, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                            }
                        }
                        // If response message indicates an answer back, triangle detected
                        else{
                            // omp_set_lock(&supplock);
                            supp[{recv[0], recv[1]}].insert(recv[2]);
                            supp[{recv[0], recv[2]}].insert(recv[1]);
                            // omp_unset_lock(&supplock);
                        }
                    }
            } 
        }
    }

    // omp_destroy_lock(&supplock);
    // omp_destroy_lock(&sendlock);
    adjList.clear();

    // if(world_rank == 0) cout << "Triangle Enumeration - For Loop Complete.\n";

    MPI_Status status;

    int FINISH[4] = {-1, -1, -1, -1};
    // send 'FINISH' message to all processes except the current process
    for(int i = 0; i < world_size; i++){
        if(i != world_rank) MPI_Send(&FINISH, 4, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    while (true) {
        int recv[4];
        // check if all processes have sent 'FINISH' message
        if(cnt == world_size - 1) break;
        // receive message from any process
        MPI_Recv(&recv, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        // check if the received message is a 'FINISH' message
        if(recv[0] == -1){
            cnt++; 
            // check if all processes have sent 'FINISH' message
            if(cnt == world_size - 1) break;
        }
        // check if the received message is a 'SUPP' message
        else if(recv[0] == -2){
            cnt2++;
        }
        else if(recv[3] == -1){
            // check if the edge (v,w) is in the 'supp' data structure
            if(supp.find({recv[1], recv[2]}) == supp.end()) continue;
            recv[3] = 1; 
            // send 'SUPP' message to the process that owns vertex v
            MPI_Send(&recv, 4, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            // insert edge (u,v) into the 'supp' data structure
            supp[{recv[1], recv[2]}].insert(recv[0]);
        }
        else{
            // insert edge (u,w) and (v,w) into the 'supp' data structure
            supp[{recv[0], recv[1]}].insert(recv[2]);
            supp[{recv[0], recv[2]}].insert(recv[1]);
        }
    }

    // if(world_rank == 0) cout << "Triangle Enumeration - Finish 1 Complete.\n";

    FINISH[0] = -2; 
    // send 'FINISH' message to all processes except the current process
    for(int i = 0; i < world_size; i++){
        if(i != world_rank) MPI_Send(&FINISH, 4, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    while (true) {
        int recv[4];
        // check if all processes have sent 'SUPP' message
        if(cnt2 == world_size - 1) break;
        // receive message from any process
        MPI_Recv(&recv, 4, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        // check if the received message is a 'FINISH' message
        if(recv[0] == -1){ 
            cnt++; 
            // check if all processes have sent 'SUPP' message
            if(cnt2 == world_size - 1) break;
        }
        // check if the received message is a 'SUPP' message
        else if(recv[0] == -2){
            cnt2++;
        }
        else if(recv[3] == -1){
            // check if the edge (v,w) is in the 'supp' data structure
            if(supp.find({recv[1], recv[2]}) == supp.end()) continue;
            recv[3] = 1; 
            // send 'SUPP' message to the process that owns vertex v
            MPI_Send(&recv, 4, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            // insert edge (u,v) into the 'supp' data structure
            supp[{recv[1], recv[2]}].insert(recv[0]);
        }
        else{
            // insert edge (u,w) and (v,w) into the 'supp' data structure
            supp[{recv[0], recv[1]}].insert(recv[2]);
            supp[{recv[0], recv[2]}].insert(recv[1]);
        }
    }

    // if(world_rank == 0) cout << "Triangle Enumeration - Finish 2 Complete.\n";
    MPI_Barrier(MPI_COMM_WORLD);

    map<int, vector<int>> components;
    vector<int> parent(graph.n), roots(graph.n, 0);

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
                // iterate over all edges in 'del'
                for(auto &x: supp[e]){
                    int u = e.first, v = e.second, w = x;
                    int min_vertex, max_vertex;
                    min_vertex = u, max_vertex = w;
                    // check if degree of vertex 'u' is greater than degree of vertex 'w'
                    // or if degrees are equal and 'u' is greater than 'w'
                    if((graph.degrees[u] > graph.degrees[w]) || (graph.degrees[u] == graph.degrees[w] && u > w)){
                        min_vertex = w; max_vertex = u;
                    }
                    // add edge (min_vertex, max_vertex, v) to the 'shareEdges' data structure
                    shareEdges[min_vertex%world_size].push_back(min_vertex);
                    shareEdges[min_vertex%world_size].push_back(max_vertex);
                    shareEdges[min_vertex%world_size].push_back(v);
                    min_vertex = v, max_vertex = w;
                    // check if degree of vertex 'v' is greater than degree of vertex 'w'
                    // or if degrees are equal and 'v' is greater than 'w'
                    if((graph.degrees[v] > graph.degrees[w]) || (graph.degrees[v] == graph.degrees[w] && v > w)){
                        min_vertex = w; max_vertex = v;
                    }
                    // add edge (min_vertex, max_vertex, u) to the 'shareEdges' data structure
                    shareEdges[min_vertex%world_size].push_back(min_vertex);
                    shareEdges[min_vertex%world_size].push_back(max_vertex);
                    shareEdges[min_vertex%world_size].push_back(u);
                }
                // remove edge 'e' from the 'supp' data structure
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

        // update adjList using supp
        for(auto &x: supp){
            adjList[x.first.first].insert(x.first.second);
            adjList[x.first.second].insert(x.first.first);
        }
        int size = supp.size();
        MPI_Allreduce(MPI_IN_PLACE, &size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(args.taskid == 1 && world_rank == 0){
            if(size){
                myfile << 1;
                if(args.verbose == 1) myfile << endl;
                if(args.verbose == 0 && k != args.endk) myfile << " ";
            }else{
                while(k<=args.endk){
                    myfile << 0;
                    if(args.verbose == 1) myfile << endl;
                    if(args.verbose == 0 && k != args.endk) myfile << " ";
                    k++;
                }break;
            }
        }
        if(args.taskid == 1 && size == 0) break;
        if(args.taskid == 1 && args.verbose == 0) continue;
        // Rank for weighted Union-Find
        vector<int> rank(graph.n, 0), parent2(graph.n);
        // Initialize parents array for Union-Find
        for(int i = 0; i < graph.n; i++){
            parent[i] = i;
        }
        // Finding local parents array
        // iterate over all edges in 'supp'
        for(auto &x: supp){
            int u = x.first.first, v = x.first.second;
            // find root of vertex 'u' and vertex 'v'
            int root_u = root(u, parent), root_v = root(v, parent);
            // if rank of root of 'u' is less than rank of root of 'v', set parent of root of 'u' to root of 'v'
            if(rank[root_u] < rank[root_v]){
                parent[root_u] = root_v;
            }
            // if rank of root of 'u' is greater than rank of root of 'v', set parent of root of 'v' to root of 'u'
            else if(rank[root_u] > rank[root_v]){
                parent[root_v] = root_u;
            }
            // if ranks are equal, set parent of root of 'u' to root of 'v' and increment rank of root of 'v'
            else{
                parent[root_u] = root_v;
                rank[root_v]++;
            }
        }
        // Distributed Union-Find
        for(int step = 2; step < 2*world_size; step *= 2){
            for(int i = 0; i < world_size - step / 2; i += step){
                // merge i and i + step/2
                if(world_rank == i + step/2){
                    MPI_Send(parent.data(), graph.n, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
                if(world_rank == i){
                    MPI_Recv(parent2.data(), graph.n, MPI_INT, i + step/2, 0, MPI_COMM_WORLD, &status);
                    for(int j = 0; j < graph.n; j++){
                        int root_u = root(j, parent), root_v = root(parent2[j], parent);
                        if(root_u != root_v){
                            // if rank of root of 'u' is less than rank of root of 'v'
                            if(rank[root_u] < rank[root_v]){
                                parent[root_u] = root_v;
                            }
                            // if rank of root of 'u' is greater than rank of root of 'v'
                            else if(rank[root_u] > rank[root_v]){
                                parent[root_v] = root_u;
                            }
                            // if ranks are equal
                            else{
                                parent[root_u] = root_v;
                                rank[root_v]++;
                            }
                        }
                    }
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        if(world_rank == 0){
            for(int i = 0; i < graph.n; i++){
                if(graph.degrees[i] > 0)
                    components[root(i, parent)].push_back(i);
            }
            if(args.taskid == 2){
                for(auto &x: components){
                    if(x.second.size() >= 2)
                        roots[x.first] = 1;
                }
            }
            if(args.taskid == 1){
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
        if(args.taskid == 1) components.clear();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // if(world_rank == 0) cout << "Components Found.\n";

    if(args.taskid == 2){
        // to use same "correct" parent vector to find root, only the last process holds right now
        MPI_Bcast(parent.data(), graph.n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(roots.data(), graph.n, MPI_INT, 0, MPI_COMM_WORLD);
        // map<int, set<int>> egoNetwork;
        vector<set<int>> egoNetwork(graph.n/world_size + 1);
        vector<int> influencers(graph.n, 0);
        #pragma omp parallel for schedule(dynamic)
        for(int i = world_rank; i < graph.n; i+=world_size){
            int givenRoot = root(i, parent);
            if(roots[givenRoot])
                egoNetwork[i/world_size].insert(givenRoot);
            for(auto &x: adjListCopy[i/world_size]){
                givenRoot = root(x, parent);
                if(roots[givenRoot])
                    egoNetwork[i/world_size].insert(givenRoot);
            }
            if(egoNetwork[i/world_size].size() >= args.p) {
                influencers[i] = 1;
            }
            else egoNetwork[i/world_size].clear();
        }
        
        if (world_rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, influencers.data(), graph.n, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Reduce(influencers.data(), influencers.data(), graph.n, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        }

        if(world_rank == 0){
            int num_infl = accumulate(influencers.begin(), influencers.end(), 0);
            myfile << num_infl << endl;
            if(args.verbose == 0){
                for(int i = 0; i < graph.n; i++){
                    if(influencers[i])
                        myfile << i << " ";
                }
            }
            if(args.verbose == 1){
                int num_infl_mine = 0;
                for(int i = world_rank; i < graph.n; i+=world_size){
                    if(influencers[i]){
                        myfile << i << endl;
                        num_infl_mine++;
                        for(auto &x: egoNetwork[i/world_size]){
                            for(auto &y: components[x]){
                                myfile << y << " ";
                            }
                        }myfile << endl;
                    }
                }
                for(int i = 0; i < num_infl - num_infl_mine; i++){
                    int metaData[2];
                    MPI_Recv(&metaData, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                    vector<int> egoData(metaData[1]);
                    myfile << metaData[0] << endl;
                    int src = status.MPI_SOURCE;
                    MPI_Recv(egoData.data(), metaData[1], MPI_INT, src, 0, MPI_COMM_WORLD, &status);
                    for(auto &x: egoData){
                        for(auto &y: components[x]){
                            myfile << y << " ";
                        }
                    }myfile << endl;
                }
            }
        }else{
            if(args.verbose == 1){
                for(int i = world_rank; i < graph.n; i+=world_size){
                    if(influencers[i]){
                        int datasize = egoNetwork[i/world_size].size();
                        int metaData[2] = {i, datasize};
                        vector<int> egoData(egoNetwork[i/world_size].begin(), egoNetwork[i/world_size].end());
                        MPI_Send(&metaData, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
                        MPI_Send(egoData.data(), metaData[1], MPI_INT, 0, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }
    }

    // if(world_rank == 0) cout << "All tasks finished.\n";

    // Finalize MPI
    MPI_Finalize();

    // Record the end time and print the time taken
    end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    // if(world_rank == 0) cout << "Total Time:" << duration.count() << endl;
    return 0;
}