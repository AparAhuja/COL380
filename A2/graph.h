#include <vector>
#include <string>

class Graph {
    public:
        Graph();
        void read_graph(std::string inputpath, std::string headerpath);
    private:
        std::vector<std::vector<int>> adj_list;
        int num_nodes;
};
