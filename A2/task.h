#include <vector>
#include <string>

class Task {
    public:
        Task(Graph& g, int startk, int endk, bool verbose);
        void run();
    private:
        Graph& graph;
        int startk, endk;
        bool verbose;
};
