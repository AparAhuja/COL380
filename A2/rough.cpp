#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv) {
    int x = 3, y = 
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    cout << world_size << " " << world_rank << endl;
    int send_data = 42;
    int recv_data = -1;
    if (world_rank == 0) {
        MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Received data: " << recv_data << std::endl;
    }
    MPI_Finalize();
    return 0;
}

