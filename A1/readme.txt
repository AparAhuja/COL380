Number of approaches: 5

Approach 1
1:Idea: Baseline - Read Matrix from the Input File and store as vector<vector<blocks>> rowBlocks. Create tasks to sort rows of the resulting vector. Square the matrix by creating tasks on the outermost for-loop (approach1.cpp: line 245) that traverses rowBlocks and store the answer in result vector. Write the result vector to the Output File.
1:Why attempted: For base results and a correct code to start with. The code has baseline paralellization only without any complicated pipeline. This code was also used for verification of correctness by setting threads to 1 / removing paralellization. Also implemented is the improved order of for-loops (approach1.cpp: line 227) in matrix-multiplication for optimised cache utilisation. Note that the time/speed might change based on the state of the css cluster, based on the load and jobs queue. The max-time cap on each run was 5 minutes during my job submissions.
1:Results (Specify speedup): 79488ms (read + square + write) with n = 2000000, m = 100, k = 2^15, 8 cores/threads. This serves as the baseline.
1:Drawbacks: Memory required is high. The result matrix takes extra memory.

Approach 2
2:Idea: Read and Write stay the same as approach 1. We square the matrix by creating tasks on the middle for-loop (approach2.cpp: line 250) that traverses rowBlocks and store the answer in result vector.
2:Why attempted: The tasks created using the middle for-loop are shorter/smaller but more in number. There is a trade-off between number of tasks vs time taken by each task. This was tried to reduce the task size but increase their quantity. This direction of thought is tested again in the next approach where the task size is even smaller.
2:Results (Specify speedup): 209679ms (read + square + write) with n = 2000000, m = 100, k = 2^15, 8 cores/threads. This is 2.6x slower than approach 1, the speedup is 0.38x.
2:Drawbacks: Memory required is high. The result matrix takes extra memory. Overhead for task creation and maintaining the task queue is very high. The time taken increased by 2.6x times approach 1.

Approach 3
3:Idea: Read and Write stay the same as approach 1. We square the matrix by creating tasks on the outermost for-loop (approach3.cpp: line 231) that multiplies the BLOCK MATRICES and store the answer in result vector.
3:Why attempted: As mentioned above this is another test for the trade-off between number of tasks vs time taken by each task. Further, the block matrix_mult function takes the most time in the code, which was tested using perf. So the idea was to speed up the bottleneck.
3:Results (Specify speedup): 149336ms (read + square + write) with n = 2000000, m = 100, k = 2^15, 8 cores/threads. This is 1.9x slower than approach 1, the speedup is 0.53x.
3:Drawbacks: Memory required is high. The result matrix takes extra memory. Overhead for task creation and maintaining the task queue is lower than approach 2. But the time taken increased by 1.9 times approach 1. Still this approach performed better than approach 2, we will try to mix approach 1 and 3 together next.

Approach 4
4:Idea: Read and Write stay the same as approach 1. We square the matrix by creating tasks on the outermost for-loop (approach4.cpp: line 247) that traverses rowBlocks and also the outermost for-loop (approach4.cpp: line 230) that multiplies the BLOCK MATRICES and store the answer in result vector. There was also a omp critical section to count the number of non-zero blocks in the output. I figured removing it from the loops might speed up the code. 
4:Why attempted: As mentioned above, we try to mix the advantages of approach 1 and approach 3. 
4:Results (Specify speedup): 73014ms (read + square + write) with n = 2000000, m = 100, k = 2^15, 8 cores/threads. This is 1.1x faster than approach 1, the speedup is 1.1x.
4:Drawbacks: Memory required is still high due to the result matrix, this can be mitigated by writing in parallel, as soon as the blocks are ready. 

Final Approach: Approach 5
5:Idea: Read stays the same as approach 1. We square the matrix by creating tasks on the outermost for-loop (approach4.cpp: line 247) that traverses rowBlocks. We create a child-task to write the block as soon as the result is ready. 
5:Why attempted: As mentioned above, we try to tackle the major memory drawback. This approach makes the write parallelized and reduces the memory requirement to half. 
5:Results (Specify speedup): 74124ms (read + {square + write}) with n = 2000000, m = 100, k = 2^15, 8 cores/threads. This is slightly, around 1.1x faster than approach 1, the speedup is 1.1x. This takes time similar to approach 4 but with a lower memory requirement.
5:Drawbacks: The BLOCK matrix multiplication algorithm is naive. We can implement strassen or other cache efficient block based multiplication schemes. This will speed up for larger values of m, however for small m this might not be optimal due to overhead costs of other algorithms. 

Final scalability analysis
Non-0 input blocks,Non-0 output blocks,2 cores,4 cores,8 cores,16 cores
1024,2048,4117.24,2050.07,1183.04,1070.89
32768,71414,57931.88,29014.51,15812.32,12235.75

Note: We can see the the code scales well on both axes for the given data points. *css wouldn't allow me to run tasks longer than 5 minutes. I wasn't able to run for larger data points(2^20 and 2^25). 

_____________________________________________

Structure Descriptions:

1. 'block': Used to store a block of a sparse matrix. The Block structure has three members: "row" and "col" are integers that store the row and column indices of the block, respectively. "data" is a vector of integers that stores the non-zero blocks.

Function Descriptions:

1. 'readMatrix': This function reads a sparse matrix stored in a binary file. It takes the filename, the number of rows, the block size, and a vector of blocks as input. It also sets a flag indicating if the input matrix is in out-matrix-format.
2. 'compareColumn': This function compares two blocks based on their column index. It is used in the readMatrix function to sort the blocks in each row based on their column index.
3. 'writeMatrix': This function writes a block sparse matrix to a binary file. It takes the filename, the number of rows, the number of columns, the number of blocks, a vector of blocks, and a flag indicating if it is writing an out-matrix-format matrix.
4. 'matrix_mult': This function computes the result of a block-matrix multiplication. It takes the output vector, two input vectors, and the block size as input.
5. 'squareMatrix': This function computes the square of a block sparse matrix. It takes the input matrix, the output matrix, the number of rows, the block size, and the number of blocks as input. It uses OpenMP to parallelize computation.
6. 'squareMatrixWrapper': This function reads a matrix from an input file, computes its square matrix, and writes the resulting matrix to an output file. It also measures the time taken to perform these operations and prints it to the console.
7. 'compareOutputFiles': This function takes as input two filenames representing binary files that store information about two sparse matrices, and then reads and compares the data in these files. The function first reads and compares the n, m, and k values, which specify the number of rows and columns of the matrices, and the number of blocks in the block representation of the matrices. If any of these values are different in the two files, the function prints an error message and returns. If the values are the same, the function proceeds to read the block representation of the matrices from the files and stores them in row-wise order in two vectors of vectors called rowBlocks1 and rowBlocks2. Finally, the function calls the 'compareMatrices' function to compare the two matrices stored in rowBlocks1 and rowBlocks2. If the matrices are equal, the function prints a success message.
8. 'compareMatrices': This function takes in two matrices represented as vectors of vectors of Block objects, and compares them element-wise. It first checks if the size of the two matrices is equal, and returns an error message if they differ in size. Then, for each element of each matrix, it checks if the row, col, and data values are equal. If the matrices differ, it prints out an error message indicating the row and block number where the difference occurred, as well as the values of the corresponding blocks in both matrices. If the function completes without encountering a difference between the two matrices, it prints out a success message.