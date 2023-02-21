# module load mpi
main:
	mpic++ -std=c++11 main.cpp -o main.out
	mpirun -np 1 ./main.out --taskid=1 --inputpath=/home/cse/btech/cs1190465/COL380/A2/test0/test-input-0.gra --headerpath=/home/cse/btech/cs1190465/COL380/A2/test0/test-header-0.dat --outputpath=test0/output0.txt --startk=1 --endk=12 --verbose=1
	
rough:
	mpic++ -std=c++11 rough.cpp -o rough.out
	mpirun -np 4 ./rough.out

clean:
	rm *.out
