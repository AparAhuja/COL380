no_of_proc = 1
taskid = 2
test_number = 3
startk = 1
endk = 3
verbose = 1
p = 4

main:
	mpic++ -std=c++17 -O3 -fopenmp -march=native main.cpp -o a3

run:
	mpirun -np $(no_of_proc) ./a3 --taskid=$(taskid) --inputpath=test$(test_number)/test-input-$(test_number).gra --headerpath=test$(test_number)/test-header-$(test_number).dat --outputpath=test$(test_number)/myoutput.txt --startk=$(startk) --endk=$(endk) --verbose=$(verbose) --p=$(p)

data:
	g++ -std=c++17 datagen.cpp -o datagen.out
	./datagen.out

clean:
	rm *.out
