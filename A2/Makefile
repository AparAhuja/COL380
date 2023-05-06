# test_number = 8
# no_of_proc = 8
# startk = 2
# endk = 6

main:
	mpic++ -std=c++17 -O3 -march=native main.cpp -o a2
# mpirun -np $(no_of_proc) ./main.out --taskid=1 --inputpath=test$(test_number)/test-input-$(test_number).gra --headerpath=test$(test_number)/test-header-$(test_number).dat --outputpath=outputs/myoutput.txt --startk=$(startk) --endk=$(endk) --verbose=0
# mpirun -np 4 ./a2 --taskid=1 --inputpath=test2/test-input-2.gra --headerpath=test2/test-header-2.dat --outputpath=outputs/myoutput.txt --startk=1 --endk=10 --verbose=0

data:
	g++ -std=c++17 datagen.cpp -o datagen.out
	./datagen.out

clean:
	rm *.out
