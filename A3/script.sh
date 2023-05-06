#!/bin/sh

taskid=2
test_number=3
startk=1
endk=3
verbose=1
p=4

nthreads=(1 2 4 8)
no_of_proc=(1 2 4 8)

echo "inputpath=test$test_number/test-input-$test_number.gra headerpath=test$test_number/test-header-$test_number.dat outputpath=test$test_number/myoutput.txt" > summary.txt
echo "taskid=$taskid startk=$startk endk=$endk verbose=$verbose p=$p" >> summary.txt

echo "--------------------" >> summary.txt

echo "#threads,#procs,#time" >> summary.txt

mpic++ -std=c++17 -O3 -fopenmp -march=native main.cpp -o a3

for np in "${no_of_proc[@]}"; do
    for nt in "${nthreads[@]}"; do
        export OMP_NUM_THREADS=$nt
        cmd="mpirun -np $np ./a3 --taskid=$taskid --inputpath=test$test_number/test-input-$test_number.gra --headerpath=test$test_number/test-header-$test_number.dat --outputpath=test$test_number/myoutput.txt --startk=$startk --endk=$endk --verbose=$verbose --p=$p"
        output=$(eval "$cmd" | grep -oP '(?<=Total Time:).*')
        echo "$nt,$np,$output" >> summary.txt
    done
done

echo "--------------------" >> summary.txt
