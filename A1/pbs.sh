#!/bin/sh
#PBS -N testrun
#PBS -P col380
#PBS -lselect=1:ncpus=8:mem=10G
### walltime=hhh:mm:ss
#PBS -l walltime=000:00:25

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PBS_O_WORKDIR
echo $PBS_JOBID STARTED >> test.txt
make test
./test "data/input2" "data/myoutput" 8 >> test.txt
echo $PBS_JOBID FINISHED >> test.txt