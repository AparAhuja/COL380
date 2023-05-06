main: main.cu
	nvcc -arch=sm_35 --std=c++11 -Xptxas -O3 -Xcompiler -O3 main.cu -o exec

approach1: approach1.cu
	nvcc -arch=sm_35 --std=c++11 -Xptxas -O3 -Xcompiler -O3 approach1.cu -o approach1

approach2: approach2.cu
	nvcc -arch=sm_35 --std=c++11 -Xptxas -O3 -Xcompiler -O3 approach2.cu -o approach2

approach3: approach3.cu
	nvcc -arch=sm_35 --std=c++11 -Xptxas -O3 -Xcompiler -O3 approach3.cu -o approach3

approach4: approach4.cu
	nvcc -arch=sm_35 --std=c++11 -Xptxas -O3 -Xcompiler -O3 approach4.cu -o approach4

finalapproach: finalapproach.cu
	nvcc -arch=sm_35 --std=c++11 -Xptxas -O3 -Xcompiler -O3 finalapproach.cu -o finalapproach

clean: exec
	rm exec