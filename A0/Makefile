#g++ -std=c++11 -O2 -mavx2 -c classify.cpp
CC=g++
# CFLAGS=-std=c++11 -O2 -fopenmp # original
CFLAGS=-std=c++11 -O2 -fopenmp -g

sources=main.cpp classify.cpp
objects=$(sources:.cpp=.o)

classify:$(objects)
	$(CC) $(CFLAGS) $^ -o $@

run: classify
#   <rangefile> <datafile> <max data to read> <number of threads to use> <repetitions>
	./classify rfile dfile 1009072 4 3 

record: classify
	perf record ./classify rfile dfile 1009072 4 10

analyse: classify
	./classify rfile dfile 1009072 4 10

#classify.o: classify.h
#	touch classify.o
#	$(CC) $(CFLAGS) -c $<
#main.o: classify.h
#	$(CC) $(CFLAGS) -c $<

%.o: %.cpp classify.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o classify
