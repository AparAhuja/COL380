CC=g++
CFLAGS=-std=c++11 -O3 -fopenmp

sources=main.cpp
objects=$(sources:.cpp=.o)

main:$(objects)
	$(CC) $(CFLAGS) $^ -L/home/btech/cs1190465/COL380/A1 -l:library1.so -o $@

%.o: %.cpp dataloader.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o main test

test: test.cpp
	g++ -std=c++11 -O3 -fopenmp -c test.cpp
	g++ -std=c++11 -O3 -fopenmp test.o -L/home/btech/cs1190465/COL380/A1 -l:library1.so -o test