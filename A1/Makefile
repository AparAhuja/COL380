CC=g++
CFLAGS=-std=c++17 -O3 -fopenmp

exec: approach5.cpp
	$(CC) $(CFLAGS) -c approach5.cpp
	$(CC) $(CFLAGS) approach5.o -L. -l:library.so -o exec

approach1: approach1.cpp
	$(CC) $(CFLAGS) -c approach1.cpp
	$(CC) $(CFLAGS) approach1.o -L. -l:library.so -o approach1

approach2: approach2.cpp
	$(CC) $(CFLAGS) -c approach2.cpp
	$(CC) $(CFLAGS) approach2.o -L. -l:library.so -o approach2

approach3: approach3.cpp
	$(CC) $(CFLAGS) -c approach3.cpp
	$(CC) $(CFLAGS) approach3.o -L. -l:library.so -o approach3

approach4: approach4.cpp
	$(CC) $(CFLAGS) -c approach4.cpp
	$(CC) $(CFLAGS) approach4.o -L. -l:library.so -o approach4

approach5: approach5.cpp
	$(CC) $(CFLAGS) -c approach5.cpp
	$(CC) $(CFLAGS) approach5.o -L. -l:library.so -o approach5

datagen: datagen.cpp
	$(CC) $(CFLAGS) datagen.cpp -o datagen

clean:
	rm *.o exec approach1 approach2 approach3 approach4 approach5 datagen