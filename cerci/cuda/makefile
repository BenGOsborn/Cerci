all: main run clean 

main: main.o tensoroperations.o
	nvcc main.o tensoroperations.o -o main

main.o: main.cu
	nvcc -c main.cu

tensoroperations.o: tensoroperations.cu
	nvcc -c tensoroperations.cu

run:
	echo "---------- PROGRAM START ----------\n" && ./main && echo "\n---------- PROGRAM END ----------"

clean:
	rm *.o main
