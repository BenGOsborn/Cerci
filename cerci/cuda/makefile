main: main.o tensoroperations.o
	nvcc main.o tensoroperations.o main

main.o: main.cu
	nvcc -c main.cu

tensoroperations.o: tensoroperations.cu
	nvcc -c tensoroperations.cu

clean:
	rm *.o main
