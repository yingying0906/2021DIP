all: hw3_p2 hw3_p1

hw3_p2: hw3_p2.cpp
	g++ -o hw3_p2 hw3_p2.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4`

hw3_p1: hw3_p1.cpp
	g++ -o hw3_p1 hw3_p1.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4`

clean:
	rm hw3_p1 hw3_p2
