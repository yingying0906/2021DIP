all: hw2_p2 hw2_p1

hw2_p2: hw2_p2.cpp
	g++ -o hw2_p2 hw2_p2.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4`

hw2_p1: hw2_p1.cpp
	g++ -o hw2_p1 hw2_p1.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4`

clean:
	rm hw2_p1 hw2_p2
