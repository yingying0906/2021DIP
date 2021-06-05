all: hw1

hw1: hw1.cpp
	g++ -o hw1 hw1.cpp `pkg-config --libs opencv4` `pkg-config --cflags opencv4`

clean:
	rm server client
