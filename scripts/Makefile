CXX = g++

CXXFLAGS = -Wall -g $(shell pkg-config --cflags opencv4)
LDLIBS = $(shell pkg-config --libs opencv4)

TARGETS = visualize_normal visualize_fisheye fisheye

all: $(TARGETS)

visualize_normal: visualize_normal.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

visualize_fisheye: visualize_fisheye.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

fisheye: fisheye.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
clean:
	rm -f $(TARGETS) *.o