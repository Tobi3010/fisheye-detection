CXX = g++

CXXFLAGS = -Wall -g $(shell pkg-config --cflags opencv4)
LDLIBS = $(shell pkg-config --libs opencv4)

TARGETS = test_distort_func

all: $(TARGETS)

test_distort_func: test_distort_func.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
clean:
	rm -f $(TARGETS) *.o