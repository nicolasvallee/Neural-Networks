CXX=g++
CPPFLAGS=-Wall -std=c++11 -g -O3

LDFLAGS=-g
LDLIBS=#-lsfml-graphics -lsfml-window -lsfml-system

SRCS=$(shell find . -name "*.cpp")
OBJS=$(patsubst %.cpp, %.o, $(SRCS))


neural-app : $(OBJS)
	$(CXX) $(LDFLAGS) -o neural-app $(OBJS) $(LDLIBS)

%.o : %.cpp
	g++ $(CPPFLAGS) -MMD -c -o $@ $<


clean:
	rm -f $(OBJS)


depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $^ > ./.depend;

include .depend


