SRCDIR := $(shell pwd)/src
OBJDIR := $(shell pwd)/obj
MAIN := $(SRCDIR)/Main.cpp
SRCS := $(filter-out $(MAIN) $(SRCDIR)/Gem5Wrapper.cpp, $(wildcard $(SRCDIR)/*.cpp))
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))


# Ramulator currently supports g++ 5.1+ or clang++ 3.4+.  It will NOT work with
#   g++ 4.x due to an internal compiler error when processing lambda functions.
CXX := clang++
# CXX := g++
UNAME_S := $(shell uname -s)
CXXFLAGS := -O3 -std=c++11 -g3 -Wall
ifeq ($(UNAME_S),Linux)
	CXXFLAGS += -Wno-unused-lambda-capture -Wno-delete-non-virtual-dtor
else ifeq ($(UNAME_S),Darwin)
	CXXFLAGS += -Wno-unused-lambda-capture -Wno-delete-non-abstract-non-virtual-dtor
endif

.PHONY: all clean depend

all: depend ramulator

clean:
	rm -f ramulator
	rm -rf $(OBJDIR)

depend: $(OBJDIR)/.depend


$(OBJDIR)/.depend: $(SRCS)
	@mkdir -p $(OBJDIR)
	@rm -f $(OBJDIR)/.depend
	@$(foreach SRC, $(SRCS), $(CXX) $(CXXFLAGS) $(CFLAGS) -DRAMULATOR -MM -MT $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRC)) $(SRC) >> $(OBJDIR)/.depend ;)

ifneq ($(MAKECMDGOALS),clean)
-include $(OBJDIR)/.depend
endif


ramulator: $(MAIN) $(OBJS) $(SRCDIR)/*.h | depend
	$(CXX) $(CXXFLAGS) $(CFLAGS) -DRAMULATOR -o $@ $(MAIN) $(OBJS)

ramulator2: $(MAIN) $(OBJS) $(SRCDIR)/*.h | depend
	$(CXX) $(CXXFLAGS) $(CFLAGS) -DRAMULATOR -o $@ $(MAIN) $(OBJS)

libramulator.a: $(OBJS) $(OBJDIR)/Gem5Wrapper.o
	libtool -static -o $@ $(OBJS) $(OBJDIR)/Gem5Wrapper.o

$(OBJS): | $(OBJDIR)

$(OBJDIR): 
	@mkdir -p $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS) -DRAMULATOR -c -o $@ $<
