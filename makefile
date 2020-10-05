src = $(wildcard *.cpp)
obj = $(src:.cpp=.o)
dep = $(obj:.o=.d)
CXX = 
MKLROOT = 
EIGEN = ./eigen/Eigen
MORTON = ./libmorton
HLIBPROROOT = 
OMPFLAG = 
LDOMP = 
LDHLIBPRO = 
LDMKL = 




DEBUG ?= F
ifeq ($(DEBUG) , F)
	CXXFLAGS = -I$(EIGEN) -I$(MKLROOT)/include -I$(MORTON)/include -I$(HLIBPROROOT)/include $(OMPFLAG) -O3 -m64 -std=c++17 -Wall
	LDFLAGS = $(LDHLIBPRO) $(LDMKL) $(LDOMP)
else
	CXXFLAGS = -I$(EIGEN) -I$(MKLROOT)/include -I$(MORTON)/include -I$(HLIBPROROOT)/include -O0 -m64 -std=c++17 -Wall
	LDFLAGS = $(LDHLIBPRO) $(LDMKL) 
endif

main: $(obj)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) $< -MM -MT $@ > $*.d
	$(CXX) $(CXXFLAGS) $(OPT) -c $< -o $@ 

include $(wildcard $(dep))

.PHONY: clean
clean:
	rm -f $(wildcard $(obj) main) 

.PHONY: cleandep
cleandep:
	rm -f $(wildcard $(dep))

