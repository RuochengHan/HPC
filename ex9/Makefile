MPICXX=mpic++

CXXFLAGS=-Wall -O3 -std=c++11

all: diffusion2d_mpi

diffusion2d_mpi: diffusion2d_mpi.cpp timer.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	-rm -f diffusion2d_mpi density_mpi.dat
