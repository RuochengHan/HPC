CXX?=g++-7
MPICXX?=mpic++

perf?=0
ifeq "$(perf)" "1"
	CPPFLAGS += -D_PERF_
endif

# these are for running on Euler!
CPPFLAGS+=-Wall -O3 -std=c++11
CPPFLAGS_THREADS=$(CPPFLAGS) -fopenmp


all: diffusion2d_serial diffusion2d_openmp diffusion2d_mpi

diffusion2d_serial: diffusion2d_openmp.cpp
	$(CXX) $(CPPFLAGS) -o $@ $<

diffusion2d_openmp: diffusion2d_openmp.cpp
	$(CXX) $(CPPFLAGS_THREADS) -o $@ $<

diffusion2d_mpi: diffusion2d_mpi.cpp
	$(MPICXX) $(CPPFLAGS) -o $@ $<

clean:
	rm -f diffusion2d_serial diffusion2d_openmp diffusion2d_mpi
