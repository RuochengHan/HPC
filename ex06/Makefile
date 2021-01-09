# Make sure to load: gcc, mkl

CC = g++
CFLAGS = -O3 -Wall 

LIBPATH=$(MKLROOT)/lib/intel64
MKL_SERIAL_LINK=-Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
MKL_PARA_LINK=-Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
LDFLAGS = -lm -lpthread -lz -ldl -fopenmp

all: pca_seq pca_omp

pca_seq: pca_seq.o 
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(MKL_SERIAL_LINK) $(LDFLAGS)

pca_seq.o: pca.cpp
	$(CC) $(CFLAGS) -o pca_seq.o -c pca.cpp

pca_omp: pca_omp.o
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(MKL_PARA_LINK) $(LDLFLAGS)

pca_omp.o: pca.cpp
	$(CC) $(CFLAGS) -fopenmp -o pca_omp.o -c pca.cpp

clean:
	rm -f *.o *~ pca_seq pca_omp

