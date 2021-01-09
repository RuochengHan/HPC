// File       : pca.cpp
// Created    : Thu Nov 02 2017 03:01:30 PM (+0100)
// Description: Principal component analysis applied to image compression
// Copyright 2017 ETH Zurich. All Rights Reserved.
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include <omp.h>
#include <zlib.h>

// interface for LAPACK routines.  On Euler, you must load the MKL library:
// $ module load mkl
#include <mkl_lapack.h>


///////////////////////////////////////////////////////////////////////////////
// helpers
double *read_gzfile(char *filename, int frows, int fcols, int rows, int cols)
{
    double *A, *buf;
    gzFile fp;
    int i;

    A = new (std::nothrow) double[rows*cols];
    assert(A != NULL);

    buf = new (std::nothrow) double[fcols];
    assert(buf != NULL);

    fp = gzopen(filename, "rb");
    if (fp == NULL) {
        std::cout << "Input file not available!\n";
        exit(1);
    }

    for (i = 0; i < rows; i++) {
        gzread(fp, buf, fcols*sizeof(double));
        memcpy(&A[i*cols], buf, cols*sizeof(double));
    }
    gzclose(fp);

    delete[] buf;

    return A;
}

void write_ascii(const char* const filename, const double* const data, const int rows, const int cols)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        std::cout << "Failed to create output file\n";
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.4lf ", data[i*cols+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// elvis.bin.gz:   469x700
// cyclone.bin.gz: 4096x4096
// earth.bin.gz:   9500x9500
//
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // input parameters (default)
    int m = 469, n = 700;                        // image size (rows, columns)
    int npc = 50;                                // number of principal components
    char *inp_filename = (char *)"elvis.bin.gz"; // input filename (compressed binary)
    char *out_filename = (char *)"elvis.50.bin"; // output filename (text)

    // parse input parameters
    if ((argc != 1) && (argc != 11)) {
        std::cout << "Usage: " << argv[0] << " -m <rows> -n <cols> -npc <number of principal components> -if <input filename> -of <output filename>\n";
        exit(1);
    }

    for(int i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-m" ) == 0 ) {
            m = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-n" ) == 0 ) {
            n = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-npc" ) == 0 ) {
            npc = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-if" ) == 0 ) {
            inp_filename = argv[i+1];
            i++;
        }
        if( strcmp( argv[i], "-of" ) == 0 ) {
            out_filename = argv[i+1];
            i++;
        }
    }

    if (npc > n) npc = n;
    double t_elapsed;

    ///////////////////////////////////////////////////////////////////////////
    // Read image data.  The image dimension is m x n.  The returned pointer
    // points to the data in row-major order.  That is, if (i,j) corresponds to
    // to the row and column index, respectively, you access the data with
    // pixel_{i,j} = I[i*n + j], where 0 <= i < m and 0 <= j < n.
    ///////////////////////////////////////////////////////////////////////////
    double *I = read_gzfile(inp_filename, m, n, m, n);

    // A = transpose(I), so image features (columns) are stored in rows.  More
    // efficient to work with the data in this layout.
    double *A = new (std::nothrow) double[n*m];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i*m + j] = I[j*n+i];
        }
    }
    delete[] I;

    ///////////////////////////////////////////////////////////////////////////
    // TODO: Implement your PCA algorithm here
    // 1. Compute mean and standard deviation of your image features
    // 2. Normalize the data
    // 3. Build the covariance matrix
    // 4. Compute the eigenvalues and eigenvectors of the covariance matrix.
    //    Use LAPACK here.
    // 5. Compute the principal components and report the compression ratio
    // 6. Reconstruct the image from the compressed data and dump the image in
    //    ascii.
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 1.

    double start_t = omp_get_wtime();

    t_elapsed = -omp_get_wtime();

    double *AMean = new (std::nothrow) double[n];
    double *AStd  = new (std::nothrow) double[n];
    assert(AMean != NULL);
    assert(AStd  != NULL);

    for (int i = 0; i < n; ++i)
    {
        AMean[i] = 0.;
        AStd[i] = 0.;
        for (int j = 0; j < m; ++j)
        {
            AMean[i] += A[i*m+j];
            AStd[i] += A[i*m+j] * A[i*m+j];
        }
        AMean[i] /= m;
        AStd[i] /= m;
        AStd[i] = sqrt(AStd[i] - AMean[i] * AMean[i]);
    }
    t_elapsed += omp_get_wtime();
    std::cout << "MEAN/STD TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 2.
    t_elapsed = -omp_get_wtime();

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            A[i*m+j] = (A[i*m+j] - AMean[i]) / AStd[i];
        }
    }
    t_elapsed += omp_get_wtime();
    std::cout << "NORMAL. TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 3.
    t_elapsed = -omp_get_wtime();
    double *C = new (std::nothrow) double[n*n];
    assert(C!=NULL);

    #pragma omp parallel for
    for (int is = 0; is < n; ++is)
    {
        for (int it = 0; it <= is; ++it)
        {
            C[is*n+it] = 0.;
            for (int j = 0; j < m; ++j)
            {
                C[is*n+it] += A[is*m+j] * A[it*m+j];
            }
            C[is*n+it] /= m;
            C[it*n+is] = C[is*n+it];
        }
    }

    t_elapsed += omp_get_wtime();
    std::cout << "C-MATRIX TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO 4. LAPACK
    t_elapsed = -omp_get_wtime();

    // see also for the interface to dsyev_():
    // http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html#ga442c43fca5493590f8f26cf42fed4044
    char jobz = 'V'; // TODO: compute both, eigenvalues and orthonormal eigenvectors
    char uplo = 'L'; // TODO: how did you compute the (symmetric) covariance matrix?
    int info, lwork;

    double *W = new (std::nothrow) double[n]; // eigenvalues
    assert(W != NULL);

    double *work = new (std::nothrow) double[2];
    assert(work != NULL);

    // first call to dsyev_() with lwork = -1 to determine the optimal
    // workspace (cheap call)
    lwork = -1;

    // TODO: call dsyev here
    dsyev_(&jobz, &uplo, &n, C, &n, W, work, &lwork, &info);

    lwork = (int)work[0];
    delete[] work;

    // allocate optimal workspace
    work = new (std::nothrow) double[lwork];
    assert(work != NULL);

    // second call to dsyev_(), eigenvalues and eigenvectors are computed here
    // TODO: call dsyev here
    dsyev_(&jobz, &uplo, &n, C, &n, W, work, &lwork, &info);

    t_elapsed += omp_get_wtime();
    std::cout << "DSYEV TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 5.
    t_elapsed = -omp_get_wtime();
    double *PCReduced = new (std::nothrow) double[m*npc];
    assert(PCReduced != NULL);

    #pragma omp parallel for
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < npc; ++j)
        {
            PCReduced[i*npc+j] = 0.;
            for (int is = 0; is < n; ++is)
            {
                PCReduced[i*npc+j] += A[is*m+i] * C[(n-j-1)*n+is];
            }
        }
    }

    // TODO: Report the compression ratio
    std::cout << "COMPRESSION RATIO=" << (m * n)/((m + n) * npc + 2 * n) * 100 << " %\n";

    t_elapsed += omp_get_wtime();
    std::cout << "PCREDUCED TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    double end_t = omp_get_wtime();
    std::cout << "OVERALL TIME=" << end_t - start_t << " seconds\n";

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 6
    double *Z = new (std::nothrow) double[m*n]; // memory for reconstructed image
    assert(Z != NULL);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // TODO: Reconstruct image here.  Don't forget to denormalize.  The
            // dimension of the reconstructed image is m x n (rows x columns).
            // Z[i*n + j] = ...
            Z[i*n+j] = 0.;
            for (int k = 0; k < npc; ++k)
            {
                Z[i*n+j] += PCReduced[i*npc+k] * C[(n-k-1)*n+j];
            }
            Z[i*n+j] = Z[i*n+j] * AStd[j] + AMean[j];
        }
    }


    // Write the reconstructed image in ascii format.  You can view the image
    // in Matlab with the show_image.m script.
    write_ascii(out_filename, Z, m, n);
    ///////////////////////////////////////////////////////////////////////////

    // cleanup
    delete[] work;
    delete[] W;
    delete[] C;
    delete[] Z;
    delete[] PCReduced;
    delete[] A;
    delete[] AMean;
    delete[] AStd;

    return 0;
}
