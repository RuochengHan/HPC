#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include "timer.hpp"

struct Diagnostics
{
    double time;
    double heat;

    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D_MPI {
public:
    Diffusion2D_MPI(const double D,
                    const double L,
                    const int N,
                    const double dt,
                    const int rank,
                    const int procs)
            : D_(D), L_(L), N_(N), dt_(dt), rank_(rank), procs_(procs)
    {
        // Real space grid spacing.
        dr_ = L_ / (N_ - 1);

        // Stencil factor.
        fac_ = dt_ * D_ / (dr_ * dr_);

        // Number of rows per process.
        local_N_ = N_ / procs_;

        // Small correction for the last process.
        if (rank_ == procs - 1)
            local_N_ += N_ % procs_;

        // Actual dimension of a row (+2 for the ghost cells).
        real_N_ = N_ + 2;

        // Total number of cells.
        Ntot_ = (local_N_ + 2) * (N_ + 2);

        rho_.resize(Ntot_, 0.0);
        rho_tmp_.resize(Ntot_, 0.0);

        // Check that the timestep satisfies the restriction for stability.
        if (rank_ == 0) {
            std::cout << "timestep from stability condition is "
                      << dr_ * dr_ / (4. * D_) << '\n';
        }

        initialize_density();
    }

    void advance()
    {
        MPI_Request req[4];
        MPI_Status status[4];

        int prev_rank = rank_ - 1;
        int next_rank = rank_ + 1;

        // Exchange ALL necessary ghost cells with neighboring ranks.
        if (prev_rank >= 0) {
            // TODO:MPI
            // ...
            MPI_Irecv(&rho_[0*real_N_+1], N_, MPI_DOUBLE, prev_rank, 99, MPI_COMM_WORLD, &req[0]);
            MPI_Isend(&rho_[1*real_N_+1], N_, MPI_DOUBLE, prev_rank, 99, MPI_COMM_WORLD, &req[1]);
        } else {
            // TODO:MPI
            // ...
            req[0] = MPI_REQUEST_NULL;
            req[1] = MPI_REQUEST_NULL;
        }

        if (next_rank < procs_) {
          // TODO:MPI
          // ...
          MPI_Irecv(&rho_[(local_N_+1)*real_N_+1], N_, MPI_DOUBLE, next_rank, 99, MPI_COMM_WORLD, &req[2]);
          MPI_Isend(&rho_[local_N_*real_N_+1], N_, MPI_DOUBLE, next_rank, 99, MPI_COMM_WORLD, &req[3]);
      } else {
          // TODO:MPI
          // ...
          req[2] = MPI_REQUEST_NULL;
          req[3] = MPI_REQUEST_NULL;
        }

        // Central differences in space, forward Euler in time with Dirichlet
        // boundaries.
        for (int i = 2; i < local_N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
                // TODO:DIFF
                // rho_tmp_[ ?? ] = ...
                rho_tmp_[i*real_N_+j] = rho_[i*real_N_+j] + fac_ * (rho_[(i-1)*real_N_+j]
                                                          + rho_[(i+1)*real_N_+j]
                                                          + rho_[i*real_N_+j+1]
                                                          + rho_[i*real_N_+j-1]
                                                          - rho_[i*real_N_+j] * 4.);
             }
        }

        // Note: This exercise is about synchronous communication, but the
        // template code is formulated for asynchronous. In the latter case,
        // when non-blocking send/recv is used, you would need to add an
        // MPI_Wait here to make sure the incoming data arrives before
        // evaluating the boundary cells (first and last row of the local
        // matrix),
        // Namely, network communication takes time and you always want to
        // perform some work while waiting. In this code it means making the
        // diffusion step for the inner part of the grid, which doesn't require
        // any data from other nodes. Afterwards, when the data from
        // neighboring nodes arrives, the first and last row are handled.
        // As this is a synchronous-only exercise, feel free to merge the
        // following for loops into the previous ones.

        // Update the first and the last rows of each rank.
        for (int i = 1; i <= local_N_; i += local_N_ - 1) {
            for (int j = 1; j <= N_; ++j) {
                // TODO:DIFF
                // rho_tmp_[ ?? ] = ...
                rho_tmp_[i*real_N_+j] = rho_[i*real_N_+j] + fac_ * (rho_[(i-1)*real_N_+j]
                                                          + rho_[(i+1)*real_N_+j]
                                                          + rho_[i*real_N_+j+1]
                                                          + rho_[i*real_N_+j-1]
                                                          - rho_[i*real_N_+j] * 4.);
            }
        }

        // Use swap instead of rho_ = rho_tmp__. This is much more efficient,
        // because it does not copy element by element, just replaces storage
        // pointers.
        using std::swap;
        swap(rho_tmp_, rho_);
    }

    void compute_diagnostics(const double t)
    {
        double heat = 0.0;

        // TODO:DIFF - Integration to compute heat
        // ...
        for (int i = 1; i <= local_N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
                heat += rho_[i*real_N_+j] * dr_ * dr_;
            }
        }

        // TODO:MPI
        // ...
        MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank_ == 0) {
#if DEBUG
            std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
            diag.push_back(Diagnostics(t, heat));
        }
    }

    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag)
            out_file << d.time << '\t' << d.heat << '\n';
        out_file.close();
    }

private:

    void initialize_density()
    {
        //TODO:DIFF Implement the initialization of the density distribution
        int global_i;
        //std::fill(rho_.begin(), rho_.end(), 0.0)
        for (int i = 1; i <= local_N_; ++i) {
            global_i = rank_ * (N_ / procs_) + i; // since for the last rank_ local_N_ != N_ / procs_
            for (int j = 1; j <= N_; ++j) {
                if (std::abs((global_i-1) * dr_ - L_ / 2.0) < 0.5 &&
                    std::abs((j-1) * dr_ - L_ / 2.0) < 0.5) {
                    rho_[i*real_N_+j] = 1.0;
                }
                else {
                    rho_[i*real_N_+j] = 0.0;
                }
            }
        }
    }

    double D_, L_;
    int N_, Ntot_, local_N_, real_N_;
    double dr_, dt_, fac_;
    int rank_, procs_;

    std::vector<double> rho_, rho_tmp_;
    std::vector<Diagnostics> diag;
};


int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D L N dt\n";
        return 1;
    }

    int rank, procs;
    //TODO:MPI Initialize MPI, number of ranks and number of processes involved in the communicator
    // ...
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    std::cout << "Total " << procs << " processes are used." << std::endl;

    const double D = std::stod(argv[1]);
    const double L = std::stod(argv[2]);
    const int N = std::stoul(argv[3]);
    const double dt = std::stod(argv[4]);

    Diffusion2D_MPI system(D, L, N, dt, rank, procs);

#if DEBUG
    system.compute_diagnostics();
#endif

    timer t;
    t.start();
    for (int step = 0; step < 10000; ++step) {
        system.advance();
#ifndef _PERF_
        system.compute_diagnostics(dt * step);
#endif
    }
    t.stop();

    if (rank == 0)
        std::cout << "Timing: " << N << ' ' << t.get_timing() << '\n';

#ifndef _PERF_
    if (rank == 0)
        system.write_diagnostics("diagnostics_mpi.dat");
#endif

    // Finalize MPI
    // TODO:MPI
    // ...
    MPI_Finalize();

    return 0;
}
