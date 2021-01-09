#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <omp.h>

#include "timer.hpp"

struct Diagnostics
{
    double time;
    double heat;

    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D
{
public:
    Diffusion2D(const double D,
                const double L,
                const int N,
                const double dt,
                const int rank)
            : D_(D), L_(L), N_(N), dt_(dt), rank_(rank)
    {
        // Real space grid spacing.
        dr_ = L_ / (N_ - 1);

        // Stencil factor.
        fac_ = dt_ * D_ / (dr_ * dr_);

        // Actual dimension of a row (+2 for the ghost cells).
        real_N_ = N + 2;

        // Total number of cells.
        Ntot_ = (N_ + 2) * (N_ + 2);

        rho_.resize(Ntot_, 0.);
        rho_tmp_.resize(Ntot_, 0.);

        // Check that the timestep satisfies the restriction for stability.
        std::cout << "timestep from stability condition is "
                  << dr_ * dr_ / (4.0 * D_) << '\n';

        initialize_density();
    }

    void advance()
    {
        // Central differences in space, forward Euler in time with Dirichlet
        // boundaries.
#pragma omp parallel for collapse(2)
        for (int i = 1; i <= N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
                rho_tmp_[i * real_N_ + j] =
                        rho_[i * real_N_ + j]
                      + fac_ * (rho_[i * real_N_ + (j + 1)]
                              + rho_[i * real_N_ + (j - 1)]
                              + rho_[(i + 1) * real_N_ + j]
                              + rho_[(i - 1) * real_N_ + j]
                              - 4.0 * rho_[i * real_N_ + j]);
            }
        }

        // Use swap instead of rho_ = rho_tmp_. This is much more efficient,
        // because it does not copy element by element, just replaces storage
        // pointers.
        using std::swap;
        swap(rho_tmp_, rho_);
    }

    void compute_diagnostics(const double t)
    {
        double heat = 0.0;
        for (int i = 1; i <= N_; ++i)
            for (int j = 1; j <= N_; ++j)
                heat += dr_ * dr_ * rho_[i * real_N_ + j];

#if DEBUG
        std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
        diag_.push_back(Diagnostics(t, heat));
    }

    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag_)
            out_file << d.time << '\t' << d.heat << '\n';
        out_file.close();
    }

private:

    void initialize_density()
    {
        /// Initialize rho(x, y, t=0).
        double bound = .5;

#pragma omp parallel for collapse(2)
        for (int i = 1; i <= N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
                if (std::abs((i - 1) * dr_ - .5 * L_) < bound
                        && std::abs((j - 1) * dr_ - .5 * L_) < bound) {
                    rho_[i * real_N_ + j] = 1;
                } else {
                    rho_[i * real_N_ + j] = 0;
                }
            }
        }
    }

    double D_, L_;
    int N_, Ntot_, real_N_;
    double dr_, dt_, fac_;
    int rank_;
    std::vector<double> rho_, rho_tmp_;
    std::vector<Diagnostics> diag_;
};

#if !defined(_OPENMP)
int omp_get_num_threads() { return 1; }
#endif

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D L N dt\n";
        return 1;
    }

#pragma omp parallel
    {
#pragma omp master
        std::cout << "Running with " << omp_get_num_threads() << " threads\n";
    }

    const double D = std::stod(argv[1]);
    const double L = std::stod(argv[2]);
    const int N = std::stoul(argv[3]);
    const double dt = std::stod(argv[4]);

    Diffusion2D system(D, L, N, dt, 0);

    timer t;
    t.start();
    for (int step = 0; step < 10000; ++step) {
        system.advance();
#ifndef _PERF_
        system.compute_diagnostics(dt * step);
#endif
    }
    t.stop();

    std::cout << "Timing: " << N << ' ' << t.get_timing() << '\n';

#ifndef _PERF_
    system.write_diagnostics("diagnostics_openmp.dat");
#endif

    return 0;
}
