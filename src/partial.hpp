#ifndef PARTIAL_HPP
#define PARTIAL_HPP

#include <math.h>
#include <vector>

#include "RcppEigen.h"
#include "optimiser.hpp"
#include "utils.hpp"
#include "survival.svb_types.h"


double opt_par_mu(double sigma, double lambda, const vec &P, const vec &x_j,
    const std::vector<uint> &order_delta);

double opt_par_sig(double mu, double lambda, const vec &P, const vec &x_j,
    const std::vector<uint> &order_delta);

double opt_par_gam(double mu, double sigma, double lambda, double a_0, double b_0,
    const vec &P, const vec &x_j, const std::vector<uint> &order_delta);

#endif
