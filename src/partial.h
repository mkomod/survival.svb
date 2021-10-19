#ifndef PARTIAL_H
#define PARTIAL_H

#include <math.h>
#include <vector>

#include "RcppEigen.h"
#include "optimiser.h"
#include "utils.h"
#include "survival.svb_types.h"


double opt_par_mu(double mu, double sigma, double lambda, const vec &P, const vec &x_j,
    const std::vector<uint> &order_delta);

double opt_par_sig(double sigma, double mu, double lambda, const vec &P, const vec &x_j,
    const std::vector<uint> &order_delta);

double opt_par_gam(double mu, double sigma, double lambda, double a_0, double b_0,
    const vec &P, const vec &x_j, const std::vector<uint> &order_delta);

#endif
