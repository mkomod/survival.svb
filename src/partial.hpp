#ifndef PARTIAL_HPP
#define PARTIAL_HPP

#include "RcppArmadillo.h"


double opt_par_mu(double sigma, double lambda, 
    const std::vector<arma::uvec> &R, const arma::uvec &F, 
    const arma::vec &P, const arma::vec &x_j);

double opt_par_sig(double mu, double lambda, 
    const std::vector<arma::uvec> &R, const arma::uvec &F, 
    const arma::vec &P, const arma::vec &x_j);

double opt_par_gam(double mu, double sigma, double a_0, double b_0, 
    double lambda, const std::vector<arma::uvec> &R, const arma::uvec &F, 
    const arma::vec &P, const arma::vec &x_j);


#endif
