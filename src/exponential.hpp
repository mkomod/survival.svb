#ifndef _EXPONENTIAL_HPP
#define _EXPONENTIAL_HPP

#include "RcppArmadillo.h"
#include "utils.hpp"
#include "optimiser.hpp"


double opt_exp_mu(double sigma, double omega, double lambda, 
	const arma::vec &P, const arma::vec &T, const arma::vec &delta,
	const arma::vec &x_j);

double opt_exp_sig(double mu, double omega, double lambda, 
	const arma::vec &P, const arma::vec &T, const arma::vec &delta,
	const arma::vec &x_j);

double opt_exp_gam(double mu, double sigma, double omega, double lambda, 
	double a_0, double b_0, const arma::vec &P, const arma::vec &T, 
	const arma::vec &delta, const arma::vec &x_j);

double opt_exp_a(double b, double a_omega, double b_omega,
	const arma::vec &P, const arma::vec &T, const int n_delta);

double opt_exp_b(double a, double a_omega, double b_omega, const arma::vec &P, 
	const arma::vec &T, int n_delta);

#endif
