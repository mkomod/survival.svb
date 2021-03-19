#ifndef _EXPONENTIAL_HPP
#define _EXPONENTIAL_HPP

#include "RcppArmadillo.h"
#include "utils.hpp"
#include "optimiser.hpp"



double objective_mu_exp(double mu, void* args);

double objective_sigma_exp(double sigma, void* args);

double optimise_gamma_exp(double mu, double sigma, double lambda, 
	double omega, double a_0, double b_0, const arma::vec &P, 
	const arma::vec &T, const arma::vec &delta, const arma::vec &x_j, 
	bool verbose);

double optimise_mu_exp(double sigma, double omega, double lambda, 
	const arma::vec &P, const arma::vec &T, const arma::vec &delta,
	const arma::vec x_j, 
	bool verbose);

double optimise_sigma_exp(double mu, double omega, double lambda, 
	const arma::vec &P, const arma::vec &T, const arma::vec &delta,
	const arma::vec x_j, bool verbose);

double optimise_a_exp(double b, double a_omega, double b_omega,
    const arma::vec &P, const arma::vec &T, const int n_delta,
    bool verbose);

double optimise_b_exp(double a, double a_omega, double b_omega, 
	const arma::vec P, const arma::vec T, int n_delta);

#endif
