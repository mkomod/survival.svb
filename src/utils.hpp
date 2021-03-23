#ifndef UTILS_HPP
#define UTILS_HPP

#include "RcppArmadillo.h"

double sigmoid(double x);

double normal_mgf(double x, double m, double s);

arma::vec normal_mgf(const arma::vec &x, double m, double s);

arma::vec init_P(const arma::mat &X, const arma::vec &m, const arma::vec &s, 
	const arma::vec &g);

arma::vec rm_P(arma::vec P, const arma::vec &x_j, double m, double s,
	double g);

arma::vec add_P(arma::vec P, const arma::vec &x_j, double m, double s, 
	double g);

#endif
