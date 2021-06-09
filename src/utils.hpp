#ifndef UTILS_HPP
#define UTILS_HPP

#include "RcppArmadillo.h"

double sigmoid(double x);

// normal funcs
double normal_mgf(double x, double m, double s);
arma::vec normal_mgf(const arma::vec &x, double m, double s);
arma::vec normal_mgf(const arma::vec &x, const arma::vec &m, const arma::vec &s);
arma::vec log_normal_mgf(const arma::vec &x, double m, double s);

// P
arma::vec init_P(const arma::mat &X, const arma::vec &m, const arma::vec &s, 
    const arma::vec &g);
arma::vec rm_P(arma::vec P, const arma::vec &x_j, double m, double s, double g);
arma::vec add_P(arma::vec P, const arma::vec &x_j, double m, double s, double g);

// log P
arma::vec init_log_P(const arma::mat &X, const arma::vec &m, const arma::vec &s, 
    const arma::vec &g);
void rm_log_P(arma::vec &P, const arma::vec &x_j, double m, double s, double g);
void add_log_P(arma::vec &P, const arma::vec &x_j, double m, double s, double g);


#endif
