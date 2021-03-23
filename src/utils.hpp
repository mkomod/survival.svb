#ifndef UTILS_HPP
#define UTILS_HPP

#include "RcppArmadillo.h"

double sigmoid(double x);

double normal_mgf(double x, double m, double s);
arma::vec normal_mgf(arma::vec x, double m, double s);

arma::vec init_P(arma::mat X, arma::vec m, arma::vec s, arma::vec g);
arma::vec rm_P(arma::vec P, arma::vec x_j, double m, double s, double g);
arma::vec add_P(arma::vec P, arma::vec x_j, double m, double s, double g);

#endif
