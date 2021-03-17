#ifndef UTILS_HPP
#define UTILS_HPP

#include "RcppArmadillo.h"

double normal_mgf(double x, double m, double s);
arma::vec normal_mgf(arma::vec x, double m, double s);

double sigmoid(double x);

#endif
