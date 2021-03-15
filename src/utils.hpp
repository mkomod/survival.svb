#ifndef _UTILS_H
#define _UTILS_H

#include "RcppArmadillo.h"

double
normal_mgf(double x, double m, double s)
{
    return exp(x * m + 1.0/2.0 * x * x * s * s);
}

arma::vec
normal_mgf(arma::vec x, double m, double s)
{
    return exp(x * m + 1.0/2.0 * pow(x * s, 2));
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}


#endif
