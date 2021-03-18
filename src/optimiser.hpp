#ifndef _OPTIMISER_HPP
#define _OPTIMISER_HPP

#include "RcppArmadillo.h"

double Brent_fmin(double ax, double bx, double (*f)(double, void *),
		  void *info, double tol);

#endif