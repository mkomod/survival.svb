#ifndef _OPTIMISER_H
#define _OPTIMISER_H

#include "RcppEigen.h"

double Brent_fmin(double ax, double bx, double (*f)(double, void *),
		  void *info, double tol);

#endif
