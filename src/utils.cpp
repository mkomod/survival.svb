#include "utils.h"


// [[Rcpp::export]]
double normal_mgf(double x, double m, double s)
{
    return exp(x * m + 0.5 * x * x * s * s);
}


double log_normal_mgf(double x, double m, double s) 
{
    return m * x + 0.5 * x * x * s * s;
}


// [[Rcpp::export]]
vec log_normal_mgf(const vec &x, double m, double s) 
{
    vec res = vec(x.rows());
    for (int i = 0; i < x.rows(); ++i) {
	res(i) = m * x(i) + 0.5 * s * s * x(i) * x(i);
    }
    return res;
}

// [[Rcpp::export]]
double sigmoid(double x) 
{
    return 1 / (1 + exp(-x));
}


// Compute the log P(x_ij) = log(1 - g + g * normal_mgf(x_ij, mu, sig))
// log(1 - g + g * exp(x)) = x + log(g + (1-g)*exp(-x))
//           ^--- multiply by exp(x)/exp(x)
static double P_ij(double x_ij,  double m, double s, double g) 
{
    if (g == 0.0)
	return 0;
    
    double x = log_normal_mgf(x_ij, m, s);
    if (g == 1.0)
	return x;
    
    if (x > 0) {
	return x + log(g + (1-g)*exp(-x));
    } else {
	return log(1-g + g*exp(x));
    }
}


// [[Rcpp::export]]
vec init_log_P(const mat &X, const vec &m, const vec &s, const vec &g)
{
    uint n = X.rows();
    uint p = X.cols();
    vec P = vec::Zero(n);

    double x = 0.0;

    for (uint i = 0; i < n; ++i) {
	for (uint j = 0; j < p; ++j)  {
	    P(i) += P_ij(X(i, j), m(j), s(j), g(j));
	}
    }

    return P;
}


void rm_log_P(vec &P, const vec &x_j, double m, double s, double g)
{
    for (uint i = 0; i < x_j.rows(); ++i)  {
	P(i) -= P_ij(x_j(i), m, s, g);
    }
}


void add_log_P(vec &P, const vec &x_j, double m, double s, double g)
{
    for (uint i = 0; i < x_j.rows(); ++i)  {
	P(i) += P_ij(x_j(i), m, s, g);
    }
}


// order the failure times
// [[Rcpp::export]]
std::vector<uint> order_T(const vec &T)
{
    // order(T)
    std::vector<uint> T_ord(T.size());
    std::iota(T_ord.begin(), T_ord.end(), 0);

    std::stable_sort(T_ord.begin(), T_ord.end(),
	    [&T](uint i1, uint i2) {return T[i1] < T[i2];});
 
    return T_ord;
}


// get the index where failures occured for sorted T (times)
// [[Rcpp::export]]
std::vector<uint> order_delta(const vec &T, const vec &delta) 
{
    std::vector<uint> T_ord = order_T(T);
    std::vector<uint> delta_ord;

    for (uint i = 0; i < T_ord.size(); ++i)
	if (delta(T_ord.at(i)) == 1) delta_ord.push_back(i);

    return delta_ord;
}


// compute the log log_likelihood
// [[Rcpp::export]]
double log_likelihood(const vec &T, const vec &delta, const mat &X, const vec &b)
{
    std::vector<uint> T_ord = order_T(T);
    std::vector<uint> delta_ord = order_delta(T, delta);

    vec xb = X * b;
    double a = xb.maxCoeff();		// prevent overflow
    double den = 0.0;
    double tot = 0.0;

    int icf = 0;			// index of current failure
    int ilf = X.rows();			// index of last failure

    for (int i = delta_ord.size() - 1; i >= 0; --i) {
	icf = delta_ord.at(i);		// set the current failure to i

	for (int j = icf; j <= ilf - 1; ++j)
	    den += exp(xb(T_ord.at(j)) - a);		// prevent overflow
	
	tot += xb(T_ord.at(icf)) - (a + log(den));

	ilf = icf;			// set the index of last failure to current
    }

    return tot;
}

