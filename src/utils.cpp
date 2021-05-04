#include "utils.hpp"


double
normal_mgf(double x, double m, double s)
{
    return exp(x * m + 1.0/2.0 * x * x * s * s);
}

arma::vec
normal_mgf(const arma::vec &x, double m, double s)
{
    return exp(x * m + 1.0/2.0 * pow(x * s, 2));
}


arma::vec
normal_mgf(const arma::vec &x, const arma::vec &m, const arma::vec &s) 
{
    return exp(x % m + 0.5 * pow(x % s, 2));
}


arma::vec
log_normal_mgf(const arma::vec &x, double m, double s)
{
    return x * m + 1.0/2.0 * pow(x * s, 2);
}

double 
sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

arma::vec
init_P(const arma::mat &X, const arma::vec &m, const arma::vec &s, 
	const arma::vec &g)
{
    int n = X.n_rows;
    int p = X.n_cols;
    arma::vec P(n, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
	for (int j = 0; j < p; ++j) {
	    // Prod x = exp(log(Prod x)) = exp(Sum log x))
	    P(i) += log(g(j) * normal_mgf(X(i, j), m(j), s(j)) + 1.0-g(j));
	}
    }
    return exp(P);
}

arma::vec
rm_P(arma::vec P, const arma::vec &x_j, double m, double s, double g)
{
    return P /= (g * normal_mgf(x_j, m, s) + (1 - g));
}

arma::vec
add_P(arma::vec P, const arma::vec &x_j, double m, double s, double g)
{
    return P %= (g * normal_mgf(x_j, m, s) + (1 - g));
}


arma::vec
init_log_P(const arma::mat &X, const arma::vec &m, const arma::vec &s, 
	const arma::vec &g)
{
    int n = X.n_rows;
    int p = X.n_cols;
    arma::vec P(n, arma::fill::zeros);
    arma::vec x_i(p, arma::fill::zeros);

    for (int i = 0; i < n; ++i) {
	x_i = X.row(i).t();
	P(i) = sum(log(g % normal_mgf(x_i, m, s) + (1-g)));
    }
    return P;
}

arma::vec
rm_log_P(arma::vec P, const arma::vec &x_j, double m, double s, double g)
{
    return P -= log(g * normal_mgf(x_j, m, s) + (1 - g));
}

arma::vec
add_log_P(arma::vec P, const arma::vec &x_j, double m, double s, double g)
{
    return P += log(g * normal_mgf(x_j, m, s) + (1 - g));
}

