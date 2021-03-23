#include "utils.hpp"


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

double 
sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

arma::vec
init_P(arma::mat X, arma::vec m, arma::vec s, arma::vec g)
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
rm_P(arma::vec P, arma::vec x_j, double m, double s, double g)
{
    return P /= g * normal_mgf(x_j, m, s) + (1 - g);
}

arma::vec
add_P(arma::vec P, arma::vec x_j, double m, double s, double g)
{
    return P %= g * normal_mgf(x_j, m, s) + (1 - g);
}
