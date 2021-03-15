#include "RcppArmadillo.h"
#include "RcppEnsmallen.h"
// [[Rcpp::depends(RcppEnsmallen)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include "utils.hpp"

// [[Rcpp::export]]
arma::vec
init_P(arma::mat X, arma::vec m, arma::vec s, arma::vec g)
{
    int n = X.n_rows;
    int p = X.n_cols;
    arma::vec P(n, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
	for (int j = 0; j < p; ++j) {
	    // Prod x = exp(log(Prod x)) = exp(Sum log x))
	    P(i) += log(g(j) * normal_mgf(X(i, j), m(j), s(j)) + 1 - g(j));
	}
    }
    return exp(P);
}

// [[Rcpp::export]]
arma::vec
rm_P(arma::vec P, arma::vec x_j, double m, double s, double g)
{
    return P /= g * normal_mgf(x_j, m, s) + (1 - g);
}

// [[Rcpp::export]]
arma::vec
add_P(arma::vec P, arma::vec x_j, double m, double s, double g)
{
    return P %= g * normal_mgf(x_j, m, s) + (1 - g);
}


Rcpp::List 
fit(arma::colvec T, arma::mat X, double omega)
{
    // init m, s, g
    int p = X.n_cols;
    int n = X.n_rows;
    arma::vec m, s, g = arma::vec(p, arma::fill::zeros);
    
    // P.i := x_j.t * m + 1/2 x_j * S * x_j.t()
    // arma::vec P = initialise_P(m, s, X);

    for (int j = 0; j < X.n_cols; ++j) {
    // opt m
	// P = sub_from_P(P, m(j), s(j), X.col(j));
	// m(j) = update_m();
	// P = add_to_P(P, m(j), s(j), X.col(j));

	// opt s
	// P = sub_from_P(P, m(j), s(j), X.col(j));
	// s(j) = update_s();
	// P = add_to_P(P, m(j), s(j), X.col(j));

	// opt g
	// g(j) = update_g();

    }

    return Rcpp::List::create(
	    Rcpp::Named("mu") = m,
	    Rcpp::Named("sigma") = s,
	    Rcpp::Named("gamma") = g
    );
}
