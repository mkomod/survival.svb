#include "RcppArmadillo.h"

arma::vec
initialise_P(arma::vec m, arma::vec s, arma::mat X)
{
    int n = X.n_rows;
    arma::vec P(n, arma::fill::zeros);

    for (int i = 0; i < n; ++i) {
	arma::rowvec x = X.row(i);
	double xSx = (x * arma::diagmat(s) * x.t()).eval().at(0, 0);
	P(i) = dot(x, m) + 1/2 + xSx;
    }

    return P;
}

arma::vec
sub_from_P(arma::vec P, double m, double s, arma::vec x_j) 
{
    for (int i = 0; i < P.n_rows; ++i) {
	P(i) -= x_j(i) * m + 1/2 * pow(s * x_j(i), 2);
    }
    return P;
}

arma::vec
add_to_P(arma::vec P, double m, double s, arma::vec x_j) 
{
    for (int i = 0; i < P.n_rows; ++i) {
	P(i) += x_j(i) * m + 1/2 * pow(s * x_j(i), 2);
    }
    return P;
}



double
update_likelihood(double m, double s, double omega, arma::vec T, 
	arma::vec x_j, arma::vec P)
{
    double res = 0;
    for (int i = 0; i < x_j.n_rows; ++i) {
	res += omega * T(i) * exp(x_j(i) * m + 1/2 * pow(x_j(i) * s, 2) + P(i)) 
	    - m * x_j(i);
    }
    return res;
}


double
update_variational(double m, double s, double l)
{
    double m_s = m / s;
    return l * s * sqrt(2 / PI) * exp( - pow(m_s, 2)) +
	l * m * (1 - R::pnorm(-m_s, 0, 1, 0, 0));
}


double 
update_m(double m, const double s, double omega, double l, arma::vec T,
	arma::vec x_j, arma::vec P) 
{
    
}


double 
update_s(double s, double m, double omega, double l, arma::vec T,
	arma::vec x_j, arma::vec P) 
{
    
}


double update_g() {

}


Rcpp::List 
fit(arma::colvec T, arma::mat X, double omega)
{
    // init m, s, g
    int p = X.n_cols;
    int n = X.n_rows;
    arma::vec m, s, g = arma::vec(p, arma::fill::zeros);
    
    // P.i := x_j.t * m + 1/2 x_j * S * x_j.t()
    arma::vec P = initialise_P(m, s, X);

    for (int j = 0; j < X.n_cols; ++j) {
    // opt m
	P = sub_from_P(P, m(j), s(j), X.col(j));
	m(j) = update_m();
	P = add_to_P(P, m(j), s(j), X.col(j));

	// opt s
	P = sub_from_P(P, m(j), s(j), X.col(j));
	s(j) = update_s();
	P = add_to_P(P, m(j), s(j), X.col(j));

	// opt g
	g(j) = update_g();

    }

    return Rcpp::List::create(
	    Rcpp::Named("mu") = m,
	    Rcpp::Named("sigma") = s,
	    Rcpp::Named("gamma") = g
    );
}
