#' Fit sparse variational Bayesian survival models.
#'
#' @param Y failure times
#' @param delta censoring indicator, 0: censored, 1: uncensored
#' @param X design matrix
#' @param lambda penalisation hyperparameter
#' @param model one of: \itemize{
#' 	\item{\code{"partial"}} (defualt),
#'	\item{\code{"exponential"}}
#' }
#' @param params additional hyperparameters, defualts: \itemize{
#' 	\item{partial}: \code{c(a_0=1, b_0=ncol(X))},
#' 	\item{exponential}: \code{c(a_0=1, b_0=ncol(X), a_omega=1, b_omega=1)}.
#' }
#' @param mu.init initial values for means, default: \code{rnorm(ncol(X))}
#' @param s.init initial values for standard deviations, default: \code{rep(0.2, ncol(X))}
#' @param g.init initial values for gamma, default: \code{rep(0.5, ncol(X))}
#' @param maxiter maximum number of iterations
#' @param tol convergence tolerance
#' @param verbose print additional information
#'
#' @examples
#' n <- 200
#' omega <- 1
#' censoring_lvl <- 0.4
#'
#' set.seed(1)
#' b <- c(1, 1, rep(0, 348))
#' p <- length(b)
#' X <- matrix(rnorm(n * p), nrow=n)
#' y <- runif(nrow(X))
#' Y <- log(1 - y) / - (exp(X %*% b) * omega)
#' 
#' delta  <- runif(n) > censoring_lvl   # 0: censored, 1: uncensored
#' Y[!delta] <- Y[!delta] * runif(sum(!delta))
#' 
#' lambda <- 0.5
#' f <- svb.fit(Y, delta, X)
#'
#' @export
svb.fit <- function(Y, delta, X, lambda=0.5, model="partial", params=NULL, 
    mu.init=NULL, s.init=NULL, g.init=NULL, maxiter=1e3, tol=1e-3, verbose=TRUE)
{

    if (!(model %in% c("partial", "exponential"))) stop("'model' unknown")
    if (!is.matrix(X)) stop("'X' must be a matrix")
    if (!(lambda > 0)) stop("'lambda' must be greater than 0")
    
    p <- ncol(X)

    if (is.null(mu.init)) mu.init <- matrix(rnorm(p), ncol=1)
    if (is.null(s.init)) s.init <- matrix(rep(0.2, p))
    if (is.null(g.init)) g.init <- matrix(rep(0.5, p))
    
    if (model == "partial") {
	if (is.null(params)) params <- c(1, p)
	if (length(params) != 2) stop("partial model requires two 'params'")

	res <- fit_partial(Y, delta, X, lambda, params[1], params[2],
	    mu.init, s.init, g.init, maxiter, tol, verbose)
    } 

    if (model == "exponential") {
	if (is.null(params)) params <- c(1, p, 1, 1)
	if (length(params) != 4) stop("exponential model requires two 'params'")

	res <- fit_exp(Y, delta, X, lambda, params[1], params[2], params[3],
	    params[4], mu.init, s.init, g.init, maxiter, verbose)
    }

    return(res)
}
