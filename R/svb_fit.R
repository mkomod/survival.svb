#' Fit sparse variational Bayesian proportional hazards model.
#'
#' @param Y failure times
#' @param delta censoring indicator, 0: censored, 1: uncensored
#' @param X design matrix
#' @param lambda penalisation hyperparameter
#' @param params additional hyperparameters, defualt: \code{c(a_0=1, b_0=ncol(X))}. For large \code{p} (~10,000) we suggest increasing \code{a_0}.
#' @param mu.init initial values for means, default: \code{rnorm(ncol(X))}.
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
#' f <- svb.fit(Y, delta, X)
#'
#' @export
svb.fit <- function(Y, delta, X, lambda=0.5, params=c(1, ncol(X)),
    mu.init=NULL, s.init=NULL, g.init=NULL, maxiter=1e3, tol=1e-3, verbose=TRUE)
{

    if (!is.matrix(X)) stop("'X' must be a matrix")
    if (!(lambda > 0)) stop("'lambda' must be greater than 0")
    
    p <- ncol(X)

    if (is.null(mu.init)) mu.init <- matrix(rnorm(p), ncol=1)
    if (is.null(s.init)) s.init <- matrix(rep(0.2, p))
    if (is.null(g.init)) g.init <- matrix(rep(0.5, p))
    
    if (model == "partial") {
	if (is.null(params)) params <- c(1, p)
	if (length(params) != 2) stop("partial model requires two 'params'")
	
	# re-order Y, delta, X by failure time.
	oY <- order(Y)
	Y <- Y[oY]
	delta <- delta[oY]
	X <- X[oY, ]

	res <- fit_partial(Y, delta, X, lambda, params[1], params[2],
	    mu.init, s.init, g.init, maxiter, tol, verbose)
    } 

    return(c(res, lambda=0.5, a0=params[1], b0=params[2]))
}
