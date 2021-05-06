#' Fit sparse variational Bayesian survival models.
#' 
#' @param Y observation times
#' @param delta Inidcator for right censoring 0: censored, 1: uncensored
#' @param X model matrix
#' @param lambda penalisation term
#' @param mu initial values for means
#' @param sig initial values for standard deviations
#' @param gam initial values for gamma, the inclusion probability
#' @param model one of "partial", "exponential"
#' @param maxiter maximum number of iterations
#' @param tol Stopping tolerance
#' @param verbose print information
#' @param threads number of threads to use
#'
#' @examples
#' n <- 250
#' omega <- 1
#' censoring_lvl <- 0.4
#'
#' set.seed(1)
#' b <- c(1, 1, rep(0, 50))
#' p <- length(b)
#' X <- matrix(rnorm(n * p), nrow=n)
#' y <- runif(nrow(X))
#' Y <- log(1 - y) / - (exp(X %*% b) * omega)
#' 
#' delta  <- runif(n) > censoring_lvl   # 0: censored, 1: uncensored
#' Y[!delta] <- Y[!delta] * runif(sum(!delta))
#' 
#' lambda <- 0.5
#' svb.fit(Y, delta, X, lambda, c(1e-3, 1e-3))
#'
#' @export
svb.fit <- function(Y, delta, X, lambda, params, mu=NULL, sig=NULL, 
    gam=NULL, model="partial", maxiter=1e3, tol=1e-3, verbose=TRUE, threads=1)
{
    # parameter checks
    if (!(model %in% c("partial", "exponential")))
	stop("svb.fit: model unknown")

    if (!is.matrix(X))
	stop("svb.fit: X is not a matrix")
    
    p <- ncol(X)
    if (is.null(mu))
	mu <- matrix(rnorm(p), ncol=1)

    if (is.null(sig))
	sig <- matrix(abs(rnorm(p)), ncol=1)

    if (is.null(gam))
	gam <- matrix(runif(p), ncol=1)


    if (model == "partial") {
	if (length(params) != 2)
	    stop("svb.fit: hyperparameter values not given. Ex: params=c(1e-3, 1e-3)")

	res <- fit_partial(Y, delta, X, lambda, params[1], params[2],
	    mu, sig, gam, maxiter, tol, verbose, threads)
    } 


    if (model == "exponential") {
	if (length(params) != 4)
	    stop("svb.fit: hyperparameter values not given. Ex: params=c(1e-3, 1e-3, 1e-3, 1e-3)")

	res <- fit_exp(Y, delta, X, lambda, params[1], params[2], params[3],
	    params[4], mu, sig, gam, maxiter, verbose, threads)
    }

    return(res)
}
