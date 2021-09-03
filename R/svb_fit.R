#' Fit sparse variational Bayesian proportional hazards model.
#'
#' @param Y Failure times
#' @param delta Censoring indicator, 0: censored, 1: uncensored
#' @param X Design matrix
#' @param lambda Penalisation parameter, default: \code{lambda=0.5}
#' @param a0 Beta distribution parameter, default: \code{a0=1}.
#' @param b0 Beta distribution parameter, default: \code{b0=rnorm(ncol(X))}.
#' @param mu.init Initial value for means, default taken from elasticnet fit
#' @param s.init initial values for standard deviations, default: \code{rep(0.05, ncol(X))}
#' @param g.init initial values for gamma, default: \code{rep(0.5, ncol(X))}
#' @param maxiter maximum number of iterations
#' @param tol convergence tolerance
#' @param alpha The elasticnet mixing parameter used for initialising \code{mu.init}, when \code{alpha=1} the lasso penalty is used and \code{alpha=0} the ridge penalty, values between 0 and 1 give a mixture of the two penalties.
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
svb.fit <- function(Y, delta, X, lambda=0.5, a0=1, b0=ncol(X),
    mu.init=NULL, s.init=rep(0.05, ncol(X)), g.init=rep(0.5, ncol(X)), 
    maxiter=1e3, tol=1e-3, alpha=1, verbose=TRUE)
{
    if (!is.matrix(X)) stop("'X' must be a matrix")
    if (!(lambda > 0)) stop("'lambda' must be greater than 0")
    
    p <- ncol(X)
    if (is.null(mu.init)) {
	y <- survival::Surv(as.matrix(Y), as.matrix(as.numeric(delta)))
	g <- glmnet::glmnet(X, y, family="cox", nlambda=10, alpha=alpha,
	    standardize=FALSE)

	# use the fit for the smallest value of glmnet's lambda seq
	mu.init <- g$beta[ , ncol(g$beta)]
    }
    
    # re-order Y, delta, X by failure time
    # Needed as the log-likelihood is computed on the sorted data
    oY <- order(Y)
    Y <- Y[oY]
    delta <- delta[oY]
    X <- X[oY, ]

    res <- fit_partial(Y, delta, X, lambda, a0, b0,
	mu.init, s.init, g.init, maxiter, tol, verbose)

    return(c(res, lambda=lambda, a0=a0, b0=b0))
}
