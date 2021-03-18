# Comparisons
mse <- function(x) mean(x^2)


# test without censoring
for (n in c(50, 100, 250, 1000)) {
    set.seed(1)
    b <- c(1, 1, rep(0, 50))
    p <- length(b)
    X <- matrix(rnorm(n * p), nrow=n)
    y <- runif(nrow(X))
    omega <- 1
    Y <- log(1 - y) / - (exp(X %*% b) * omega)
    delta  <- rep(1, n)

    m <- matrix(rnorm(p), ncol=1)
    s <- matrix(abs(rnorm(p)), ncol=1)
    g <- matrix(runif(p), ncol=1)

    lambda <- 4
    res <- fit(Y, delta, X, omega, lambda, 1, 1, m, s, g, 1000, F)
    m.1 <- survival::coxph(survival::Surv(Y) ~ ., data = as.data.frame(X))

    cat("Sparse Surv: ", mse(res$m - b), "\n",
	"CoxPH: ", mse(coef(m.1) - b), "\n")
}

plot(b, col="grey", pch=20, ylim=c(-0.2, 1.2), ylab=expression(beta))
points(res$m, col="red", pch=20)

# Sparse Surv:  0.0073
# CoxPH:  (NA) 7232.961
# Sparse Surv:  0.0035
# CoxPH:  0.0894
# Sparse Surv:  0.0029
# CoxPH:  0.0109
# Sparse Surv:  0.00088
# CoxPH:  0.00135

microbenchmark::microbenchmark(
    fit(Y, delta, X, omega, lambda, 1, 1, m, s, g, 1000, F),
    survival::coxph(survival::Surv(Y) ~ ., data = as.data.frame(X)),
    times=10
)

# Unit: millisecondsn asymptoti
#        min        lq      mean    median        uq        max   neval cld
# svb   602.2979 620.6298 741.68978 640.76466 808.35299  1513.036  100   b
# coxph 45.80873  63.3845  70.65804  67.79749  74.95692   139.988  100   a


# test with censoring
for (n in c(50, 100, 250, 1000)) {
    for (censoring_lvl in c(0.10, 0.25, 0.5, 0.75)) {
	set.seed(1)
	b <- c(1, 1, rep(0, 50))
	p <- length(b)
	X <- matrix(rnorm(n * p), nrow=n)
	y <- runif(nrow(X))
	omega <- 1
	Y <- log(1 - y) / - (exp(X %*% b) * omega)
	delta  <- runif(n) < censoring_lvl   # 0: censored, 1: uncensored
	Y[!delta] <- Y[!delta] * runif(sum(!delta))

	# params
	m <- matrix(rnorm(p), ncol=1)
	s <- matrix(abs(rnorm(p)), ncol=1)
	g <- matrix(runif(p), ncol=1)
	lambda <- 4
	res <- fit(Y, delta, X, omega, lambda, 1, 1, m, s, g, 1000, F)
	m.1 <- survival::coxph(survival::Surv(Y, delta) ~ ., 
			       data = as.data.frame(X))
	cat("Sparse Surv: ", mse(res$m - b), "\n",
	"CoxPH: ", mse(coef(m.1) - b), "\n")

    }
}



