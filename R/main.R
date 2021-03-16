Rcpp::sourceCpp("../src/survival_svb.cpp", verbose=T, rebuild=T)

# init data
set.seed(1)
b <- c(1, 1, rep(0, 50))
n <- 100
p <- length(b)
X <- matrix(rnorm(n * p), nrow=n)
y <- runif(nrow(X))
omega <- 1
Y <- log(1 - y) / - (exp(X %*% b) * omega)


m <- matrix(rnorm(p), ncol=1)
s <- matrix(abs(rnorm(p)), ncol=1)
g <- matrix(runif(p), ncol=1)


fit(Y, X, omega, lambda, 1, 1, m, s, g, 100, F)
