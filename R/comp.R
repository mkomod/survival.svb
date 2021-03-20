# Comparisons
mse <- function(x) mean(x^2)

# simulations
for (n in c(50, 100, 250, 1000)) {
    for (censoring_lvl in c(0.00, 0.10, 0.25, 0.5, 0.75)) {
	set.seed(1)
	b <- c(1, 1, rep(0, 50))
	p <- length(b)
	X <- matrix(rnorm(n * p), nrow=n)
	y <- runif(nrow(X))
	omega <- 1
	Y <- log(1 - y) / - (exp(X %*% b) * omega)
	delta  <- runif(n) > censoring_lvl   # 0: censored, 1: uncensored
	Y[!delta] <- Y[!delta] * runif(sum(!delta))

	# params
	m <- matrix(rnorm(p), ncol=1)
	s <- matrix(abs(rnorm(p)), ncol=1)
	g <- matrix(runif(p), ncol=1)
	lambda <- 4
	res <- fit(Y, delta, X, lambda, 1, 1, 1e-3, 1e-3, m, s, g, 1000, F)
	# m.1 <- survival::coxph(survival::Surv(Y, delta) ~ ., 
	# 		       data = as.data.frame(X))
	cat("n:", n, 
	    sprintf("\tc: %.3f", censoring_lvl), 
	    sprintf("\tmse: %.5f", mse(res$m - b)), "\n")
    }
}


microbenchmark::microbenchmark(
    fit(Y, delta, X, lambda, 1, 1, 1e-3, 1e-3, m, s, g, 1000, F),
    survival::coxph(survival::Surv(Y) ~ ., data = as.data.frame(X)),
    times=10
)

# ms
#     	   mean  median 
# svb   7336.12 7357.00 
# coxph   68.59   67.72 

