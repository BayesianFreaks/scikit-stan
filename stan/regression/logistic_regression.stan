data {
  int N;
  int F;
  matrix[N, F] x;
  int y[N];
  real shrinkage;
}

parameters {
  vector[F] alpha;
  real beta;
}

transformed parameters {
  vector[F] yp;
  yp <- x * alpha + beta;
}

model {
  alpha ~ normal(0, shrinkage);
  beta ~ normal(0, shrinkage);
  y ~ bernoulli_logit(yp);
}
