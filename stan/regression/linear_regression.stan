data{
  int N;
  int F;
  matrix[N,F] x;
  vector[N] y;
  real shrinkage;
  real sigma_upper;
}

parameters{
  vector[F] alpha;
  real beta;
  real<lower=0> sigma;
}

transformed parameters{
  vector[N] yp;
  yp <- x * alpha + beta;
}

model{
  alpha ~ normal(0, shrinkage);
  beta ~ normal(0, shrinkage);
  sigma ~ uniform(0, sigma_upper);
  y ~ normal(yp, sigma);
}
