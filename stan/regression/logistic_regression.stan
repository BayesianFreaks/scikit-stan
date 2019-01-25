data{
    int n;
    int f;
    matrix[n,f] x;
    int y[n];
    real shrinkage;
}

parameters{
    vector[f] alpha;
    real beta;
}

transformed parameters{
    vector[n] yp;
    yp <- x*alpha + beta;
}

model{
    alpha ~ normal(0, shrinkage);
    beta ~ normal(0, shrinkage);
    y ~ bernoulli_logit(yp);
}
