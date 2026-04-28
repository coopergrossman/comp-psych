data {
  int<lower=1> N;
  int<lower=1> T;
  array[N] int<lower=1, upper=T> Tsesh;
  array[N, T] int<lower=0, upper=1> choice;
  array[N, T] int<lower=0, upper=1> action;
  array[N, T] int<lower=0, upper=1> outcome;
  array[N, T] int<lower=0, upper=1> block_change;
  array[N, T] int<lower=0, upper=1> block_loss;
}
transformed data {
  vector[2] initQ;  // initial values for Q
  initQ = rep_vector(0.0, 2);
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(participant)-parameters
  vector[3] mu_p;
  vector<lower=0>[3] sigma;

  // Session-level raw parameters
  vector[N] a_win_pr;     // learning rate after win
  vector[N] a_lose_pr;    // learning rate after loss
  vector[N] beta_pr;      // inverse temperature
}
transformed parameters {
// Transform session-level raw parameters
  vector<lower=0, upper=1>[N] a_win;
  vector<lower=0, upper=1>[N] a_lose;
  vector<lower=0, upper=10>[N] beta;

  for (n in 1:N) {
    a_win[n] = Phi_approx(mu_p[1] + sigma[1] * a_win_pr[n]);
    a_lose[n]= Phi_approx(mu_p[2] + sigma[2] * a_lose_pr[n]);
    beta[n]  = Phi_approx(mu_p[3] + sigma[3] * beta_pr[n]) * 10;
  }
}
model {
  // Hyperparameters
  mu_p  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // individual parameters
  a_win_pr  ~ normal(0, 1);
  a_lose_pr ~ normal(0, 1);
  beta_pr   ~ normal(0, 1);

  // session loop and trial loop
  for (n in 1:N) {
    vector[2] Q; // expected value
    real PE;      // prediction error
    vector[Tsesh[n]] Qdiff;
    real a

    for (t in 1:(Tsesh[n])) {

      if (block_change[n,t] == 1){
        Q = initQ;
      }

      Qdiff[t] = Q[2] - Q[1];
      choice[n, t] ~ bernoulli_logit(beta[n] * Qdiff[t]);

      if (outcome[n,t] == 1){
        a = a_win[n];
      }else{
        a = a_lose[n];
      }

      if (choice[n,t] == 1) {
        PE = outcome[n, t] - Q[2];
        Q[2] += a * PE;
      }else{
        PE = outcome[n, t] - Q[1];
        Q[1] += a * PE;
      }
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_a_win;
  real<lower=0, upper=1> mu_a_lose;
  real<lower=0, upper=10> mu_beta;

  mu_a_win  = Phi_approx(mu_p[1]);
  mu_a_lose = Phi_approx(mu_p[2]);
  mu_beta   = Phi_approx(mu_p[3])*10;

}