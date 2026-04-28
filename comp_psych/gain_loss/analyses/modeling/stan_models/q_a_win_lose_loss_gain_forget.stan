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
  vector[6] mu_p;
  vector<lower=0>[6] sigma;

  // Session-level raw parameters
  vector[N] a_win_gain_pr;      // win in gain block learning rate
  vector[N] a_lose_gain_pr;     // lose in gain block learning rate
  vector[N] a_win_loss_pr;      // win in loss block learning rate
  vector[N] a_lose_loss_pr;     // lose in loss block learning rate
  vector[N] forget_pr;          // forgetting rate
  vector[N] beta_pr;            // inverse temperature
}
transformed parameters {
// Transform session-level raw parameters
  vector<lower=0, upper=1>[N] a_win_gain;
  vector<lower=0, upper=1>[N] a_lose_gain;
  vector<lower=0, upper=1>[N] a_win_loss;
  vector<lower=0, upper=1>[N] a_lose_loss;
  vector<lower=0, upper=1>[N] forget;
  vector<lower=0, upper=10>[N] beta;

  for (n in 1:N) {
    a_win_gain[n]  = Phi_approx(mu_p[1] + sigma[1] * a_win_gain_pr[n]);
    a_lose_gain[n] = Phi_approx(mu_p[2] + sigma[2] * a_lose_gain_pr[n]);
    a_win_loss[n]  = Phi_approx(mu_p[3] + sigma[3] * a_win_loss_pr[n]);
    a_lose_loss[n] = Phi_approx(mu_p[4] + sigma[4] * a_lose_loss_pr[n]);
    forget[n]      = Phi_approx(mu_p[5] + sigma[5] * forget_pr[n]);
    beta[n]        = Phi_approx(mu_p[6] + sigma[6] * beta_pr[n]) * 10;
  }
}
model {
  // Hyperparameters
  mu_p  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // individual parameters
  a_win_gain_pr  ~ normal(0, 1);
  a_lose_gain_pr ~ normal(0, 1);
  a_win_loss_pr  ~ normal(0, 1);
  a_lose_loss_pr ~ normal(0, 1);
  forget_pr      ~ normal(0, 1);
  beta_pr        ~ normal(0, 1);

  // session loop and trial loop
  for (n in 1:N) {
    vector[2] Q; // expected value
    real PE;      // prediction error
    vector[Tsesh[n]] Qdiff;
    real a;

    for (t in 1:(Tsesh[n])) {

      if (block_change[n,t] == 1){
        Q = initQ;
      }

      Qdiff[t] = Q[2] - Q[1];
      choice[n, t] ~ bernoulli_logit(beta[n] * Qdiff[t]);

      if (block_loss[n,t] == 0){            // gain block
        if (outcome[n,t] == 1){
          a = a_win_gain[n];                // win
        }else{
          a = a_lose_gain[n];               // lose
        }
      }else{                                // loss block
        if (outcome[n,t] == 1){
          a = a_win_loss[n];                // win
        }else{
          a = a_lose_loss[n];               // lose
        }
      }

      if (choice[n,t] == 1) {
        PE = outcome[n, t] - Q[2];
        Q[2] += a * PE;
        Q[1] = Q[1] * forget[n];
      }else{
        PE = outcome[n, t] - Q[1];
        Q[1] += a * PE;
        Q[2] = Q[2] * forget[n];
      }
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_a_win_gain;
  real<lower=0, upper=1> mu_a_lose_gain;
  real<lower=0, upper=1> mu_a_win_loss;
  real<lower=0, upper=1> mu_a_lose_loss;
  real<lower=0, upper=1> mu_forget;
  real<lower=0, upper=10> mu_beta;

  mu_a_win_gain  = Phi_approx(mu_p[1]);
  mu_a_lose_gain = Phi_approx(mu_p[2]);
  mu_a_win_loss  = Phi_approx(mu_p[3]);
  mu_a_lose_loss = Phi_approx(mu_p[4]);
  mu_forget      = Phi_approx(mu_p[5]);
  mu_beta        = Phi_approx(mu_p[6])*10;

}