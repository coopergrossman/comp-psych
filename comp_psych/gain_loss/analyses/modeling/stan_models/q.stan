data {
  int<lower=1> N;                                   // number of sessions
  int<lower=1> T;                                   // max number of trial across sessions
  array[N] int<lower=1, upper=T> Tsesh;             // array number of trials in each session
  array[N, T] int<lower=0, upper=1> choice;         // choice stimulus A or B
  array[N, T] int<lower=0, upper=1> action;         // L or R action
  array[N, T] int<lower=0, upper=1> outcome;        // win / loss
  array[N, T] int<lower=0, upper=1> block_change;   // new block with new stimuli
  array[N, T] int<lower=0, upper=1> block_loss;     // loss condition block
}
transformed data {
  vector[2] initQ;  // initial values for Q
  initQ = rep_vector(0.0, 2);
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(participant)-parameters
  vector[2] mu_p;
  vector<lower=0>[2] sigma;

  // Session-level raw parameters
  vector[N] a_pr;    // learning rate
  vector[N] beta_pr;  // inverse temperature
}
transformed parameters {
// Transform session-level raw parameters
  vector<lower=0, upper=1>[N] a;
  vector<lower=0, upper=10>[N] beta;

  for (n in 1:N) {
    a[n]    = Phi_approx(mu_p[1] + sigma[1] * a_pr[n]);
    beta[n] = Phi_approx(mu_p[2] + sigma[2] * beta_pr[n]) * 10;
  }
}
model {
  // Hyperparameters
  mu_p  ~ normal(0, 1);
  sigma ~ cauchy(0, 5);

  // individual parameters
  a_pr   ~ normal(0, 1);
  beta_pr ~ normal(0, 1);

  // session loop and trial loop
  for (n in 1:N) {
    vector[2] Q;              // expected value
    real PE;                  // prediction error
    vector[Tsesh[n]] Qdiff;   // relative value

    for (t in 1:(Tsesh[n])) {

      if (block_change[n,t] == 1){    // reset Q values at new block
        Q = initQ;
      }

      Qdiff[t] = Q[2] - Q[1];         // compute relative value

      if (choice[n,t] == 1) {         // if stimulus B
        PE = outcome[n, t] - Q[2];    // compute RPE
        Q[2] += a[n] * PE;            // update chosen value with learning rule
      }else{
        PE = outcome[n, t] - Q[1];    // if stimulus A
        Q[1] += a[n] * PE;
      }
    }
    choice[n, 1:Tsesh[n]] ~ bernoulli_logit(beta[n] * Qdiff);   // run objective function once per session
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_a;
  real<lower=0, upper=10> mu_beta;

  // For log likelihood calculation
  array[N] real log_lik;

  // For posterior predictive check
  array[N,T] real y_pred;

  // Set all posterior predictions to 0 (avoids NULL values)
  for (n in 1:N) {
    for (t in 1:T) {
      y_pred[n, t] = -1;
    }
  }

  mu_a    = Phi_approx(mu_p[1]);
  mu_beta = Phi_approx(mu_p[2])*10;


  { // local section, this saves time and space
    for (n in 1:N) {
      vector[2] Q; // expected value
      real PE;      // prediction error
      vector[Tsesh[n]] Qdiff;

      log_lik[n] = 0;

      for (t in 1:(Tsesh[n])) {

        if (block_change[n,t] == 1){
          Q = initQ;
        }

        Qdiff[t] = Q[2] - Q[1];

        // compute log likelihood of current trial
        log_lik[n] = log_lik[n] + bernoulli_logit_lpmf(choice[n, t] | beta[n] * Qdiff[t]);

        // generate posterior prediction for current trial
        y_pred[n, t] = categorical_rng(softmax(beta[n] * Q));

        if (choice[n,t] == 1) {
          PE = outcome[n, t] - Q[2];
          Q[2] += a[n] * PE;
        }else{
          PE = outcome[n, t] - Q[1];
          Q[1] += a[n] * PE;
        }
      }
    }
  }
}