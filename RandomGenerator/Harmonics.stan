data {
  int l;
  int m;
}

parameters {
  real<lower=0, upper=2*3.14159> theta; // controls the dependence between hazards in adjacent intervals
}

model {
  
  if (l == 1 && (m == -1 || m == 1)) {
    target += log(sin(theta)^2);
  } else if (l == 1 && m == 0) {
    target += log(cos(theta)^2);
  }
  
  if (l == 2 && (m == -2 || m == 2)) {
    target += log(sin(theta)^4);
  } else if (l == 2 && (m == -1 || m == 1)) {
    target += log(sin(theta)^2) + log(cos(theta)^2);
  } else if (l == 2 && m == 0) {
    target += log( (3*cos(theta)^2 - 1)^2 );
  }

}
