# Analytic status

## Analytic pieces that should be used in the manuscript

1. Analytic island:

```text
d(theta + 1) = 3,
z = 1,
sqrt(3) * mu = 4.
```

2. Visible Yang--Mills zero mode:

```text
b0(u) = 4(1 - u^-2)
w1(u) = u^2/(1 + u^2)^2
```

3. Subleading temporal correction:

```text
b2(u) = 71/6720 * (1 - u^-2)
        + (5 + 7u^2 - 9u^4 - 3u^6)/(96u^2(1+u^2)^3)
```

4. D=3 and D=4 leading metric backreaction at the analytic locus, used as validation gates.

5. Minimal hidden-SU(2) scaling at `mu_X=0`:

```text
mu_c(alpha_dm)/mu_c(0) = sqrt(1 - alpha_dm^2/4)
HEE_coeff(alpha_dm)/HEE_coeff(0) = 1 - alpha_dm^2/4
```

## Numerical-only pieces in the current workflow

1. General off-locus `(theta,z)` Sturm--Liouville zero modes.
2. General off-locus metric response and RT/interval integrals.
3. Tensor-portal HEE response under the `Z(Phi)` source-reweighting prescription.
4. Full profile-level dark-sector re-solves for `omega(u), b(u)` are not part of the current source-of-record output.
