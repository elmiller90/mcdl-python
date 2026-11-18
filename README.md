# MCDL\-Python

A Python research implementation of nonnegative multiscale convolutional dictionary learning for recovering sparse, finite\-duration trajectories in position\-scale\-time coefficient fields.

The project is based on the MATLAB implementation accompanying the space\-time multiscale convolutional dictionary\-learning work of Banco, Miller, and collaborators. The Python implementation is independently testable and is not intended as a literal line\-by\-line translation.

## Current research objective

The current coefficient\-recovery model solves

\\widehat X

=

\\arg\\min\_\{X\\ge0\}

\\left\\\{

\\frac12\\\|\\mathcal A\_DX\-S\\\|\_F^2

\+

\\lambda\_S R\_\{\\log\}(X)

\+

\\lambda\_T R\_T(X)

\\right\\\},where:

- (\\mathcal A\_D) is the multiscale convolution operator for a fixed base dictionary (D);
- (R\_\{\\log\}) is a nonconvex logarithmic sparsity penalty;
- (R\_T) is a temporal regularizer for sparse trajectories in position\-scale\-time.

The active temporal model is

R\_T(X)

=

R\_\{\\mathrm\{motion\}\}(X)

\+

\\beta\_VR\_\{\\mathrm\{spread\}\}(X).The motion term regularizes the trajectory of each frame’s position and log\-scale barycenter. The spread term suppresses diffuse or mixed representations having a favorable barycenter. A smooth squared activity gate permits trajectories to appear and disappear without a first\-order birth or death penalty.

## Validated components

### Multiscale convolution

- Multiscale resampling and convolution operators
- Structured coefficient convention (X\\in\\mathbb R^\{L\\times K\\times J\\times T\})
- Forward and adjoint implementations
- Zero\-padded convolution convention matching the MATLAB experiments
- Composite operator adjoint tests

### Sparse coefficient recovery

- Nonnegative coefficient constraints
- Logarithmic sparsity penalty
- Continuation from an approximately (\\ell\_1) penalty to the nonconvex log penalty
- ADMM splitting
- Nonnegative L\-BFGS\-B coefficient update
- Adaptive ADMM penalty parameter

### Temporal regularization

- Position and log\-scale barycentric motion
- Smooth squared activity gate satisfying g(0)=g'(0)=0
- Gated within\-frame variance penalty
- Explicit analytical gradients
- Directional finite\-difference validation near machine precision
- Exact stationary, moving, birth, and death geometry tests

### Nonconvex continuation

The solver supports full\-state continuation of

(X,Y,U,\\rho,a)between nearby parameter values.

A compatibility test verifies that the stateful solver reproduces the original cold\-start ADMM solution exactly when no warm start is supplied.

### Automatic parameter selection

The project includes a hierarchical adaptive selector for

(\\lambda\_S,\\lambda\_T,\\beta\_V).Selection uses observable quantities only:

1. expected\-noise discrepancy;
2. residual\-regularizer tradeoff geometry;
3. solution stability over neighboring parameter values;
4. connected stable parameter regions.

Two unsupervised rules are currently compared:

- discrepancy and stability\-region selection;
- nested L\-curve selection.

Synthetic truth is revealed only after selection. A searched synthetic oracle is retained as a post\-hoc benchmark for separating selector failure from candidate\-family or identifiability failure.

## Noise convention

Monte Carlo noise is generated as

N\_i\\overset\{\\mathrm\{iid\}\}\{\\sim\}\\mathcal N(0,\\sigma^2).The prescribed variance is chosen from the requested SNR using expected noise energy. Individual realizations are not rescaled to have exactly the requested SNR.

Parameter selection uses

D\_\{\\mathrm\{exp\}\}

=

\\frac12N\_\{\\mathrm\{obs\}\}\\sigma^2and the discrepancy ratio

r\_D

=

\\frac\{

\\frac12\\\|\\mathcal A\_DX\-S\\\|\_F^2

\}\{

D\_\{\\mathrm\{exp\}\}

\}.The realized noise energy is recorded only for post\-hoc diagnostics and is not available to the selector.


Current numerical status
The fixed-dictionary coefficient-recovery phase has reached a validated development milestone.
The validation now includes:

- multiscale forward and adjoint operators;
- nonnegative log-sparse coefficient recovery;
- barycentric motion regularization;
- smooth finite-duration activity gating;
- gated within-frame spread regularization;
- analytical temporal gradients validated by directional finite differences;
- full-state ADMM continuation;
- expected-noise discrepancy selection;
- hierarchical stability and L-curve parameter selectors;
- paired temporal-versus-nontemporal ablation;
- multi-SNR Monte Carlo characterization.
Development-scale Monte Carlo study
The adaptive selector was evaluated on
5\ \text{SNR levels}\times10\ \text{independent realizations}
=
50\ \text{datasets},using
\mathrm{SNR}\in\{20,17,14,11,8\}\ {\rm dB}.Regularization parameters were selected independently for every realization. Synthetic truth was used only after parameter selection.
Under nested L-curve selection, exact position-scale trajectory recovery was:
SNR (dB)	Exact trajectory rate
20	100%
17	90%
14	70%
11	20%
8	10%

The same selector recovered exactly eight significant coefficients in all 50 trials.
Position recovery was substantially more robust than scale recovery. Performance degradation below approximately 14 dB is dominated by ambiguity among neighboring scale atoms rather than loss of the sparse temporal support.
The searched synthetic oracle achieved exact trajectory rates of
[100,\;90,\;70,\;20,\;20]\%.Thus, through 11 dB, the adaptive L-curve selector achieved the same exact-trajectory success probability as the searched oracle candidate family in this development study.
Temporal-versus-nontemporal ablation
A paired comparison was performed using exactly the same 50 noise realizations.
The non-temporal baseline solved
\min_{X\ge0}
\frac12\|\mathcal A_DX-S\|_F^2
+
\lambda_SR_{\log}(X)with (\lambda_S) selected independently for each dataset.
The temporal estimator additionally used
\lambda_T
\left[
R_{\mathrm{motion}}(X)
+
\beta_VR_{\mathrm{spread}}(X)
\right].The clearest adaptive benefit of temporal regularization was structural.
Under L-curve selection, the non-temporal exact-eight-support rates were
[90,\;70,\;70,\;30,\;50]\%,whereas the temporal estimator achieved
\boxed{100\%}exact-eight-support recovery at every tested SNR.
Exact trajectory rates changed from
[90,\;80,\;60,\;10,\;10]\%without temporal regularization to
[100,\;90,\;70,\;20,\;10]\%with temporal regularization.
Median coefficient errors under L-curve selection changed from
[0.0272,\;0.0317,\;0.1612,\;0.5445,\;0.7080]to
[0.0163,\;0.0221,\;0.0338,\;0.5049,\;0.7081].The temporal model therefore shows its most consistent benefit in suppressing extraneous coefficients and preserving the intended one-coefficient-per-active-frame geometry. At moderate SNR, this structural benefit is accompanied by improved trajectory recovery and substantially lower median coefficient error.
Oracle model comparison
The best searched non-temporal and temporal candidates were also compared post hoc.
Median non-temporal oracle coefficient errors were
[0.01261,\;0.01915,\;0.03263,\;0.49351,\;0.55932],while median temporal-oracle errors were
[0.00982,\;0.01396,\;0.02526,\;0.49326,\;0.49942].The corresponding exact trajectory rates were identical:
[100,\;90,\;70,\;20,\;20]\%.The oracle comparison indicates that temporal regularization frequently improves coefficient accuracy even when both model families recover the same discrete trajectory.
The temporal search includes (\lambda_T=0), so it can revert naturally to the non-temporal solution when temporal regularization is not beneficial.
Interpretation of the coefficient-recovery milestone
The current evidence supports three conclusions.

1. Sparse temporal structure is robust.
The barycentric motion, spread, and activity-gating model reliably produces compact finite-duration coefficient tracks.

1. Temporal regularization adds measurable structural value.
Its most consistent benefit is elimination of extraneous coefficients and stabilization of the intended one-coefficient-per-frame representation.

1. Low-SNR failure is primarily a scale-identification problem.
Position and activity remain substantially more robust than discrete scale recovery as SNR decreases.
The current 10-trial-per-SNR experiment is a development study. The original MATLAB study used 100 noise realizations per SNR, so publication-scale experiments may eventually increase the number of Monte Carlo realizations after the complete dictionary-learning pipeline is available.

## Project layout

### Root modules

The project root contains the active numerical implementation:

- multiscale\_convolution.py: multiscale forward and adjoint operator
- zero\_padded\_convolution.py: convolution engine
- x\_update.py: smooth nonnegative coefficient update
- log\_admm.py: original log\-penalty ADMM solver
- log\_admm\_stateful\_compat.py: full\-state continuation\-compatible solver
- barycentric\_transition.py: motion, activity\-gate, and spread regularization
- local\_graph.py: shift\-scale geometry
- local\_transition.py: archived transition geometry still used by validation code
- gaussian\_experiment.py: synthetic problem and variance\-controlled noise utilities

### Tests

tests/ contains numerical validation for:

- operators and adjoints;
- convolution conventions;
- temporal gradients;
- ADMM compatibility;
- end\-to\-end recovery;
- historical softmin, Sinkhorn, and transition formulations.

Tests should be run as Python modules from the project root.

### Active experiments

experiments/barycentric/ contains finite\-track and continuation experiments.

experiments/parameter\_selection/ contains:

- regularization grids;
- discrepancy and L\-curve analyses;
- the adaptive hierarchical selector;
- the Monte Carlo wrapper.

experiments/diagnostics/ contains scale\-confusion and comparative diagnostics.

### Archived approaches

archive/sinkhorn/ preserves the local optimal\-transport implementation and benchmarks.

archive/gated\_markov/ preserves the Markov/Hellinger transition experiments.

These approaches are retained because their negative results motivate the final barycentric model.

### Documentation

docs/barycentric\_temporal\_regularization.qmd is the living mathematical and numerical specification.

The Quarto report contains:

- model derivations;
- gradient formulas;
- optimization details;
- parameter\-selection methodology;
- validated numerical results;
- Monte Carlo analysis as it becomes available.

## Running the current workflow

The principal workflow is:

1. run the numerical validation tests;
2. run the single\-realization adaptive selector;
3. inspect its JSON summary;
4. run the Monte Carlo wrapper;
5. render the Quarto technical report.

The Monte Carlo wrapper saves one summary per trial and skips completed trials when restarted. This permits long studies to resume safely after interruption.


Current milestone
The fixed-dictionary coefficient-recovery phase is now considered validated at development scale.
Validated components are:

- multiscale convolution and adjoints;
- log-sparse nonnegative recovery;
- barycentric temporal motion;
- finite-duration activity gating;
- within-frame spread regularization;
- analytical gradients;
- full-state nonconvex continuation;
- expected-noise adaptive parameter selection;
- multi-SNR Monte Carlo characterization;
- temporal-versus-nontemporal paired ablation;
- oracle model comparison.
The authoritative mathematical description is maintained in:
docs/barycentric_temporal_regularization.qmd
Experimental summaries are stored under:
results/barycentric_monte_carlo/
Next phase: dictionary learning
The next major development phase is estimation of the base dictionary atoms rather than treating the dictionary as known.
The original MCDL formulation represents each scaled atom as
d_{k,j}
=
\alpha_{k,j}M_jd_k,
\qquad
\alpha_{k,j}
=
\frac1{\|M_jd_k\|_2}.During a dictionary update, the coefficient tensor (X) and normalization variables (\alpha) are held fixed. The resulting map from the base dictionary to the signal is linear:
(\mathcal B_{X,\alpha}D)_t
=
\sum_{k,j}
(\alpha_{k,j}M_jd_k)*x_{k,j,t}.The dictionary ADMM subproblem is therefore quadratic:
D^{(\ell+1)}
=
\arg\min_D
\left\{
\frac12\|\mathcal B_{X,\alpha}D-S\|_F^2
+
\frac{\rho_D}{2}
\|D-G^{(\ell)}+H^{(\ell)}\|_F^2
\right\}.Its normal equation is
\left(
\mathcal B_{X,\alpha}^{*}
\mathcal B_{X,\alpha}
+
\rho_DI
\right)D
=
\mathcal B_{X,\alpha}^{*}S
+
\rho_D(G-H).The base atom is then projected onto the feasible set defined by:

- prescribed finite support;
- nonnegativity;
- unit (\ell_2) norm.
After projection,
\alpha_{k,j}
=
\frac1{\|M_jg_k\|_2}is recomputed explicitly.
Planned dictionary-learning milestones

1. Implement the fixed-(X,\alpha) dictionary forward operator.
2. Implement and verify its adjoint.
3. Validate the dictionary data-fidelity gradient.
4. Implement the CG/CGLS quadratic dictionary update.
5. Implement support, nonnegativity, and unit-norm projection.
6. Validate the explicit scale-normalization update.
7. Recover a perturbed base atom with (X=X_{\rm true}) fixed.
8. Alternate coefficient and dictionary updates.
9. Reproduce the original MATLAB single-Gaussian experiments.
10. Repeat temporal-versus-nontemporal evaluation with learned rather than known dictionaries.
No changes to the currently validated coefficient-side MultiscaleConvolutionOperator should be made until the dictionary-side operator and its adjoint have been validated independently.
Next development branch
The recommended next branch is:
feature/dictionary-learning
The dictionary-learning phase should begin from the validated state documented here and in the Quarto technical report.


## Development policy

A component is considered validated only after its mathematical adjoint or gradient passes numerical testing and its behavior is confirmed in a controlled synthetic experiment.

Validated milestones should be preserved through:

- a focused Git commit;
- an update to this README;
- an update to the Quarto report when the mathematical model or scientific conclusions change materially.

Experimental code and negative results should be archived rather than silently deleted.

## Reference implementation

The original MATLAB implementation is available in the dbanco/CHESS\-Research repository under MATLAB/MultiscaleDL.

The Python project is designed to provide:

- explicit mathematical structure;
- independently validated operators and gradients;
- reproducible parameter selection;
- restartable Monte Carlo experiments;
- a clear path from coefficient recovery to full dictionary learning.

