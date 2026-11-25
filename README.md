# MCDL-Python

A Python research implementation of nonnegative multiscale convolutional dictionary learning (MCDL) for recovering sparse, finite-duration trajectories in position-scale-time coefficient fields.

The project is based on the MATLAB implementation accompanying the space-time multiscale convolutional dictionary-learning work of Banco, Miller, and collaborators. The Python implementation emphasizes explicit mathematical structure, independent numerical validation, reproducibility, and controlled experimental development rather than literal line-by-line translation.

## Current research objective

The full problem consists of jointly estimating a multiscale base dictionary \(D\) and a structured coefficient tensor \(X\).

For fixed dictionary \(D\), coefficient recovery solves

\[
\widehat X
=
\arg\min_{X\ge0}
\left\{
\frac12\|\mathcal A_DX-S\|_F^2
+
\lambda_S R_{\log}(X)
+
\lambda_T R_T(X)
\right\},
\]

where:

- \(\mathcal A_D\) is the multiscale convolution operator for the current base dictionary;
- \(R_{\log}\) is a nonconvex logarithmic sparsity penalty;
- \(R_T\) is a temporal regularizer for sparse trajectories in position-scale-time.

The active temporal model is

\[
R_T(X)
=
R_{\mathrm{motion}}(X)
+
\beta_V R_{\mathrm{spread}}(X).
\]

The motion term regularizes each frame's position and log-scale barycenter. The spread term suppresses diffuse or mixed representations that nevertheless have a favorable barycenter. A smooth squared activity gate permits finite-duration trajectories to appear and disappear without a first-order birth or death penalty.

Dictionary learning uses the multiscale atom model

\[
d_{k,j}
=
\alpha_{k,j}M_jd_k,
\]

with

\[
\alpha_{k,j}
=
\frac{1}{\|M_jg_k\|_2}.
\]

During each dictionary \(D\)-update, \(X\) and \(\alpha\) are held fixed. The resulting dictionary forward operator is therefore linear in the base dictionary.

## Current validation status

Three major parts of the algorithm have now been independently validated:

1. fixed-dictionary coefficient recovery;
2. fixed-coefficient dictionary recovery;
3. controlled alternating coefficient/dictionary recovery.

The current dictionary-learning phase reproduces the Gaussian/ramp simulation and initialization workflow used in the reference MATLAB implementation.

The reproduced 20 dB Gaussian/ramp problem uses

\[
N=80,\qquad
M=41,\qquad
T=80,\qquad
K=2,\qquad
J=10,
\]

with rational scales

\[
\left\{
\frac15,\frac14,\frac13,\frac25,\frac12,
\frac35,\frac23,\frac34,\frac45,1
\right\}.
\]

The current non-temporal pipeline reproduces the flat-dictionary initialization, sparse-penalty schedule, relaxation parameters, adaptive ADMM penalties, and padded Fourier-domain coefficient update used by the MATLAB workflow.

The immediate next phase is temporally regularized dictionary learning initialized from this validated non-temporal MCDL prefit.

## Validated coefficient-recovery components

### Multiscale convolution

Validated functionality includes:

- multiscale resampling and convolution;
- structured coefficient convention
  \[
  X\in\mathbb R^{L\times K\times J\times T};
  \]
- forward and adjoint operators;
- zero-padded convolution consistent with the MATLAB formulation;
- numerical composite-operator adjoint tests.

### Sparse coefficient recovery

The coefficient solver includes:

- nonnegative coefficient constraints;
- logarithmic sparsity penalty;
- continuation from an approximately \(\ell_1\) penalty to the nonconvex log penalty;
- ADMM splitting;
- nonnegative L-BFGS-B coefficient updates;
- adaptive ADMM penalty control.

The Banco-compatible dictionary-learning path additionally implements the reference MATLAB sparse schedule:

- 20 initial iterations of nonnegative \(\ell_1\) shrinkage;
- subsequent logarithmic shrinkage;
- fixed log parameter
  \[
  a=\frac{0.95}{\lambda_S}.
  \]

### Temporal regularization

The validated temporal model includes:

- position barycentric motion;
- log-scale barycentric motion;
- smooth squared activity gating satisfying
  \[
  g(0)=g'(0)=0;
  \]
- gated within-frame variance regularization;
- explicit analytical gradients;
- directional finite-difference gradient validation near machine precision;
- stationary, moving, birth, and death geometry tests.

### Nonconvex continuation

The coefficient solver supports full-state continuation of

\[
(X,Y,U,\rho,a)
\]

between nearby parameter values.

A compatibility test verifies that the stateful solver reproduces the original cold-start ADMM solution when no warm start is supplied.

### Automatic parameter selection

The project includes a hierarchical adaptive selector for

\[
(\lambda_S,\lambda_T,\beta_V).
\]

Selection uses observable quantities only:

1. expected-noise discrepancy;
2. residual-regularizer tradeoff geometry;
3. solution stability over neighboring parameter values;
4. connected stable parameter regions.

Two unsupervised strategies have been investigated:

- discrepancy plus stability-region selection;
- nested L-curve selection.

Synthetic truth is revealed only after parameter selection. A searched synthetic oracle is retained as a post-hoc benchmark for separating selector failure from candidate-family or identifiability failure.

## Noise convention

Monte Carlo noise is generated as

\[
N_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2).
\]

The prescribed variance is chosen from the requested SNR using expected noise energy. Individual realizations are not rescaled to have exactly the requested SNR.

Parameter selection uses

\[
D_{\mathrm{exp}}
=
\frac12N_{\mathrm{obs}}\sigma^2
\]

and discrepancy ratio

\[
r_D
=
\frac{
\frac12\|\mathcal A_DX-S\|_F^2
}{
D_{\mathrm{exp}}
}.
\]

Realized noise energy is retained only for post-hoc diagnostics and is not available to the selector.

## Development-scale Monte Carlo study

The adaptive coefficient-recovery selector was evaluated on

\[
5\text{ SNR levels}\times10\text{ independent realizations}
=
50\text{ datasets},
\]

using

\[
\mathrm{SNR}\in\{20,17,14,11,8\}\ {\rm dB}.
\]

Regularization parameters were selected independently for every realization. Synthetic truth was used only after parameter selection.

Under nested L-curve selection, exact position-scale trajectory recovery was:

| SNR (dB) | Exact trajectory rate |
|---:|---:|
| 20 | 100% |
| 17 | 90% |
| 14 | 70% |
| 11 | 20% |
| 8 | 10% |

The same selector recovered exactly eight significant coefficients in all 50 trials.

Position recovery was substantially more robust than scale recovery. Performance degradation below approximately 14 dB was dominated by ambiguity among neighboring scale atoms rather than loss of sparse temporal support.

The searched synthetic oracle achieved exact trajectory rates

\[
[100,\;90,\;70,\;20,\;20]\%.
\]

Thus, through 11 dB, the adaptive L-curve selector achieved the same exact-trajectory success rate as the searched oracle candidate family in this development study.

### Temporal-versus-nontemporal ablation

A paired comparison used exactly the same 50 noise realizations.

The non-temporal baseline solved

\[
\min_{X\ge0}
\frac12\|\mathcal A_DX-S\|_F^2
+
\lambda_SR_{\log}(X),
\]

with \(\lambda_S\) selected independently for every dataset.

The temporal estimator additionally used

\[
\lambda_T
\left[
R_{\mathrm{motion}}(X)
+
\beta_VR_{\mathrm{spread}}(X)
\right].
\]

Under L-curve selection, the non-temporal exact-eight-support rates were

\[
[90,\;70,\;70,\;30,\;50]\%,
\]

whereas the temporal estimator achieved

\[
100\%
\]

exact-eight-support recovery at every tested SNR.

Exact trajectory rates changed from

\[
[90,\;80,\;60,\;10,\;10]\%
\]

without temporal regularization to

\[
[100,\;90,\;70,\;20,\;10]\%
\]

with temporal regularization.

Median coefficient errors changed from

\[
[0.0272,\;0.0317,\;0.1612,\;0.5445,\;0.7080]
\]

to

\[
[0.0163,\;0.0221,\;0.0338,\;0.5049,\;0.7081].
\]

The most consistent benefit of temporal regularization is therefore structural: suppression of extraneous coefficients and stabilization of the intended one-coefficient-per-active-frame representation. At moderate SNR this is accompanied by improved trajectory recovery and substantially lower coefficient error.

### Oracle model comparison

The best searched non-temporal and temporal candidates were also compared post hoc.

Median non-temporal oracle coefficient errors were

\[
[0.01261,\;0.01915,\;0.03263,\;0.49351,\;0.55932],
\]

while median temporal-oracle errors were

\[
[0.00982,\;0.01396,\;0.02526,\;0.49326,\;0.49942].
\]

The corresponding exact trajectory rates were identical:

\[
[100,\;90,\;70,\;20,\;20]\%.
\]

The oracle comparison indicates that temporal regularization can improve coefficient accuracy even when both model families recover the same discrete trajectory.

The temporal search includes \(\lambda_T=0\), allowing it to revert naturally to the non-temporal solution when temporal regularization provides no benefit.

### Interpretation of the coefficient-recovery milestone

The development-scale evidence supports three conclusions:

1. **Sparse temporal structure is robust.**  
   The barycentric motion, spread, and activity-gating model reliably produces compact finite-duration coefficient tracks.

2. **Temporal regularization adds measurable structural value.**  
   Its most consistent benefit is elimination of extraneous coefficients and stabilization of the intended one-coefficient-per-frame representation.

3. **Low-SNR failure is primarily a scale-identification problem.**  
   Position and activity remain substantially more robust than discrete scale recovery as SNR decreases.

The current 10-trial-per-SNR experiment remains a development study. The original MATLAB study used 100 noise realizations per SNR, so publication-scale experiments may eventually increase the number of Monte Carlo realizations after the complete dictionary-learning pipeline is available.

## Dictionary learning

The dictionary update uses fixed coefficients \(X\) and fixed scale factors \(\alpha\) during each \(D\)-subproblem.

Define

\[
(\mathcal B_{X,\alpha}D)_t
=
\sum_{k,j}
(\alpha_{k,j}M_jd_k)*x_{k,j,t}.
\]

The ADMM \(D\)-subproblem is

\[
D^{(\ell+1)}
=
\arg\min_D
\left\{
\frac12
\|\mathcal B_{X,\alpha}D-S\|_F^2
+
\frac{\rho_D}{2}
\|D-G^{(\ell)}+H^{(\ell)}\|_F^2
\right\}.
\]

Its normal equation is

\[
\left(
\mathcal B_{X,\alpha}^{*}\mathcal B_{X,\alpha}
+
\rho_DI
\right)D
=
\mathcal B_{X,\alpha}^{*}S
+
\rho_D(G-H).
\]

The projected dictionary variable is updated according to the MATLAB nonnegative-dictionary constraint:

\[
G_k
=
\frac{[D_{r,k}+H_k]_+}
{\|[D_{r,k}+H_k]_+\|_2}
\]

for each nonzero atom, followed by recomputation of the scale-normalization factors

\[
\alpha_{k,j}
=
\frac{1}{\|M_jg_k\|_2}.
\]

The dual variable then updates as

\[
H^{(\ell+1)}
=
H^{(\ell)}
+
D_r^{(\ell+1)}
-
G^{(\ell+1)}.
\]

Thus the MATLAB-compatible update order is

\[
D
\longrightarrow
G
\longrightarrow
\alpha
\longrightarrow
H.
\]

## Validated dictionary-learning components

The fixed-\(X\) dictionary-update path has been validated incrementally.

### Dictionary forward operator and adjoint

The fixed-\(X\), fixed-\(\alpha\) forward operator satisfies linearity to relative error

\[
2.81\times10^{-16}.
\]

Its numerical adjoint test gives relative error

\[
1.57\times10^{-14}.
\]

No changes to the previously validated coefficient-side `MultiscaleConvolutionOperator` were required.

### Dictionary objective gradient

For

\[
f(D)
=
\frac12
\|\mathcal B_{X,\alpha}D-S\|_F^2,
\]

the analytical gradient

\[
\nabla_Df
=
\mathcal B_{X,\alpha}^{*}
(\mathcal B_{X,\alpha}D-S)
\]

was verified by centered finite differences. The best relative directional-derivative error was approximately

\[
1.08\times10^{-13}.
\]

### Normal operator and CG solve

The normal operator

\[
\mathcal N(D)
=
\mathcal B_{X,\alpha}^{*}
\mathcal B_{X,\alpha}D
+
\rho_DD
\]

was verified to be symmetric and positive definite for \(\rho_D>0\).

The matrix-free CG solution agreed with an explicitly assembled direct solve to relative error

\[
5.87\times10^{-13},
\]

with relative normal-equation residual

\[
1.86\times10^{-12}.
\]

### Dictionary projection

The MATLAB-compatible nonnegative projection has been implemented and tested.

For nonzero atoms it enforces:

- prescribed finite base support;
- nonnegativity;
- unit \(\ell_2\) norm.

Zero projected atoms are handled explicitly without generating NaNs.

### Scale normalization

After the \(G\)-projection,

\[
n_{k,j}
=
\|M_jg_k\|_2
\]

and

\[
\alpha_{k,j}
=
\frac1{n_{k,j}}
\]

are recomputed.

Tests verify

\[
\|\alpha_{k,j}M_jg_k\|_2=1
\]

to better than \(10^{-12}\).

Base-atom normalization and scale-dependent resampled-atom normalization are therefore explicitly treated as separate operations.

## Known-\(X\) dictionary recovery

### One-atom recovery

A noiseless one-atom experiment was performed with exact known coefficients and three scales.

Starting from an unrelated feasible dictionary atom:

- cosine similarity after the first iteration was \(0.999876\);
- cosine similarity was effectively \(1.00000000\) by iteration 10;
- final cosine similarity was \(1.0000000000\);
- final relative reconstruction residual was
  \[
  5.78\times10^{-7}.
  \]

### Two-atom recovery

A second noiseless experiment used two distinct nonnegative atoms with exact known coefficients independently exciting both atoms across three scales and 12 observations.

The true atoms had mutual cosine similarity approximately \(0.607\), providing a nontrivial atom-separation test.

After 150 dictionary iterations:

- final relative reconstruction residual was
  \[
  5.83\times10^{-6};
  \]
- both correct-channel cosine similarities were effectively 1;
- the final similarity matrix was approximately

\[
\begin{bmatrix}
1.000000 & 0.606691\\
0.606690 & 1.000000
\end{bmatrix}.
\]

These experiments establish accurate one- and two-atom dictionary recovery when coefficient information is exact.

## Banco Gaussian/ramp alternating-recovery milestone

The Python implementation now reproduces the principal components of the MATLAB Gaussian/ramp MCDL initialization workflow.

The exact simulation uses a unit-normalized Gaussian and 30-sample normalized ramp over 80 frames. Both atoms are active once per frame, giving

\[
\|X_{\mathrm{true}}\|_0=160.
\]

The reproduced initialization includes:

- zero coefficient initialization;
- centered flat dictionary atoms of widths 21 and 41;
- the exact 10-scale rational grid;
- 20 initial iterations of nonnegative \(\ell_1\) shrinkage followed by logarithmic shrinkage;
- fixed log parameter
  \[
  a=0.95/\lambda_S;
  \]
- coefficient and dictionary over-relaxation parameters of 1.8;
- initial ADMM penalties
  \[
  \rho_X=1000,\qquad\rho_D=500;
  \]
- standard-residual adaptive penalty balancing at each main iteration;
- padded Fourier-domain rank-one coefficient solves corresponding to the MATLAB `solvedbi_sm` pathway.

The multirate resampling implementation was compared directly with the MATLAB `lowpassM`, `uSampleCenter`, `dSampleCenter`, `reSampleCustomArrayCenter3`, and transpose formulations. The triangular interpolation kernel, 2-norm normalization, centered rational resampling, shifts, and scale-normalization conventions agree with the reference formulation.

### Diagnostic \(\lambda_S\) sweep at 20 dB

A focused oracle-development sweep used

\[
\lambda_S
\in
\{0.15,0.20,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.30,0.35,0.40\}.
\]

The best local ramp recovery occurred near

\[
\lambda_S=0.24.
\]

This value is an **oracle development choice**, selected using known dictionary truth. It is not yet an objective or deployable parameter-selection result.

For the 100-iteration flat-box prefit at

\[
\lambda_S=0.24,
\]

the final results were:

- clean reconstruction error: \(0.18417\);
- Gaussian cosine similarity: \(0.99035\);
- ramp cosine similarity: \(0.88409\);
- significant coefficients: 188.

The true coefficient tensor contains 160 nonzeros.

### Continued non-temporal MCDL

The learned dictionary and coefficient variable were then used to initialize a continued non-temporal MCDL solve. The sparse penalty remained in the logarithmic phase rather than restarting the initial \(\ell_1\) schedule.

After 200 additional iterations:

- clean reconstruction error was
  \[
  0.16709;
  \]
- Gaussian cosine similarity was
  \[
  0.99243;
  \]
- ramp cosine similarity was
  \[
  0.88441;
  \]
- the number of significant coefficients was
  \[
  158,
  \]
  compared with 160 in truth.

The second non-temporal stage therefore improves reconstruction, Gaussian recovery, and support recovery, but does not materially sharpen the ramp beyond the prefit result.

The final coefficient support is very close to truth, and the dictionary updates are small by the end of the continued solve. This provides a stable non-temporal learned-dictionary reference before temporal regularization is introduced.

## Current validation boundary

The following are now considered validated:

### Coefficient side

- multiscale convolution and adjoints;
- nonnegative log-sparse recovery;
- barycentric temporal motion;
- finite-duration activity gating;
- within-frame spread regularization;
- analytical temporal gradients;
- full-state nonconvex continuation;
- expected-noise adaptive parameter selection;
- multi-SNR Monte Carlo characterization;
- temporal-versus-nontemporal paired ablation;
- oracle model comparison.

### Dictionary side

- fixed-\(X\), fixed-\(\alpha\) dictionary forward operator;
- dictionary adjoint;
- dictionary objective gradient;
- SPD normal operator;
- matrix-free CG \(D\)-solve;
- nonnegative finite-support unit-norm projection;
- explicit scale-normalization update;
- noiseless known-\(X\) one-atom recovery;
- noiseless known-\(X\) two-atom recovery;
- exact Gaussian/ramp simulation geometry;
- MATLAB-compatible flat initialization;
- Banco-compatible sparse-penalty schedule;
- adaptive coefficient and dictionary ADMM penalties;
- padded Fourier-domain coefficient update;
- controlled joint Gaussian/ramp coefficient/dictionary recovery at 20 dB.

Controlled joint coefficient/dictionary recovery is therefore now validated for the reproduced 20 dB Gaussian/ramp problem.

Validation of the complete temporally regularized learned-dictionary pipeline remains in progress.

## Next development milestone

The next phase is temporally regularized joint dictionary learning initialized from the validated Banco-style non-temporal prefit.

The planned progression is:

1. preserve the validated 20 dB Gaussian/ramp non-temporal result as a regression benchmark;
2. initialize the softmin temporal-regularized solve from the learned dictionary and coefficient representation;
3. determine an objective selection procedure for \(\lambda_S\) and the temporal regularization parameter without using synthetic truth;
4. compare temporal and non-temporal learned dictionaries on identical noise realizations;
5. extend the validated workflow across SNR and independent noise realizations;
6. proceed to publication-scale Monte Carlo experiments only after the complete learned-dictionary pipeline is stable.

The current

\[
\lambda_S=0.24
\]

value is retained only as a 20 dB oracle-development reference and must not be presented as an unsupervised selected parameter.

Previously validated coefficient and dictionary components should remain unchanged unless an integration test identifies a specific inconsistency.

## Project layout

### Root modules

The project root contains the active numerical implementation:

- `multiscale_convolution.py`: multiscale coefficient-side forward and adjoint operator
- `zero_padded_convolution.py`: zero-padded convolution engine
- `multiscale_operator.py`: multiscale resampling and its adjoint
- `x_update.py`: smooth nonnegative coefficient update
- `log_admm.py`: original log-penalty ADMM solver
- `log_admm_stateful_compat.py`: full-state continuation-compatible solver
- `barycentric_transition.py`: motion, activity-gate, and spread regularization
- `local_graph.py`: shift-scale geometry
- `local_transition.py`: historical transition geometry still used by validation code
- `gaussian_experiment.py`: synthetic problem and variance-controlled noise utilities
- `dictionary_forward_operator.py`: fixed-\(X\), fixed-\(\alpha\) dictionary forward, adjoint, and normal operators
- `dictionary_cg.py`: matrix-free CG dictionary \(D\)-solve
- `dictionary_projection.py`: nonnegative unit-norm base-dictionary projection
- `dictionary_alpha.py`: scale-dependent normalization-factor update
- `dictionary_initialization.py`: MATLAB-compatible flat-dictionary initialization
- `banco_gaussian_ramp.py`: reference Gaussian/ramp geometry and 10-scale grid
- `banco_sparse_penalty.py`: MATLAB-compatible \(\ell_1\)-to-log sparse schedule
- `banco_rho_update.py`: reference residual-balancing ADMM penalty update
- `banco_solvedbi_sm.py`: rank-one Sherman-Morrison Fourier solve
- `banco_coefficient_fourier.py`: padded Fourier-domain coefficient update
- `banco_coefficient_step.py`: Banco-compatible coefficient ADMM step
- `coefficient_quadratic_cg.py`: independently validated physical-domain quadratic coefficient solve used for comparison

### Tests

`tests/` contains numerical validation for:

- multiscale operators and adjoints;
- zero-padded convolution conventions;
- coefficient-recovery behavior;
- temporal gradients;
- ADMM compatibility;
- dictionary forward linearity;
- dictionary adjoint consistency;
- dictionary objective gradients;
- dictionary normal operator and CG solve;
- dictionary projection;
- scale-normalization updates;
- Banco flat initialization;
- Banco Gaussian/ramp simulation geometry;
- Banco sparse-penalty switching;
- Banco adaptive-\(\rho\) update;
- rank-one Fourier Sherman-Morrison solve;
- padded Fourier coefficient update;
- Banco coefficient-step composition;
- physical-domain quadratic coefficient CG;
- historical softmin, Sinkhorn, and transition formulations.

Some historical tests correspond to archived experimental implementations and are not part of the current core regression suite.

### Active experiments

`experiments/barycentric/` contains finite-track and continuation experiments.

`experiments/parameter_selection/` contains:

- regularization grids;
- discrepancy and L-curve analyses;
- adaptive hierarchical selection;
- Monte Carlo wrappers;
- paired temporal-versus-nontemporal comparisons.

`experiments/diagnostics/` contains scale-confusion and comparative diagnostics.

`experiments/dictionary_learning/` contains known-\(X\), alternating, and Banco-reproduction experiments, including:

- `known_x_one_atom`
- `known_x_two_atom`
- `banco_exact_prefit_20db`
- `banco_lambda_s_sweep`
- `banco_prefit_then_mcdl_20db`

The known-\(X\) experiments remain regression-quality development benchmarks. Banco-style alternating experiments are maintained separately so that integration work does not alter the validated component tests.

### Archived approaches

`archive/sinkhorn/` preserves the local optimal-transport implementation and benchmarks.

`archive/gated_markov/` preserves the Markov/Hellinger transition experiments.

These approaches are retained because their negative or intermediate results motivate the final barycentric temporal model.

### Documentation

`docs/barycentric_temporal_regularization.qmd` is the living mathematical and numerical specification.

It contains:

- model derivations;
- analytical gradient formulas;
- optimization details;
- parameter-selection methodology;
- Monte Carlo results;
- temporal-versus-nontemporal comparisons;
- dictionary ADMM formulation;
- dictionary operator validation;
- known-\(X\) dictionary-recovery results;
- the current validation boundary and next planned experiments.

## Running the current workflow

### Environment

Create and activate a machine-specific Python virtual environment. Virtual environments should not be copied between Windows and Linux systems.

Install the repository dependencies from:

`requirements.txt`

Development testing additionally requires `pytest`.

### Regression testing

Tests should be run from the repository root using:

`python -m pytest`

The current core validation can also be run selectively when working on a particular component.

### Coefficient experiments

The coefficient-recovery workflow consists of:

1. numerical operator and gradient validation;
2. single-realization adaptive parameter selection;
3. inspection of selected-parameter summaries;
4. Monte Carlo evaluation;
5. temporal-versus-nontemporal comparison;
6. Quarto report rendering.

Long Monte Carlo studies save per-trial summaries and can skip completed trials when restarted.

### Dictionary experiments

Known-\(X\) dictionary experiments are run from the repository root as Python modules.

Regression benchmarks include:

- `experiments.dictionary_learning.known_x_one_atom`
- `experiments.dictionary_learning.known_x_two_atom`

Banco Gaussian/ramp experiments include:

- `experiments.dictionary_learning.banco_exact_prefit_20db`
- `experiments.dictionary_learning.banco_lambda_s_sweep`
- `experiments.dictionary_learning.banco_prefit_then_mcdl_20db`

The current 20 dB oracle-development reference uses

\[
\lambda_S=0.24.
\]

Generated numerical results should be stored under `results/` rather than committed as source files.

## Development policy

A numerical component is considered validated only after its relevant mathematical identity, adjoint, or gradient passes numerical testing and its behavior is confirmed in a controlled synthetic experiment.

Validated components should not be casually refactored while introducing new functionality. New integration logic should be developed in parallel files or experiments whenever practical.

Validated milestones should be preserved through:

- focused Git commits;
- README updates when the validated repository state or workflow changes;
- Quarto updates when mathematics, methodology, or validated scientific results change materially.

Experimental code and negative results should be archived rather than silently deleted.

Git remains the source of truth when moving development between the local Windows environment and the Tufts Linux cluster. Machine-specific virtual environments are not part of the repository.

## Reference implementation

The original MATLAB implementation is available in the `dbanco/CHESS-Research` repository under `MATLAB/MultiscaleDL`, with the current dictionary-learning reference code examined under `MCDL-OF`.

The Python project is designed to provide:

- explicit mathematical structure;
- independently validated operators and gradients;
- reproducible parameter selection;
- restartable Monte Carlo experiments;
- controlled validation of dictionary learning;
- a clear progression from fixed-dictionary coefficient recovery to full alternating multiscale dictionary learning.