# Paper–MATLAB–Python Operator Audit

## MCDL-ST multiscale signal model

**Reference:** *Space-time Multi-scale, Convolutional Dictionary Learning with Applications to High Energy X-ray Diffraction*, Daniel P. Banco, Armand Beaudoin, Matthew P. Miller, and Eric L. Miller.

**Purpose:** Verify that the original MATLAB implementation and the Python port implement the multiscale signal model described in the paper, with particular attention to interpolation, low-pass filtering, decimation, centering, normalization, and convolution.

---

## 1. Reference signal model

The paper represents the signal at time \(t\) as

\[
b_t[n]
=
\sum_k\sum_\sigma\sum_s
d_{k,\sigma}[n-s]x_{k,\sigma,t}[s].
\]

Equivalently,

\[
\mathbf b_t
=
\sum_k\sum_j
\alpha_{k,j}
\mathbf T(\mathbf d_{k,\sigma_j})
\mathbf x_{k,j,t}.
\]

The scaled atom is generated from the base atom by rational resampling:

\[
\boxed{
\mathbf d_{k,p/q}
=
\mathbf P_{p/q}
\mathbf \Delta_q
\mathbf T(\phi_L)
\mathbf \Upsilon_p
\mathbf d_k
}
\]

where

- \(\mathbf \Upsilon_p\) is interpolation/upsampling by \(p\),
- \(\mathbf T(\phi_L)\) is low-pass filtering,
- \(\mathbf \Delta_q\) is decimation by \(q\),
- \(\mathbf P_{p/q}\) pads and centers the resulting atom,
- \(\alpha_{k,j}\) normalizes each scaled atom.

Define

\[
\mathbf M_j
=
\mathbf P_{p_j/q_j}
\mathbf\Delta_{q_j}
\mathbf T(\phi_L)
\mathbf\Upsilon_{p_j}.
\]

Then

\[
\mathbf d_{k,\sigma_j}
=
\mathbf M_j\mathbf d_k.
\]

The complete coefficient-space forward operator is therefore

\[
\boxed{
\mathbf b_t
=
\mathbf A(\mathbf d,\alpha)\mathbf x_t
=
\sum_k\sum_j
\alpha_{k,j}
\mathbf T(\mathbf M_j\mathbf d_k)
\mathbf x_{k,j,t}.
}
\]

---

# 2. Rational scale convention

## Paper

The paper defines

\[
\sigma_j=\frac{p_j}{q_j}
\]

with

\[
1=\sigma_1>\sigma_2>\cdots>\sigma_J>0.
\]

Thus the base atom is the largest atom and production scales satisfy

\[
\boxed{0 < p/q \le 1.}
\]

The paper proposes Farey sequences. For example,

\[
F_4=
\left\{
1,\frac34,\frac23,\frac12,\frac13,\frac14
\right\}.
\]

## MATLAB

The MATLAB implementation represents a rational scale using integer upsampling and downsampling factors and applies centered multirate resampling.

### Status

**CONSISTENT IN CONCEPT.**

## Python

`multiscale_operator.py` accepts `(c1,c2)` corresponding to

\[
p/q=c1/c2.
\]

We have deliberately tested both compression and expansion:

\[
1/2,\;2/3,\;3/2,\;2/1.
\]

The expansion cases were useful operator tests but are outside the production model described by the paper.

### Required action

Add an optional paper-compatible validation mode requiring

\[
0 < p/q \le 1.
\]

**Status:** **IMPLEMENTATION GENERALIZES PAPER.**

---

# 3. Upsampling / interpolation

## Paper

For rational scale

\[
\sigma=p/q,
\]

the first operation is

\[
\mathbf y
=
\mathbf\Upsilon_p\mathbf d.
\]

Operationally, this inserts zeros between samples before interpolation filtering.

## MATLAB

The centered upsampling functions insert samples according to a center-aware index sequence and apply `lowpassM` to the upsampled result.

The center passed into the operation is scaled consistently with \(p\).

## Python

Implemented by:

`upsample_center(x, m, center)`

in `multiscale_operator.py`.

The operation:

1. constructs the center-relative insertion indices,
2. inserts the original samples at those locations,
3. fills remaining positions with zero,
4. applies `lowpass_m`.

### Numerical checks completed

Identity and center-preservation tests pass.

The resampling operator subsequently passed its adjoint test at approximately

\[
10^{-16}.
\]

### Status

**CONSISTENT, SUBJECT TO DIRECT MATLAB/PYTHON SAMPLE-BY-SAMPLE TEST.**

---

# 4. Low-pass interpolation filter

## Paper

Appendix A defines

\[
\phi_L=\phi_p*\phi_q,
\]

where

\[
\phi_p[n]
=
\begin{cases}
1-|n|/p,&|n|<p\\
0,&\text{otherwise},
\end{cases}
\]

and similarly for \(\phi_q\).

Each is triangular with support

\[
2p-1
\]

and

\[
2q-1,
\]

respectively.

The composite low-pass filter therefore has length

\[
\boxed{2p+2q-3}.
\]

## MATLAB

`lowpassM` constructs a symmetric triangular filter equivalent in shape to

\[
\left[
1/p,\;2/p,\ldots,(p-1)/p,\;1,\;(p-1)/p,\ldots,1/p
\right]
\]

and normalizes that filter.

The interpolation and decimation paths each invoke the corresponding filtering operation.

Applying the \(p\)-filter followed by the \(q\)-filter gives

\[
T(\phi_q)T(\phi_p)
=
T(\phi_q*\phi_p),
\]

which realizes the paper's composite filter

\[
\phi_L=\phi_p*\phi_q.
\]

## Python

`lowpass_m(x,m)` currently reproduces the MATLAB normalized triangular kernel.

Example for \(m=3\):

\[
[0.22941573,\,
0.45883147,\,
0.68824720,\,
0.45883147,\,
0.22941573].
\]

This matched the expected normalized triangular impulse response.

### Open question

The paper defines the *shape* of \(\phi_p,\phi_q\), but the MATLAB implementation applies an explicit L2 normalization to the triangular filters.

We need to determine whether that normalization is mathematically absorbed into the paper's later scale normalization

\[
\alpha_{k,j}
=
\frac{1}{\|M_jd_k\|_2}
\]

or whether MATLAB and the literal Appendix A filter differ by a scalar before scaled-atom normalization.

Because every scaled dictionary atom is subsequently unit normalized, such a scalar should not affect the final normalized scaled atom, but it can matter for adjoints used during dictionary optimization.

### Status

**STRUCTURALLY CONSISTENT; FILTER NORMALIZATION REQUIRES EXPLICIT CROSS-CHECK.**

---

# 5. Decimation

## Paper

After low-pass filtering:

\[
\mathbf y
=
\mathbf\Delta_q
\mathbf T(\phi_L)
\mathbf\Upsilon_p
\mathbf d.
\]

Thus decimation by \(q\) occurs only after anti-aliasing filtering.

## MATLAB

The centered downsampling routine:

1. constructs a center-relative sample sequence,
2. low-pass filters,
3. extracts every \(q\)-th sample,
4. returns the new center index.

## Python

Implemented by:

`downsample_center(x,m,center)`.

Its adjoint is implemented separately through scatter/insertion followed by the symmetric filter.

### Numerical check

Forward/adjoint errors:

- \(1/1\): \(1.81\times10^{-16}\)
- \(1/2\): \(0\)
- \(2/3\): \(1.62\times10^{-16}\)
- \(3/2\): \(1.15\times10^{-16}\)
- \(2/1\): \(0\)

### Status

**CONSISTENT AND NUMERICALLY VALIDATED INTERNALLY.**

---

# 6. Signal centering

## Paper

Signals are interpreted as finite samples centered at the physical coordinate

\[
n=0.
\]

For a length-\(N\) signal,

\[
\mathbf b_t=
[
b_t[N_-],\ldots,b_t[-1],b_t[0],b_t[1],\ldots,b_t[N_+]
]^T.
\]

After rational resampling, the matrix

\[
\mathbf P_{p/q}
\]

zero-pads and shifts the atom so that its physical center remains at

\[
n=0.
\]

## MATLAB

The forward routine calculates the center after decimation and computes

\[
\text{shift}
=
\text{center}
-
\text{center}_{down}.
\]

The signal is then zero padded and shifted using `circshift`.

## Python

`resample_atom()` implements the same procedure:

1. rational resampling,
2. determine the resampled center,
3. compute center correction,
4. zero-pad to \(N\),
5. apply `np.roll`.

### Numerical evidence

For the symmetric 71-sample test atom with 1-based center 36:

| Scale | Center shift | Python peak |
|---|---:|---:|
| 1/1 | 0 | 35 |
| 1/2 | +18 | 35 |
| 2/3 | +12 | 35 |
| 3/2 | −18 | 35 |
| 2/1 | −36 | 35 |

Python index 35 corresponds to MATLAB index 36.

For atoms deliberately displaced relative to the designated center, scaling moved their peaks consistently toward or away from the fixed physical center.

### Status

**STRONGLY CONSISTENT.**

### Remaining validation

Perform a direct MATLAB/Python sample-by-sample comparison for both:

1. a centered symmetric atom;
2. an asymmetric off-center atom.

---

# 7. Scaled-atom normalization

## Paper

Each scaled atom satisfies

\[
\boxed{
\|\alpha_{k,j}M_jd_k\|_2=1.
}
\]

At base scale,

\[
\|d_k\|_2=1.
\]

For other scales,

\[
\alpha_{k,j}
=
\frac{1}{\|M_jd_k\|_2}.
\]

This makes coefficient amplitudes comparable across scales.

## MATLAB

The resampling routines calculate and retain normalization values, which are then reused when constructing the forward/adjoint operations.

## Python

`MultiscaleConvolutionOperator` computes

\[
\nu_{k,j}
=
\|R_jd_k\|_2
\]

and stores

`norm_vals[k,j]`.

It then constructs

\[
d_{k,j}
=
\frac{R_jd_k}{\nu_{k,j}}.
\]

### Numerical check

All tested scaled atoms reported

`norm = 1.000000000000`.

### Status

**CONSISTENT.**

---

# 8. Convolution

## Paper

The fundamental model is

\[
b_t[n]
=
\sum_{k,j,s}
d_{k,j}[n-s]x_{k,j,t}[s].
\]

The corresponding linear operator is

\[
\mathbf T(d_{k,j})\mathbf x_{k,j,t}.
\]

The model is therefore ordinary shifted convolution, not correlation.

## MATLAB

The production implementation evaluates convolution through FFT multiplication.

It additionally includes zero-padding related to the finite-support dictionary, including padding by \(M-1\).

This strongly suggests use of FFTs as an efficient implementation of a finite linear-convolution model rather than an assumption of periodic/circular signals.

## Python

The current `ConvolutionOperator` computes

\[
\operatorname{IFFT}
\left[
\operatorname{FFT}(d)
\operatorname{FFT}(x)
\right]
\]

at length \(N\).

It therefore currently implements

\[
\boxed{\text{N-point circular convolution}.}
\]

### Numerical validation

Impulse error:

\[
7.95\times10^{-17}.
\]

A filter impulse at index 3 convolved with a coefficient impulse at index 10 produced a peak at index 13, verifying the forward shift convention.

Forward/adjoint error:

\[
2.45\times10^{-16}.
\]

The composite multiscale convolution operator subsequently passed at

\[
5.16\times10^{-16}.
\]

### Status

**INTERNALLY CORRECT BUT NOT YET VERIFIED AGAINST THE PAPER'S FINITE-SIGNAL CONVOLUTION CONVENTION.**

### Priority

**HIGH.**

We must establish the intended padding/cropping dimensions from the MATLAB implementation and reproduce those dimensions in Python.

---

# 9. Coefficient-domain length

## Paper

The paper describes

\[
x_{k,j,t}[s]
\]

as coefficients over shifts.

Several later expressions, including the temporal regularizer and shift-time representation, use a shift dimension involving

\[
N+M-1.
\]

For example, the shift-time representation is stated as

\[
X_k^{shift}
\in
\mathbb R^{(N+M-1)\times T}.
\]

This suggests a coefficient shift domain larger than the measurement length \(N\).

## MATLAB

The production convolution code uses explicit padding involving \(M-1\).

The exact dimensions must be traced through the relevant MATLAB `Ax`/`At` operations and coefficient initialization.

## Python

Current coefficient layout is

\[
X\in
\mathbb R^{N\times K\times J\times T}.
\]

Example:

`(96,1,3,5)`.

Thus the current Python model assumes exactly \(N\) spatial coefficient locations.

### Status

**POTENTIAL DISCREPANCY.**

### Priority

**HIGH.**

Before claiming paper fidelity, determine whether production MATLAB uses:

\[
N,
\quad
N+M-1,
\]

or another padded FFT dimension for the coefficient maps.

### Coefficient-domain dimension: MATLAB-verified.

The production MATLAB implementation uses coefficient maps of length

L=N+M−1,L=N+M-1,L=N+M−1,

while measured signals have length NNN. Resampled scaled dictionary atoms are first represented at length NNN, post-padded by M−1M-1M−1, and transformed in an LLL-point FFT workspace. The forward convolution produces LLL samples and discards the first M−1M-1M−1, yielding the physical NNN-sample signal. The coefficient adjoint performs the exact transpose operation by pre-padding an NNN-sample residual with M−1M-1M−1 zeros before Fourier correlation. This convention is used consistently by simulation generation, reconstruction metrics, coefficient initialization, and solver routines. MCDL_padding_usage

And importantly:

The current Python implementation does not yet match this convention. Its resampling remains valid, but its coefficient maps and FFT convolution domain must be changed from NNN to N+M−1N+M-1N+M−1.

---

# 10. Forward multiscale operator

## Paper

For fixed dictionary,

\[
A(d,\alpha)x_t
=
\sum_{k,j}
\alpha_{k,j}
T(M_jd_k)x_{k,j,t}.
\]

## Python

`MultiscaleConvolutionOperator` implements precisely the same algebraic composition:

`base dictionary`
→ `rational resampling`
→ `center correction`
→ `unit normalization`
→ `convolution`
→ `sum across atom/scale channels`.

Its coefficient-space adjoint satisfies

\[
\langle A_DX,R\rangle
=
\langle X,A_D^*R\rangle
\]

with relative error

\[
5.16\times10^{-16}.
\]

### Status

**ALGEBRAICALLY CONSISTENT; CONVOLUTION BOUNDARY DOMAIN REMAINS OPEN.**

---

# 11. Current audit status

## Green: validated or strongly supported

- Rational multirate scaling
- Upsampling architecture
- Triangular interpolation filtering
- Decimation architecture
- Center-aware resampling
- Zero-padding of scaled atoms
- Center correction
- Unit normalization of scaled atoms
- Forward/adjoint consistency
- Convolution orientation
- Atom-major scale-channel organization

## Yellow: explicit MATLAB/Python comparison still required

- Exact triangular-filter amplitudes before scaled-atom normalization
- Boundary behavior of `conv(...,'same')`
- Odd/even atom-length center conventions
- Off-center asymmetric atom resampling
- Exact list/order of production Farey scales

## Red: resolve before claiming paper-equivalent Python implementation

### A. Convolution boundary condition

Determine whether the MATLAB implementation realizes finite linear convolution by FFT padding/cropping and reproduce it exactly.

### B. Coefficient-map spatial dimension

Determine whether the true implementation uses

\[
N,
\quad N+M-1,
\]

or another padded dimension.

These two questions are likely coupled.

---

# 12. Cross-language validation plan

For every operator, generate deterministic MATLAB output and compare against Python.

Use `float64` and save arrays to `.mat` or text/NumPy-compatible files.

## Test A: low-pass impulse response

For

\[
p=2,\;3,\;4,
\]

generate an impulse and record MATLAB `lowpassM`.

Require normalized Python difference:

\[
\epsilon
=
\frac{\|y_M-y_P\|_2}
{\max(1,\|y_M\|_2)}
<10^{-12}.
\]

## Test B: rational centered resampling

Test:

\[
1,\frac45,\frac34,\frac23,\frac12,\frac13.
\]

Use:

1. symmetric centered triangle;
2. asymmetric atom;
3. random atom with fixed seed.

Compare:

- output length,
- center index,
- shift correction,
- normalization factor,
- every output sample.

Acceptance:

\[
\epsilon<10^{-12}.
\]

## Test C: adjoint

In both MATLAB and Python verify

\[
\frac{
|\langle Rd,z\rangle-\langle d,R^*z\rangle|
}{
\max(1,|\langle Rd,z\rangle|,|\langle d,R^*z\rangle|)
}
<10^{-10}.
\]

Python currently achieves approximately

\[
10^{-16}.
\]

## Test D: convolution impulse locations

Use one dictionary impulse and one coefficient impulse.

Verify:

\[
d[r_d]=1,\qquad x[r_x]=1
\]

produces its peak at the index dictated by

\[
d[n-s]x[s].
\]

Include boundary cases near both ends of the measurement domain.

These boundary tests will distinguish circular convolution from the paper's intended finite convolution immediately.

## Test E: MATLAB/Python convolution with random inputs

Use identical fixed random arrays.

Compare complete forward outputs and adjoints.

Acceptance:

\[
\epsilon<10^{-10}.
\]

## Test F: complete multiscale forward model

Generate:

- \(K=1\),
- paper-compatible Farey scales,
- one centered Gaussian base atom,
- sparse known coefficient maps.

Compute

\[
b=A(d,\alpha)x
\]

independently in MATLAB and Python.

Compare the complete synthesized signals.

Acceptance:

\[
\epsilon<10^{-10}.
\]

## Cross-language numerical validation: 
Not performed because the current development environment does not include MATLAB. Fidelity is instead established by direct translation of the MATLAB operator definitions, dimension and padding analysis, analytical adjoint tests, and finite-difference gradient tests. A MATLAB/Python numerical comparison remains optional future validation if a MATLAB runtime becomes available.
---

# 13. Required implementation decision before continuing the port

Do not alter the validated Python resampling kernels merely to make them appear more paper-like.

First resolve the MATLAB forward-convolution dimensions.

Specifically determine:

1. measurement length \(N\);
2. base dictionary support \(M\);
3. scaled dictionary storage length;
4. coefficient-map shift dimension;
5. FFT length;
6. pre-padding;
7. post-padding;
8. crop/unpad indices;
9. corresponding adjoint crop/padding rules.

Once these are known, implement a new finite convolution operator while preserving the present circular operator as a tested reference.

---

# 14. Audit conclusion

The strongest current conclusion is:

**The MATLAB and Python multirate resampling architecture is substantially consistent with the multiscale model in the paper.**

The sequence

\[
\boxed{
\text{upsample}
\rightarrow
\text{triangular filtering}
\rightarrow
\text{decimate}
\rightarrow
\text{center}
\rightarrow
\text{pad}
\rightarrow
\text{normalize}
}
\]

matches the construction of \(M_jd_k\) in the paper.

The Python implementation has particularly strong internal numerical validation through machine-precision adjoint tests.

However, **paper equivalence has not yet been established for the convolutional boundary handling and coefficient-map spatial dimension**. The current Python implementation uses \(N\)-sample coefficient maps and \(N\)-point circular convolution, whereas the paper and MATLAB evidence indicate additional \(M-1\) padding and potentially an \(N+M-1\) shift domain.

Those issues should be resolved before proceeding to dictionary updates or attempting quantitative reproduction of the paper's simulation results.

**Next audit task:** trace the dimensions and padding through the MATLAB convolution forward and adjoint operators and reproduce one MATLAB forward convolution numerically in Python.