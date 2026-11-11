# MCDL-Python

Python implementation of multiscale convolutional dictionary learning
with softmin trajectory regularization.

This project is a Python reimplementation of MATLAB research code from
the CHESS-Research MultiscaleDL project.

## Current status

The project is under active development.

Implemented:

- Softmin scale-position trajectory regularizer
- Analytical softmin gradient
- Finite-difference gradient validation
- Stationary trajectory test
- Moving trajectory test
- Scale-position diagonal trajectory test
- L-BFGS coefficient-map update
- Softmin regularization parameter sweep

Current numerical validation:

- Softmin directional-derivative relative error: approximately 1e-12
- Full X-objective directional-derivative relative error: approximately 1e-12

## Softmin regularizer

For consecutive coefficient maps, the regularizer compares nine local
transitions in scale-position space:

    (-1,-1) (-1,0) (-1,+1)
    ( 0,-1) ( 0,0) ( 0,+1)
    (+1,-1) (+1,0) (+1,+1)

For transition residuals r_s,

    R(X) = -tau log sum_s exp(-r_s / tau)

so that, as tau approaches zero,

    R(X) -> min_s r_s.

The current implementation has been validated using analytical versus
finite-difference directional derivatives.

## Development

Create a virtual environment:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

Run the validation tests:

    python .\test_softmin.py
    python .\test_x_update.py

## Project structure

    softmin.py
        Reference implementation of the softmin regularizer.

    x_update.py
        L-BFGS implementation of the softmin-regularized X update.

    test_softmin.py
        Numerical and behavioral validation of softmin.

    test_x_update.py
        Gradient, optimization, and parameter-sweep experiments.

## Roadmap

1. Multiscale resampling operators
2. Convolutional forward operator
3. Adjoint validation
4. Sparse ADMM splitting
5. Dictionary update
6. Reproduction of the MATLAB single-Gaussian experiments

## Reference implementation

The original MATLAB implementation is available in the
dbanco/CHESS-Research repository under MATLAB/MultiscaleDL.

The Python code in this repository is intended to be independently
testable rather than a literal line-by-line MATLAB translation.