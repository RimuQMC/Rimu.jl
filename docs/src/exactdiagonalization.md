# Exact Diagonalization

The main functionality of Rimu for exact diagonalization is contained in the module `ExactDiagonalization`.

```@docs
ExactDiagonalization
```

## `ExactDiagonalizationProblem`

```@docs
ExactDiagonalizationProblem
solve(::ExactDiagonalizationProblem)
init(::ExactDiagonalizationProblem)
estimate_memory_requirement
```

## Solver algorithms

```@docs
KrylovKitSolver
LinearAlgebraSolver
ArpackSolver
LOBPCGSolver
```

## Converting a Hamiltonian in to a matrix

```@docs
BasisSetRepresentation
build_basis
Matrix
sparse
LinearMap
Rimu.ExactDiagonalization.OperatorAsMap
```

## Deprecated
```@docs
BasisSetRep
```
