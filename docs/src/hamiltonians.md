# Module `Hamiltonians`

This module contains definitions of Hamiltonians, in particular specific
physical models of interest. These are organised by means of an interface
around the abstract type [`AbstractHamiltonian`](@ref), in the spirit of the
`AbstractArray` interface as discussed in the [Julia Documentation](https://docs.julialang.org/en/v1/manual/interfaces/).

The Hamiltonians can be used for projector quantum Monte Carlo with [`ProjectorMonteCarloProblem`](@ref) or for exact diagonalization with [`ExactDiagonalizationProblem`](@ref), see [Exact Diagonalization](@ref).

```@docs
Hamiltonians
```

Here is a list of fully implemented model Hamiltonians. There are several variants
of the Hubbard model in real and momentum space, as well as some other models.

## Real space Hubbard models
```@docs
HubbardReal1D
HubbardReal1DEP
HubbardRealSpace
ExtendedHubbardReal1D
```

## Momentum space Hubbard models
```@docs
HubbardMom1D
HubbardMom1DEP
ExtendedHubbardMom1D
```

## Harmonic oscillator models
```@docs
HOCartesianContactInteractions
HOCartesianEnergyConservedPerDim
HOCartesianCentralImpurity
```

## Other model Hamiltonians
```@docs
MatrixHamiltonian
Transcorrelated1D
FroehlichPolaron
```

## Convenience functions
```@docs
rayleigh_quotient
momentum
hubbard_dispersion
continuum_dispersion
shift_lattice
shift_lattice_inv
```

## Hamiltonian wrappers
The following Hamiltonians are constructed from an existing
Hamiltonian instance and change its behaviour:
```@docs
GutzwillerSampling
GuidingVectorSampling
ParitySymmetry
TimeReversalSymmetry
Stoquastic
```

## Observables
`Rimu.jl` offers two other supertypes for operators that are less
restrictive than [`AbstractHamiltonian`](@ref).
[`AbstractObservable`](@ref) and [`AbstractOperator`](@ref)s both
can represent a physical observable. Their expectation values can be sampled during a [`ProjectorMonteCarloProblem`](@ref) simulation by
passing them into a suitable [`ReplicaStrategy`](@ref), e.g.
[`AllOverlaps`](@ref). Some observables are also [`AbstractHamiltonian`](@ref)s. The full type hierarchy is
```julia
AbstractHamiltonian{T} <: AbstractOperator{T} <: AbstractObservable{T}
```

```@docs
ParticleNumberOperator
G2RealCorrelator
G2RealSpace
G2MomCorrelator
SuperfluidCorrelator
StringCorrelator
DensityMatrixDiagonal
SingleParticleExcitation
TwoParticleExcitation
ReducedDensityMatrix
Momentum
AxialAngularMomentumHO
```

## Geometry

Lattices in higher dimensions are defined here and can be passed with the keyword argument
`geometry` to [`HubbardRealSpace`](@ref) and [`G2RealSpace`](@ref).

```@docs
Hamiltonians.Geometry
CubicGrid
PeriodicBoundaries
HardwallBoundaries
LadderBoundaries
HoneycombLattice
HexagonalLattice
Hamiltonians.Directions
Hamiltonians.Displacements
Hamiltonians.neighbor_site
Hamiltonians.periodic_dimensions
Hamiltonians.num_dimensions
```

## Index
```@index
Pages   = ["hamiltonians.md"]
```
