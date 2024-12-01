# Custom Hamiltonians and observables

`Rimu` can be used to work with custom Hamiltonians and observables that are user-defined and
 not part of the `Rimu.jl` package. To make this possible and reliable, `Rimu` exposes a number 
 of interfaces and provides helper functions to test compliance with the interfaces through the submodule [`Rimu.InterfaceTests`](@ref), see [Interface tests](@ref).

 In order to define custom Hamiltonians or observables it is useful to know how the operator type hierarchy works in `Rimu`. For an example of how to code custom Hamiltonians that are not part of the `Rimu.jl` package, see [`RimuLegacyHamiltonians.jl`](https://github.com/RimuQMC/RimuLegacyHamiltonians.jl).

## Operator type hierarchy

`Rimu` offers a hierarchy of abstract types that define interfaces with different requirements
for operators:
```julia
AbstractHamiltonian <: AbstractOperator <: AbstractObservable
```
The different abstract types have different requirements and are meant to be used for different purposes. 
- [`AbstractHamiltonian`](@ref)s are fully featured models that define a Hilbert space and a linear operator over a scalar field. They can be passed as a Hamiltonian into [`ProjectorMonteCarloProblem`](@ref) or [`ExactDiagonalizationProblem`](@ref).
- [`AbstractOperator`](@ref) and [`AbstractObservable`](@ref) are supertypes of [`AbstractHamiltonian`](@ref) with less stringent conditions. They are useful for defining observables that can be used in a three-way `dot` product, or passed as observables into a [`ReplicaStrategy`](@ref) in a [`ProjectorMonteCarloProblem`](@ref).

## Hamiltonians interface

Behind the implementation of a particular model is a more abstract interface for defining
Hamiltonians. If you want to define a new model you should make use of this interface. A new 
model Hamiltonian should subtype to `AbstractHamiltonian` and implement the relevant methods.

```@docs
AbstractHamiltonian
offdiagonals
diagonal_element
starting_address
```

The following functions may be implemented instead of [`offdiagonals`](@ref).

```@docs
num_offdiagonals
get_offdiagonal
```

The following functions come with default implementations, but may be customized.

```@docs
random_offdiagonal
Hamiltonians.LOStructure
dimension
has_adjoint
allows_address_type
Base.eltype
VectorInterface.scalartype
mul!
```

This interface relies on unexported functionality, including
```@docs
Hamiltonians.adjoint
Hamiltonians.dot
Hamiltonians.AbstractOffdiagonals
Hamiltonians.Offdiagonals
Hamiltonians.check_address_type
Hamiltonians.number_conserving_dimension
Hamiltonians.number_conserving_bose_dimension
Hamiltonians.number_conserving_fermi_dimension
```

## Interface tests
Helper functions that can be used for testing the various interfaces are provided in the 
(unexported) submodule `Rimu.InterfaceTests`. 

```@docs
Rimu.InterfaceTests
```

### Testing functions
```@docs
Rimu.InterfaceTests.test_hamiltonian_interface
Rimu.InterfaceTests.test_hamiltonian_structure
Rimu.InterfaceTests.test_observable_interface
Rimu.InterfaceTests.test_operator_interface
```

## Index
```@index
Pages   = ["custom_hamiltonians.md"]
```
