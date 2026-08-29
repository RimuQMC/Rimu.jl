# Module `BitStringAddresses`

This module contains the implementations of the underlying data structures to efficiently represent
[Fock states](https://en.wikipedia.org/wiki/Fock_state). In Rimu.jl, Fock states are used as the basis 
for linear (Hilbert) space. The concrete implementations of Fock states are used as addresses to identify
matrix elements of operators, or elements of state vectors.

## Fock addresses

Many Hamiltonians in Rimu.jl are implemented in a generic way and crucial information that defines the size of Hilbert space and the quantum statistics of particle are defined by passing a Fock state address. Rimu.jl provides a variety of address types that enable the efficient implementation of 
many-body Hamiltonians. All implementations of Fock states are subtype to the
[`AbstractFockAddress`](@ref) abstract type. For Fock states representing a single component of indistinguishable quantum particles there is the more specialised 
type [`SingleComponentFockAddress`](@ref).

Examples of Fock addresses are:

- [`BoseFS`](@ref): Bosonic Fock state with fixed mode number and fixed or variable
    particle number.
- [`FermiFS`](@ref): Fermionic Fock state with fixed mode number and fixed or variable
    particle number.
- [`HardcoreBoseFS`](@ref): Fock state for hardcore bosons with fixed mode number and fixed
    or variable particle number.
- [`CompositeFS`](@ref) Multi-component Fock state composed of the above types.

The various address types make use efficient underlying data storage types like [`BitString`](@ref) and [`SortedParticleList`](@ref).

### Fock address API

```@docs
Rimu.Interfaces.AbstractFockAddress
Rimu.Interfaces.num_particles
Rimu.Interfaces.num_components
Rimu.Interfaces.num_modes
Rimu.Interfaces.num_modes_check_equal
Rimu.Interfaces.num_modes_are_equal
```

```@autodocs
Modules = [BitStringAddresses]
Pages = ["BitStringAddresses.jl","fockaddress.jl","bosefs.jl","hardcorebosefs.jl","fermifs.jl","multicomponent.jl"]
Private = false
```

## Internal representations

The addresses types [`BoseFS`](@ref), [`FermiFS`](@ref) and [`HardcoreBoseFS`](@ref), where the 
particle number is either fixed (for fixed particle number `BoseFS`) or severely constrained, are 
implemented as either bitstrings through [`BitString`](@ref), or sorted lists of particles 
with [`SortedParticleList`](@ref). This allows for a space efficient representation.

Therewhile, [`BoseFS{missing}`](@ref), which allows for flexible and unconstrained particle number
internally uses the occupation number representation, which allows it to handle excitation 
operations that change the particle number. This is fast but requires more storage space.

### Internal APIs

```@autodocs
Modules = [BitStringAddresses]
Pages = ["bitstring.jl", "sortedparticlelist.jl"]
Private = false
```

The following APIs are used by [Module `Hamiltonians`](@ref).
```@docs
BitStringAddresses.ModeMap
BitStringAddresses.FermiFS2CModes
BitStringAddresses.full_mode_maps
```

## Index
```@index
Pages   = ["addresses.md"]
```
