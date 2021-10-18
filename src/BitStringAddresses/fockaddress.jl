"""
    AbstractFockAddress

Supertype representing a Fock state.
"""
abstract type AbstractFockAddress{N,M} end

"""
    num_particles(::Type{<:AbstractFockAddress})
    num_particles(::AbstractFockAddress)

Number of particles represented by address.
"""
num_particles(a::AbstractFockAddress) = num_particles(typeof(a))
num_particles(::Type{<:AbstractFockAddress{N}}) where {N} = N

"""
    num_modes(::Type{<:AbstractFockAddress})
    num_modes(::AbstractFockAddress)

Number of modes represented by address.
"""
num_modes(a::AbstractFockAddress) = num_modes(typeof(a))
num_modes(::Type{<:AbstractFockAddress{<:Any,M}}) where {M} = M

"""
    num_components(::Type{<:AbstractFockAddress})
    num_components(::AbstractFockAddress)

Number of components in address.
"""
num_components(b::AbstractFockAddress) = num_components(typeof(b))

"""
    SingleComponentFockAddress{M}

A type representing a single component Fock state with `M` modes.

Implemented subtypes: [`BoseFS`](@ref), [`FermiFS`](@ref).

# Supported functionality

* [`find_mode`](@ref)
* [`find_occupied_mode`](@ref)
* [`num_occupied_modes`](@ref)
* [`occupied_modes`](@ref)
* [`move_particle`](@ref)

"""
abstract type SingleComponentFockAddress{N,M} <: AbstractFockAddress{N,M} end

num_components(::Type{<:SingleComponentFockAddress}) = 1

"""
    find_mode(::SingleComponentFockAddress, i)

Find the `i`-th mode in address. Returns [`BoseFSIndex`](@ref) for [`BoseFS`](@ref), and
an integer index for [`FermiFS`](@ref). Does not check bounds.

```jldoctest
julia> find_mode(BoseFS((1, 0, 2)), 2)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 0
 2
 2

julia> find_mode(FermiFS((1, 1, 1, 0)), 2)
FermiFSIndex(occnum=1, mode=2)
```
"""
find_mode

"""
    find_occupied_mode(::SingleComponentFockAddress, k)

Find the `k`-th occupied mode in address. Returns [`BoseFSIndex`](@ref) for
[`BoseFS`](@ref), and an integer for [`FermiFS`](@ref). When unsuccessful it
returns zero.

# Example

```jldoctest
julia> find_occupied_mode(BoseFS((1, 0, 2)), 2)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 2
 3
 3

julia> find_occupied_mode(FermiFS((1, 1, 1, 0)), 2)
FermiFSIndex(occnum=1, mode=2)
```
"""
find_occupied_mode

"""
    is_occupied(::SingleComponentFockAddress, i)

Return `true` if index `i` points to an occupied mode.
"""
is_occupied

"""
    num_occupied_modes(::SingleComponentFockAddress)

Get the number of occupied modes in address. Equivalent to
`length(`[`occupied_modes`](@ref)`(address))`, or the number of non-zeros in its ONR
representation.

# Example

```jldoctest
julia> num_occupied_modes(BoseFS((1, 0, 2)))
2
julia> num_occupied_modes(FermiFS((1, 1, 1, 0)))
3
```
"""
num_occupied_modes

"""
    occupied_modes(::SingleComponentFockAddress)

Iterate over all occupied modes in an address. Iterates [`BoseFSIndex`](@ref) for
[`BoseFS`](@ref), and an integers for [`FermiFS`](@ref).

# Example

```jldoctest
julia> b = BoseFS((1,5,0,4));

julia> foreach(println, occupied_modes(b))
BoseFSIndex(occnum=1, mode=1, offset=0)
BoseFSIndex(occnum=5, mode=2, offset=2)
BoseFSIndex(occnum=4, mode=4, offset=9)
```

```jldoctest
julia> f = FermiFS((1,1,0,1,0,0,1));

julia> foreach(println, occupied_modes(f))
FermiFSIndex(occnum=1, mode=1)
FermiFSIndex(occnum=1, mode=2)
FermiFSIndex(occnum=1, mode=4)
FermiFSIndex(occnum=1, mode=7)
```
"""
occupied_modes

"""
    move_particle(add::SingleComponentFockAddress, i, j) -> nadd, α

Move particle from mode `i` to mode `j`. Returns the new Fock state address `nadd` and
amplitude `α`. Equivalent to
```math
a^{\\dagger}_i a_j |\\mathrm{add}\\rangle \\to α|\\mathrm{nadd}\\rangle
```

Note that the modes in [`BoseFS`](@ref) are indexed by [`BoseFSIndex`](@ref), while the ones
in [`FermiFS`](@ref) are indexed by integers (see example below).  For illegal moves where `α
== 0` the value of `nadd` is undefined.

# Example

```jldoctest
julia> b = BoseFS((1, 1, 3, 0))
BoseFS{5,4}((1, 1, 3, 0))

julia> i = find_occupied_mode(b, 2)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 1
 2
 2

julia> j = find_mode(b, i.mode + 1)
3-element Rimu.BitStringAddresses.BoseFSIndex with indices SOneTo(3):
 3
 3
 4

julia> move_particle(b, i, j)
(BoseFS{5,4}((1, 0, 4, 0)), 2.0)

julia> move_particle(b, j, j)
(BoseFS{5,4}((1, 1, 3, 0)), 3.0)
```

```jldoctest
julia> f = FermiFS((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0))
FermiFS{7,12}((1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0))

julia> i = find_occupied_mode(f, 2)
FermiFSIndex(occnum=1, mode=6)

julia> j = find_mode(f, i.mode - 1)
FermiFSIndex(occnum=0, mode=5)

julia> move_particle(f, i, j)
(FermiFS{7,12}((1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0)), 1.0)

```
"""
move_particle

"""
    excitation(a::SingleComponentFockAddress, creations::NTuple{N}, destructions::NTuple{N})

Generate an excitation on address `a` by applying `creations` and `destructions`, which are
tuples of the appropriate address indices (i.e. integers for fermions and `BoseFSIndex` for
bosons).

Returns the new address and the value. If the excitations is illegal, returns an arbitrary
address and 0.0.

# Example

```jldoctest
julia> f = FermiFS((1,1,0,0,1,1,1,1))
FermiFS{6,8}((1, 1, 0, 0, 1, 1, 1, 1))

julia> i, j, k, l = find_mode(f, (3,4,2,5))
(FermiFSIndex(occnum=0, mode=3), FermiFSIndex(occnum=0, mode=4), FermiFSIndex(occnum=1, mode=2), FermiFSIndex(occnum=1, mode=5))

julia> excitation(f, (i,j), (k,l))
(FermiFS{6,8}((1, 0, 1, 1, 0, 1, 1, 1)), -1.0)

```
"""
excitation

"""
    OccupiedModeMap(add)

Get a map of occupied modes in address as an `AbstractVector` of indices compatible with
[`excitation`](@ref) or [`move_particle`](@ref).

`OccupiedModeMap(add)[i]` contains the index for the `i`-th occupied mode.

This is useful because repeatedly looking for occupied modes with `find_occupied_mode` can
be time-consuming.

# Example

```jldoctest
julia> b = BoseFS((10, 0, 0, 0, 2, 0, 1))
BoseFS{13,7}((10, 0, 0, 0, 2, 0, 1))

julia> OccupiedModeMap(b)
3-element OccupiedModeMap{13, Rimu.BitStringAddresses.BoseFSIndex}:
 BoseFSIndex(occnum=10, mode=1, offset=0)
 BoseFSIndex(occnum=2, mode=5, offset=14)
 BoseFSIndex(occnum=1, mode=7, offset=18)

julia> f = FermiFS((1,1,1,1,0,0,0,0,1))
FermiFS{5,9}((1, 1, 1, 1, 0, 0, 0, 0, 1))

julia> OccupiedModeMap(f)
5-element OccupiedModeMap{5, Rimu.BitStringAddresses.FermiFSIndex}:
 FermiFSIndex(occnum=1, mode=1)
 FermiFSIndex(occnum=1, mode=2)
 FermiFSIndex(occnum=1, mode=3)
 FermiFSIndex(occnum=1, mode=4)
 FermiFSIndex(occnum=1, mode=9)
```
"""
struct OccupiedModeMap{N,T} <: AbstractVector{T}
    indices::SVector{N,T}
    length::Int
end

function OccupiedModeMap(add::SingleComponentFockAddress{N}) where {N}
    modes = occupied_modes(add)
    T = eltype(modes)
    # There are at most N occupied modes. This could be @generated for cases where N > M
    indices = MVector{N,T}(undef)
    i = 0
    for index in modes
        i += 1
        @inbounds indices[i] = index
    end
    return OccupiedModeMap(SVector(indices), i)
end

Base.size(om::OccupiedModeMap) = (om.length,)
function Base.getindex(om::OccupiedModeMap, i)
    @boundscheck 1 ≤ i ≤ om.length || throw(BoundsError(om, i))
    return om.indices[i]
end
