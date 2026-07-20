"""
    AbstractFockAddress{N,M}

Abstract type representing a Fock state with `N` particles and `M` modes.

See also [`SingleComponentFockAddress`](@ref Main.SingleComponentFockAddress),
[`CompositeFS`](@ref Main.CompositeFS), [`BoseFS`](@ref Main.BoseFS),
[`FermiFS`](@ref Main.FermiFS), [`num_particles`](@ref num_particles),
[`num_modes`](@ref num_modes), [`num_components`](@ref num_components),
[`maximum_mode_occupation`](@ref maximum_mode_occupation).
"""
abstract type AbstractFockAddress{N,M} end

# `AbstractFockAddress`es can be reconstructed from their printout.
Base.typeinfo_implicit(::Type{<:AbstractFockAddress}) = true

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
    maximum_mode_occupation(::Type{<:AbstractFockAddress})
    maximum_mode_occupation(::AbstractFockAddress)
    maximum_mode_occupation(::AbstractHamiltonian)

Maximum number of particles that can occupy a single mode in the Fock space spanned by the
address type. When called on an [`AbstractHamiltonian`](@ref) it may provide further
information about the maximum mode occupation based on the Hamiltonian's structure.

Returns an integer for [`SingleComponentFockAddress`](@ref)s, and a tuple for the
multi-component [`CompositeFS`](@ref) Fock addresses.

## Example
```jldoctest
julia> maximum_mode_occupation(FermiFS{2,4})
1

julia> maximum_mode_occupation(BoseFS(3, 10, 0))
13

julia> maximum_mode_occupation(BoseFS{missing}(3, 10, 0; type=UInt16)) |> Int
65535

julia> maximum_mode_occupation(CompositeFS(BoseFS(1,2,3), FermiFS(1,0,0)))
(6, 1)
```

See also [`num_particles`](@ref), [`num_modes`](@ref), [`num_components`](@ref),
[`BoseFS`](@ref), [`FermiFS`](@ref), [`HardcoreBoseFS`](@ref).
"""
maximum_mode_occupation(a::AbstractFockAddress) = maximum_mode_occupation(typeof(a))
