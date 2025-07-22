"""
    AbstractFockAddress{N,M}

Abstract type representing a Fock state with `N` particles and `M` modes.

See also [`SingleComponentFockAddress`](@ref Main.SingleComponentFockAddress),
[`CompositeFS`](@ref Main.CompositeFS), [`BoseFS`](@ref Main.BoseFS),
[`FermiFS`](@ref Main.FermiFS), [`num_particles`](@ref num_particles),
[`num_modes`](@ref num_modes), [`num_components`](@ref num_components).
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
