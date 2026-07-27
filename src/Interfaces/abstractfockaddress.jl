"""
    AbstractFockAddress

Abstract type representing a Fock state.

## Implementations
* [`SingleComponentFockAddress`](@ref Main.BitStringAddresses.SingleComponentFockAddress)
    is a supertype for single-component Fock states.
* [`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS) for multi-component Fock states,

## Interface
The following functions can be called on the type or an instance of a subtype of
`AbstractFockAddress:
* [`num_components()`](@ref): Returns the number of components in the Fock state.
* [`num_modes()`](@ref): Returns the number of modes in the Fock state. This returns a tuple
    for multi-component Fock states, and an integer for single-component Fock states.
* [`num_modes_are_equal()`](@ref): Returns `true` if all components of a multi-component
    Fock state have the same number of modes, and `false` otherwise. Returns `true` for
    single-component Fock states.
* [`num_modes_check_equal()`](@ref): Returns the number of modes in a Fock state as a single
    integer, and throws an `ArgumentError` if the components of a multi-component Fock state
    have different numbers of modes.
* [`num_particles()`](@ref): Returns the number of particles in the Fock state. May return
    `missing` if called on a type that allows for a variable number of particles.
* [`maximum_mode_occupation()`](@ref): Returns the maximum number of particles that can occupy
    a single mode in the Fock state. Returns an integer for single-component Fock states,
    and a tuple for multi-component Fock states.

See also [`SingleComponentFockAddress`](@ref Main.BitStringAddresses.SingleComponentFockAddress),
[`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS), [`BoseFS`](@ref Main.BitStringAddresses.BoseFS),
[`FermiFS`](@ref Main.BitStringAddresses.FermiFS), [`num_particles`](@ref num_particles),
[`num_modes`](@ref), [`num_modes_check_equal`](@ref), [`num_components`](@ref),
[`maximum_mode_occupation`](@ref).
"""
abstract type AbstractFockAddress{N} end

# `AbstractFockAddress`es can be reconstructed from their printout.
Base.typeinfo_implicit(::Type{<:AbstractFockAddress}) = true

"""
    num_particles(::Type{<:AbstractFockAddress})
    num_particles(::AbstractFockAddress)

Number of particles represented by address.

See also [`num_modes`](@ref), [`num_modes_check_equal`](@ref), [`num_components`](@ref),
[`maximum_mode_occupation`](@ref),
[`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS).
"""
num_particles(a::AbstractFockAddress) = num_particles(typeof(a))
num_particles(::Type{<:AbstractFockAddress{N}}) where {N} = N

"""
    num_modes(::Type{<:AbstractFockAddress})
    num_modes(::AbstractFockAddress)

Number of modes represented by address. Returns a tuple for multi-component addresses, and
an integer for single-component addresses.

See also [`num_modes_check_equal`](@ref), [`num_modes_are_equal`](@ref),
[`num_particles`](@ref), [`num_components`](@ref), [`maximum_mode_occupation`](@ref),
[`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS).
"""
num_modes(a::AbstractFockAddress) = num_modes(typeof(a))

"""
    num_modes_are_equal(::Type{<:AbstractFockAddress)::Bool
    num_modes_are_equal(address::AbstractFockAddress)::Bool

Check if all components of a multi-component address have the same number of modes. Returns
`true` for single-component addresses. For multi-component addresses, returns `true` if all
components have the same number of modes, and `false` otherwise.

See also [`num_modes_check_equal`](@ref), [`num_modes`](@ref), [`num_particles`](@ref),
[`num_components`](@ref), [`maximum_mode_occupation`](@ref),
[`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS).
"""
num_modes_are_equal(a::AbstractFockAddress) = num_modes_are_equal(typeof(a))

"""
    num_modes_check_equal(::Type{<:AbstractFockAddress})::Int
    num_modes_check_equal(address::AbstractFockAddress)::Int
    num_modes_check_equal(address1, address2)::Int

Check that all components of a multi-component address have the same number of modes, and
return the number of modes. Throws an `ArgumentError` if the components have different
numbers of modes. For a single-component address, simply returns the number of modes.
When called with two addresses, checks that both addresses have the same number of modes,
and throws an `ArgumentError` if they do not.

See also [`num_modes_are_equal`](@ref), [`num_modes`](@ref), [`num_components`](@ref),
[`maximum_mode_occupation`](@ref),
[`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS),
[`SingleComponentFockAddress`](@ref Main.BitStringAddresses.SingleComponentFockAddress).
"""
num_modes_check_equal(a::AbstractFockAddress) = num_modes_check_equal(typeof(a))
function num_modes_check_equal(a::AbstractFockAddress, b::AbstractFockAddress)
    num_modes_check_equal(typeof(a), typeof(b))
end
function num_modes_check_equal(A::Type{<:AbstractFockAddress}, B::Type{<:AbstractFockAddress})
    MA = num_modes_check_equal(A)
    MB = num_modes_check_equal(B)
    if MA != MB
        throw(ArgumentError("Address types $A and $B have different numbers of modes: $MA and $MB"))
    end
    return MA
end

"""
    num_components(::Type{<:AbstractFockAddress})
    num_components(::AbstractFockAddress)

Number of components in address.

See also [`num_modes`](@ref), [`num_modes_check_equal`](@ref), [`num_particles`](@ref),
[`maximum_mode_occupation`](@ref),
[`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS).
"""
num_components(b::AbstractFockAddress) = num_components(typeof(b))

"""
    maximum_mode_occupation(::Type{<:AbstractFockAddress})
    maximum_mode_occupation(::AbstractFockAddress)
    maximum_mode_occupation(::AbstractHamiltonian)

Maximum number of particles that can occupy a single mode in the Fock space spanned by the
address type. The minimum is always zero. When called on an [`AbstractHamiltonian`](@ref) it
may provide further information about the maximum mode occupation based on the Hamiltonian's
structure.

Returns an integer for [`SingleComponentFockAddress`](@ref Main.BitStringAddresses.SingleComponentFockAddress)s, and a tuple for the
multi-component [`CompositeFS`](@ref Main.BitStringAddresses.CompositeFS) Fock addresses.

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

julia> maximum_mode_occupation(FroehlichPolaron(BoseFS{missing,20}(); mode_cutoff=5))
5
```

See also [`num_particles`](@ref), [`num_modes`](@ref), [`num_components`](@ref),
[`BoseFS`](@ref Main.BitStringAddresses.BoseFS), [`FermiFS`](@ref Main.BitStringAddresses.FermiFS), [`HardcoreBoseFS`](@ref Main.BitStringAddresses.HardcoreBoseFS).
"""
maximum_mode_occupation(a::AbstractFockAddress) = maximum_mode_occupation(typeof(a))
