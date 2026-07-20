"""
    CompositeFS(addresses::SingleComponentFockAddress...) <: AbstractFockAddress

Used to encode addresses for multi-component models. All component addresses
are expected to have the same number of modes.

See also: [`BoseFS`](@ref), [`FermiFS`](@ref), [`SingleComponentFockAddress`](@ref),
[`num_modes`](@ref), [`FermiFS2C`](@ref), [`AbstractFockAddress`](@ref).
"""
struct CompositeFS{C,N,M,T} <: AbstractFockAddress{N,M}
    components::T
    # C: components, N: total particles, M: modes in each component,
    # T: tuple type with constituent address types
    function CompositeFS{C,N,M,T}(adds::T) where {C,N,M,T}
        return new{C,N,M,T}(adds)
    end
    function CompositeFS{C,N,M,T}(adds...) where {C,N,M,T}
        return new{C,N,M,T}(adds)
    end
end

# Slow constructor - not to be used internally
function CompositeFS(adds::Vararg{SingleComponentFockAddress})
    N = sum(a -> num_particles(typeof(a)), adds)
    M1, M2 = extrema(num_modes, adds)
    if M1 ≠ M2
        throw(ArgumentError("all addresses must have the same number of modes"))
    end
    return CompositeFS{length(adds),N,M1,typeof(adds)}(adds)
end
function Interfaces.num_particles(cfs::CompositeFS{<:Any,missing})
    sum(num_particles, cfs.components)
end # only required for missing, as the fallback for others is defined in the abstract type

function Interfaces.maximum_mode_occupation(::Type{<:CompositeFS{C,N,M,T}}) where {C,N,M,T}
    Interfaces.maximum_mode_occupation.(fieldtypes(T))
end

Interfaces.num_components(::Type{<:CompositeFS{C}}) where {C} = C
Base.hash(c::CompositeFS, u::UInt) = hash(c.components, u)

function print_address(io::IO, c::CompositeFS{C}; compact=false) where {C}
    if compact
        for add in c.components[1:end-1]
            print_address(io, add; compact)
            print(io, " ⊗ ")
        end
        print_address(io, c.components[end]; compact)
    else
        println(io, "CompositeFS(")
        for add in c.components
            println(io, "  ", add, ",")
        end
        print(io, ")")
    end
end

function Base.reverse(c::CompositeFS)
    typeof(c)(map(reverse, c.components))
end

"""
    time_reverse(addr)
Apply the time-reversal operation on a two-component Fock address that flips all the spins.

Requires each component address to have the same type.
"""
function time_reverse(c::CompositeFS{2,N,M,T}) where {N, M, T <: NTuple{2}}
    return CompositeFS{2,N,M,T}(reverse(c.components))
end

"""
    update_component(c::CompositeFS, new, ::Val{i})

Replace the `i`-th component in `c` with `new`. Used for updating a single component in the
address.
"""
function update_component(c::CompositeFS, new, ::Val{I}) where {I}
    return typeof(c)(_update_component(c.components, new, Val(I)))
end

@inline _update_component((a, as...), new, ::Val{1}) = (new, as...)
@inline function _update_component((a, as...), new, ::Val{I}) where {I}
    return (a, _update_component(as, new, Val(I - 1))...)
end

Base.isless(a::T, b::T) where {T<:CompositeFS} = isless(a.components, b.components)

function onr(a::CompositeFS)
    map(onr, a.components)
end

# Convenience
"""
    FermiFS2C <: AbstractFockAddress
    FermiFS2C(onr_a, onr_b)
    FermiFS2C{missing}(onr_a, onr_b)

Fock state address with two fermionic (spin) components. Alias for [`CompositeFS`](@ref)
with two [`FermiFS`](@ref) components. If the type parameter `missing` is specified, the
number of particles is not known at compile time. This is useful for spinful fermionic
systems where the number of particles in each spin channel may vary.

Construct by specifying either two compatible [`FermiFS`](@ref)s, two [`onr`](@ref)s, or
the number of modes followed by `mode => occupation_number` pairs, where
`occupation_number = 1` will put a particle in the first
component and `occupation_number = -1` will put a particle in the second component.
See examples below.

# Examples

```jldoctest
julia> FermiFS2C(FermiFS(1,0,0), FermiFS(0,1,1))
CompositeFS(
  FermiFS(1, 0, 0),
  FermiFS(0, 1, 1),
)

julia> FermiFS2C((1,0,0), (0,1,1))
CompositeFS(
  FermiFS(1, 0, 0),
  FermiFS(0, 1, 1),
)

julia> FermiFS2C{missing}((1,0,0), (0,1,1)) # number non-conserving, spin flips allowed
CompositeFS(
  FermiFS{missing}((1, 0, 0)),
  FermiFS{missing}((0, 1, 1)),
)

julia> FermiFS2C{missing}(3, 1 => 1, 2 => -1, 3 => -1)
CompositeFS(
  FermiFS{missing}((1, 0, 0)),
  FermiFS{missing}((0, 1, 1)),
)

julia> fs"|↑↓↓⟩" # \\uparrow(tab) -> ↑, \\downarrow(tab) -> ↓, \\rangle(tab) -> ⟩
CompositeFS(
  FermiFS(1, 0, 0),
  FermiFS(0, 1, 1),
)

julia> fs"|↑↓↓⇅⟩{}" # \\dblarrowupdown(tab) -> ⇅
CompositeFS(
  FermiFS{missing}((1, 0, 0, 1)),
  FermiFS{missing}((0, 1, 1, 1)),
)
```

See also: [`CompositeFS`](@ref), [`FermiFS`](@ref), [`@fs_str`](@ref).
"""
const FermiFS2C{N1,N2,M,N,F1,F2} =
    CompositeFS{2,N,M,Tuple{F1,F2}} where {F1<:FermiFS{N1,M},F2<:FermiFS{N2,M}}

FermiFS2C(f1::FermiFS{<:Any,M}, f2::FermiFS{<:Any,M}) where {M} = CompositeFS(f1, f2)
FermiFS2C(onr_a, onr_b) = FermiFS2C(FermiFS(onr_a), FermiFS(onr_b))
FermiFS2C{missing}(onr_a, onr_b) = FermiFS2C(FermiFS{missing}(onr_a), FermiFS{missing}(onr_b))
FermiFS2C(M::Integer, pairs::Pair...) = FermiFS2C(M, pairs)
FermiFS2C{missing}(M::Integer, pairs::Pair...) = FermiFS2C{missing}(M, pairs)
function FermiFS2C(M::Integer, pairs)
    up_pairs = filter(p -> p[2] > 0, pairs)
    down_pairs = map(p -> p[1] => -p[2], filter(p -> p[2] < 0, pairs))
    return FermiFS2C(FermiFS(M, up_pairs), FermiFS(M, down_pairs))
end
function FermiFS2C{missing}(M::Integer, pairs)
    up_pairs = filter(p -> p[2] > 0, pairs)
    down_pairs = map(p -> p[1] => -p[2], filter(p -> p[2] < 0, pairs))
    return FermiFS2C(FermiFS{missing}(M, up_pairs), FermiFS{missing}(M, down_pairs))
end

function print_address(io::IO, f::FermiFS2C; compact=false)
    if compact
        o1, o2 = onr(f)
        str = join(
            [i && j ? '⇅' : i ? '↑' : j ? '↓' : '⋅' for (i, j) in zip(Bool.(o1), Bool.(o2))]
        )
        if ismissing(num_particles(typeof(f)))
            print(io, "|", str, "⟩{}")
        else
            print(io, "|", str, "⟩")
        end
    else
        # Show as normal CompositeFS
        invoke(print_address, Tuple{typeof(io),CompositeFS}, io, f)
    end
end

"""
    FermiFS2CModes

This struct stores the occupied and unoccupied mode maps associated with an
address of type [`FermiFS2C`](@ref). It should be constructed using the
[`full_mode_maps`](@ref) function.

The struct has two fields, `occupied` and `unoccupied`, each containing a
`ModeMap` represented as a two-element `Tuple`:

- Index `1` corresponds to the α spin channel
- Index `2` corresponds to the β spin channel

This convention follows the spin-channel indexing defined in [`FermiFS2C`](@ref).

See also
[`FermiFS2C`](@ref), [`ModeMap`](@ref), [`occupied_mode_map`](@ref),
and [`unoccupied_mode_map`](@ref).
"""
struct FermiFS2CModes{TI<:FermiFSIndex,OA,OB,UA,UB}
    occupied::Tuple{ModeMap{OA,TI},ModeMap{OB,TI}}
    unoccupied::Tuple{ModeMap{UA,TI},ModeMap{UB,TI}}
end

"""
    full_mode_maps(addr::FermiFS2C)

The constructor function of [`FermiFS2CModes`](@ref).
"""
function full_mode_maps(addr::FermiFS2C)
    occupied_modes = (occupied_mode_map(addr.components[1]), occupied_mode_map(addr.components[2]))
    unoccupied_modes = (unoccupied_mode_map(addr.components[1]), unoccupied_mode_map(addr.components[2]))
    FermiFS2CModes(occupied_modes, unoccupied_modes)
end
