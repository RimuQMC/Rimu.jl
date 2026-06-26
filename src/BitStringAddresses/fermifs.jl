"""
    FermiFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` fermions of the same spin in `M` modes by
wrapping a [`BitString`](@ref), or a [`SortedParticleList`](@ref). Which is wrapped is
chosen automatically based on the properties of the address. Set `N` to `missing` if the
number of particles is not known at compile time.

# Constructors

* `FermiFS{[N,M]}(val::Integer...)`: Create `FermiFS{N,M}` from occupation numbers. This is
  type-stable if the number of modes `M` and the number of particles `N` are provided.
  Otherwise, `M` and `N` are inferred from the arguments.

* `FermiFS{[N,M]}(onr)`: Create `FermiFS{N,M}`  from occupation number representation, see
  [`onr`](@ref). This is efficient if `N` and `M` are provided, and `onr` is a
  statically-sized collection, such as a `Tuple{M}` or `SVector{M}`.

* `FermiFS{[N,M]}([M, ]pairs...)`: Provide the number of modes `M` and pairs of the form
  `mode => 1`. If `M` is provided as a type parameter, it should not be provided as the
  first argument.  Useful for creating sparse addresses. `pairs` can be multiple arguments
  or an iterator of pairs.

* `FermiFS{N,M,S}(bs::S)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`, or whether each mode only contains one particle.

* [`@fs_str`](@ref): Addresses are sometimes printed in a compact manner. This
  representation can also be used as a constructor. See the examples below.

# Examples

```jldoctest
julia> FermiFS{3,5}(0, 1, 1, 1, 0)
FermiFS{3,5}(0, 1, 1, 1, 0)

julia> FermiFS([abs(i - 3) ≤ 1 for i in 1:5])
FermiFS{3,5}(0, 1, 1, 1, 0)

julia> FermiFS(5, 2 => 1, 3 => 1, 4 => 1)
FermiFS{3,5}(0, 1, 1, 1, 0)

julia> FermiFS{3,5}(i => 1 for i in 2:4)
FermiFS{3,5}(0, 1, 1, 1, 0)

julia> fs"|⋅↑↑↑⋅⟩" # \\uparrow(tab) -> ↑, \\cdot(tab) -> ⋅, \\rangle(tab) -> ⟩
FermiFS{3,5}(0, 1, 1, 1, 0)

julia> fs"|f 5: 2 3 4⟩"
FermiFS{3,5}(0, 1, 1, 1, 0)
```

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`CompositeFS`](@ref),
[`FermiFS2C`](@ref), [`BitString`](@ref), [`OccupationNumberFS`](@ref), [`@fs_str`](@ref).
"""
struct FermiFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S
end

function check_fermi_onr(onr, N, M)
    ismissing(N) || sum(onr) == N ||
        throw(ArgumentError("Invalid ONR: $N particles expected, $(sum(onr)) given."))
    length(onr) == M ||
        throw(ArgumentError("Invalid ONR: $M modes expected, $(length(onr)) given."))
    all(in((0, 1)), onr) ||
        throw(ArgumentError("Invalid ONR: may only contain 0s and 1s."))
end

function FermiFS{N,M,S}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {N,M,S}
    @boundscheck begin
        check_fermi_onr(onr, N, M)
        if S <: BitString
            B = num_bits(S)
            M == B || throw(ArgumentError(
                "invalid ONR: $B-bit BitString does not fit $M modes"
            ))
        elseif S <: SortedParticleList
            N == num_particles(S) && M == num_modes(S) || throw(ArgumentError(
                "invalid ONR: $S does not fit $N particles in $M modes"
            ))
        end
    end
    return FermiFS{N,M,S}(from_fermi_onr(S, onr))
end
function FermiFS{N,M}(onr::Union{AbstractArray{<:Integer},NTuple{M,<:Integer}}) where {N,M}
    @boundscheck check_fermi_onr(onr, N, M)
    if ismissing(N)
        S = typeof(BitString{M}(0))
        return FermiFS{N,M,S}(from_fermi_onr(S, onr))
    end

    spl_type = select_int_type(M)
    # Pick smaller address type, but prefer dense.
    # Alway pick dense if it fits into one chunk.

    # Compute the size of container in words
    sparse_sizeof = ceil(Int, N * sizeof(spl_type) / 8)
    dense_sizeof = ceil(Int, M / 64)
    if dense_sizeof == 1 || dense_sizeof ≤ sparse_sizeof
        S = typeof(BitString{M}(0))
    else
        S = SortedParticleList{N,M,spl_type}
    end
    return FermiFS{N,M,S}(from_fermi_onr(S, onr))
end
function FermiFS(onr)
    onr = Tuple(onr)
    M = length(onr)
    N = sum(onr)
    return FermiFS{N,M}(onr)
end
function FermiFS{N}(onr) where {N}
    onr = Tuple(onr)
    M = length(onr)
    return FermiFS{N,M}(onr)
end

FermiFS(vals::Integer...) = FermiFS(vals) # list occupation numbers
FermiFS(val::Integer) = FermiFS((val,)) # single mode
FermiFS{N}(vals::Integer...) where N = FermiFS{N}(vals) # list occupation numbers
FermiFS{N}(val::Integer) where {N} = FermiFS{N}((val,)) # single mode
FermiFS{N,M}(vals::Integer...) where {N,M} = FermiFS{N,M}(vals)

# Sparse constructors
FermiFS(M::Integer, pairs::Pair...) = FermiFS(M, pairs)
FermiFS{N}(M::Integer, pairs::Pair...) where {N} = FermiFS{N}(M, pairs)
FermiFS(M::Integer, pairs) = FermiFS(sparse_to_onr(M, pairs))
FermiFS{N}(M::Integer, pairs) where {N} = FermiFS{N}(sparse_to_onr(M, pairs))
FermiFS{N,M}(pairs::Vararg{Pair}) where {N,M} = FermiFS{N,M}(pairs)
FermiFS{N,M}(pairs) where {N,M} = FermiFS{N}(sparse_to_onr(M, pairs))
FermiFS(pairs::Pair...) = throw(ArgumentError("number of modes must be provided"))

function print_address(io::IO, f::FermiFS{N,M}; compact=false) where {N,M}
    if compact && f.bs isa SortedParticleList
        print(io, "|f ", M, ": ", join(Int.(f.bs.storage), ' '), "⟩")
    elseif compact && ismissing(N)
        print(io, "|", join(map(o -> o == 0 ? '⋅' : '↑', onr(f))), "⟩{}")
    elseif compact
        print(io, "|", join(map(o -> o == 0 ? '⋅' : '↑', onr(f))), "⟩")
    elseif f.bs isa SortedParticleList
        print(io, "FermiFS{$N,$M}(", onr_sparse_string(onr(f)), ")")
    else
        print(io, "FermiFS{$N,$M}", tuple(onr(f)...))
    end
end

function excitation(
    a::FermiFS{N,M,S}, creations::NTuple{NC}, destructions::NTuple{ND}
) where {N,M,S,NC,ND}
    new_bs, value = fermi_excitation(a.bs, creations, destructions)
    NN = ismissing(N) ? missing : N + NC - ND # done at compile time
    return FermiFS{NN,M,S}(new_bs), value # carries sign, different from HardcoreBoseFS
end

# joint functions for FermiFS and HardcoreBoseFS
const FermiOrHardcoreBoseFS{N,M,S} = Union{FermiFS{N,M,S}, HardcoreBoseFS{N,M,S}}

Interfaces.num_particles(a::FermiOrHardcoreBoseFS{missing}) = count_ones(a.bs)
# only required for missing, as the fallback for other types is defined in the abstract type
Base.bitstring(a::FermiOrHardcoreBoseFS) = bitstring(a.bs)
Base.isless(a::F, b::F) where {F <: FermiOrHardcoreBoseFS} = isless(a.bs, b.bs)
Base.hash(a::FermiOrHardcoreBoseFS, h::UInt) = hash(a.bs, h)
Base.:(==)(a::FermiOrHardcoreBoseFS, b::FermiOrHardcoreBoseFS) = a.bs == b.bs

num_occupied_modes(a::FermiOrHardcoreBoseFS) = num_particles(a)
num_unoccupied_modes(a::FermiOrHardcoreBoseFS) = num_modes(a) - num_particles(a)

occupied_modes(a::FermiOrHardcoreBoseFS{N,<:Any,S}) where {N,S} = FermiOccupiedModes{N,S}(a.bs)
unoccupied_modes(a::FermiOrHardcoreBoseFS{N,M,S}) where {N,M,S} = FermiUnoccupiedModes{M - N,S}(a.bs)
function unoccupied_modes(a::FermiOrHardcoreBoseFS{missing,<:Any,S}) where {S}
    FermiUnoccupiedModes{missing,S}(a.bs)
end

@inline function onr(a::FermiOrHardcoreBoseFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    @inbounds for (_, mode) in occupied_modes(a)
        result[mode] = 1
    end
    return SVector(result)
end

find_mode(a::FermiOrHardcoreBoseFS, i, occ=nothing) = fermi_find_mode(a.bs, i)
function find_occupied_mode(a::FermiOrHardcoreBoseFS, i::Integer)
    for k in occupied_modes(a)
        i -= 1
        i == 0 && return k
    end
    return FermiFSIndex(0, 0, 0)
end

function Base.reverse(f::FermiOrHardcoreBoseFS)
    return typeof(f)(reverse(f.bs))
end

"""
    unoccupied_mode_map(address::Union{FermiFS, HardcoreBoseFS}) <: AbstractVector

Get a map of unoccupied modes in `address` as an `AbstractVector` of indices compatible
with [`excitation`](@ref).

`unoccupied_mode_map(address)[i]` contains the index for the `i`-th unoccupied mode.
This is useful because unoccupied modes is required in some cases.
`unoccupied_mode_map(address)` is an eager version of the iterator returned by
[`unoccupied_modes`](@ref). It is similar to [`onr`](@ref) but contains more information.

Note that this function is only implemented for addresses of type [`FermiFS`](@ref) and
[`HardcoreBoseFS`](@ref).

# Example

```jldoctest
julia> f = FermiFS(1,1,0,0)
FermiFS{2,4}(1, 1, 0, 0)

julia> mf = unoccupied_mode_map(f)
2-element Rimu.BitStringAddresses.ModeMap{2, FermiFSIndex}:
 FermiFSIndex(occnum=0, mode=3, offset=2)
 FermiFSIndex(occnum=0, mode=4, offset=3)

julia> mf == collect(unoccupied_modes(f))
true

```
See also [`occupied_mode_map`](@ref).
"""
function unoccupied_mode_map(addr::FermiOrHardcoreBoseFS)
    modes = unoccupied_modes(addr)
    T = eltype(modes)
    L = num_unoccupied_modes(addr)
    indices = MVector{L,T}(undef)
    i = 0
    for index in modes
        i += 1
        @inbounds indices[i] = index
    end
    return ModeMap(SVector(indices), i)
end
