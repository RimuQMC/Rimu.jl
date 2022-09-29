"""
    FermiFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` fermions of the same spin in `M` modes by
wrapping a `[`BitString`](@ref), or a [`SortedParticleList`](@ref). Which is wrapped is
chosen automatically based on the properties of the address.

# Constructors

* `FermiFS{[N,M]}(onr)`: Create `FermiFS{N,M}` from [`onr`](@ref) representation. This is
  efficient if `N` and `M` are provided, and `onr` is a statically-sized collection, such as
  a `Tuple{M}` or `SVector{M}`.

* `FermiFS{[N,M]}(M, pairs...)`: Provide the number of modes and `mode => occupation_number`
  pairs. If `N` and `M` are provided, the first argument is not needed. Useful for creating
  sparse addresses. `pairs` can be multiple arguments or an iterator of pairs.

* `FermiFS{N,M,S}(bs::S)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`, or whether each mode only contains one particle.

* [`@fs_str`](@ref): addresses are sometimes printed in a compact manner. This
  representation can also be used as a constructor. See the last example below.

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`BitString`](@ref).

# Examples

```jldoctest
julia> FermiFS{3,5}((0, 1, 1, 1, 0))
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> FermiFS([abs(i - 3) ≤ 1 for i in 1:5])
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> FermiFS(5, 2 => 1, 3 => 1, 4 => 1)
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> FermiFS{3,5}(i => 1 for i in 2:4)
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> fs"|⋅↑↑↑⋅⟩"
FermiFS{3,5}((0, 1, 1, 1, 0))

julia> fs"|f 5: 2 3 4⟩"
FermiFS{3,5}((0, 1, 1, 1, 0))
```

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`CompositeFS`](@ref),
[`FermiFS2C`](@ref).
"""
struct FermiFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S
end

function check_fermi_onr(onr, N)
    sum(onr) == N ||
        throw(ArgumentError("Invalid ONR: $N particles expected, $(sum(onr)) given."))
    all(in((0, 1)), onr) ||
        throw(ArgumentError("Invalid ONR: may only contain 0s and 1s."))
end

function FermiFS{N,M,S}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {N,M,S}
    @boundscheck check_fermi_onr(onr, N)
    return FermiFS{N,M,S}(from_fermi_onr(S, onr))
end
function FermiFS{N,M}(onr::Union{AbstractArray{<:Integer},NTuple{M,<:Integer}}) where {N,M}
    @boundscheck check_fermi_onr(onr, N)
    spl_type = select_int_type(M)
    S_sparse = SortedParticleList{N,M,spl_type}
    S_dense = typeof(BitString{M}(0))
    # Pick smaller address type, but prefer dense.
    # Alway pick dense if it fits into one chunk.
    sparse_size_64 = ceil(Int, sizeof(S_sparse) / 8)
    dense_size_64 = ceil(Int, sizeof(S_dense) / 8)
    if num_chunks(S_dense) == 1 || dense_size_64 ≤ sparse_size_64
        S = S_dense
    else
        S = S_sparse
    end
    return FermiFS{N,M,S}(from_fermi_onr(S, SVector{M,Int}(onr)))
end
function FermiFS(onr::Union{AbstractArray,Tuple})
    M = length(onr)
    N = sum(onr)
    return FermiFS{N,M}(onr)
end

# Sparse constructors
FermiFS(M, pair::Pair) = FermiFS(M, (pair,))
FermiFS(M, pairs...) = FermiFS(M, pairs)
FermiFS(M, pairs) = FermiFS(sparse_to_onr(M, pairs))
FermiFS{1,M}(pair::Pair) where {M} = FermiFS{1,M}((pair,))
FermiFS{N,M}(pairs...) where {N,M} = FermiFS{N,M}(pairs)
FermiFS{N,M}(pairs) where {N,M} = FermiFS{N,M}(sparse_to_onr(M, pairs))

function print_address(io::IO, f::FermiFS{N,M}; compact=false) where {N,M}
    if compact && f.bs isa SortedParticleList
        print(io, "|f ", M, ": ", join(Int.(f.bs.storage), ' '), "⟩")
    elseif compact
        print(io, "|", join(map(o -> o == 0 ? '⋅' : '↑', onr(f))), "⟩")
    elseif f.bs isa SortedParticleList
        print(io, "FermiFS{$N,$M}(", onr_sparse_string(onr(f)), ")")
    else
        print(io, "FermiFS{$N,$M}(", tuple(onr(f)...), ")")
    end
end

Base.bitstring(a::FermiFS) = bitstring(a.bs)
Base.isless(a::FermiFS, b::FermiFS) = isless(a.bs, b.bs)
Base.hash(a::FermiFS,  h::UInt) = hash(a.bs, h)
Base.:(==)(a::FermiFS, b::FermiFS) = a.bs == b.bs
num_occupied_modes(::FermiFS{N}) where {N} = N
occupied_modes(a::FermiFS{N,<:Any,S}) where {N,S} = FermiOccupiedModes{N,S}(a.bs)

function near_uniform(::Type{FermiFS{N,M}}) where {N,M}
    return FermiFS([fill(1, N); fill(0, M - N)])
end

@inline function onr(a::FermiFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    @inbounds for (_, mode) in occupied_modes(a)
        result[mode] = 1
    end
    return SVector(result)
end

find_mode(a::FermiFS, i) = fermi_find_mode(a.bs, i)

function find_occupied_mode(a::FermiFS, i::Integer)
    for k in occupied_modes(a)
        i -= 1
        i == 0 && return k
    end
    return FermiFSIndex(0, 0, 0)
end

function Base.reverse(f::FermiFS)
    return typeof(f)(reverse(f.bs))
end

function excitation(a::FermiFS{N,M,S}, creations, destructions) where {N,M,S}
    new_bs, value = fermi_excitation(a.bs, creations, destructions)
    return FermiFS{N,M,S}(new_bs), value
end
