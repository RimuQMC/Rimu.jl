"""
    HardcoreBoseFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` fermions of the same spin in `M` modes by
wrapping a [`BitString`](@ref), or a [`SortedParticleList`](@ref). Which is wrapped is
chosen automatically based on the properties of the address.

# Constructors

* `HardcoreBoseFS{[N,M]}(val::Integer...)`: Create `HardcoreBoseFS{N,M}` from occupation numbers. This is
  type-stable if the number of modes `M` and the number of particles `N` are provided.
  Otherwise, `M` and `N` are inferred from the arguments.

* `HardcoreBoseFS{[N,M]}(onr)`: Create `HardcoreBoseFS{N,M}`  from occupation number representation, see
  [`onr`](@ref). This is efficient if `N` and `M` are provided, and `onr` is a
  statically-sized collection, such as a `Tuple{M}` or `SVector{M}`.

* `HardcoreBoseFS{[N,M]}([M, ]pairs...)`: Provide the number of modes `M` and pairs of the form
  `mode => 1`. If `M` is provided as a type parameter, it should not be provided as the
  first argument.  Useful for creating sparse addresses. `pairs` can be multiple arguments
  or an iterator of pairs.

* `HardcoreBoseFS{N,M,S}(bs::S)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`, or whether each mode only contains one particle.

* [`@fs_str`](@ref): Addresses are sometimes printed in a compact manner. This
  representation can also be used as a constructor. See the examples below.

# Examples

```jldoctest
julia> HardcoreBoseFS{3,5}(0, 1, 1, 1, 0)
HardcoreBoseFS{3,5}(0, 1, 1, 1, 0)

julia> HardcoreBoseFS([abs(i - 3) ≤ 1 for i in 1:5])
HardcoreBoseFS{3,5}(0, 1, 1, 1, 0)

julia> HardcoreBoseFS(5, 2 => 1, 3 => 1, 4 => 1)
HardcoreBoseFS{3,5}(0, 1, 1, 1, 0)

julia> HardcoreBoseFS{3,5}(i => 1 for i in 2:4)
HardcoreBoseFS{3,5}(0, 1, 1, 1, 0)

julia> fs"|⋅↑↑↑⋅⟩"
HardcoreBoseFS{3,5}(0, 1, 1, 1, 0)

julia> fs"|f 5: 2 3 4⟩"
HardcoreBoseFS{3,5}(0, 1, 1, 1, 0)
```

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`CompositeFS`](@ref),
[`HardcoreBoseFS2C`](@ref), [`BitString`](@ref), [`OccupationNumberFS`](@ref).
"""
struct HardcoreBoseFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S
end

function HardcoreBoseFS{N,M,S}(onr::Union{SVector{M},MVector{M},NTuple{M}}) where {N,M,S}
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
    return HardcoreBoseFS{N,M,S}(from_fermi_onr(S, onr))
end
function HardcoreBoseFS{N,M}(onr::Union{AbstractArray{<:Integer},NTuple{M,<:Integer}}) where {N,M}
    @boundscheck check_fermi_onr(onr, N, M)
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
    return HardcoreBoseFS{N,M,S}(from_fermi_onr(S, onr))
end
function HardcoreBoseFS(onr)
    onr = Tuple(onr)
    M = length(onr)
    N = sum(onr)
    return HardcoreBoseFS{N,M}(onr)
end
HardcoreBoseFS(vals::Integer...) = HardcoreBoseFS(vals) # list occupation numbers
HardcoreBoseFS(val::Integer) = HardcoreBoseFS((val,)) # single mode
HardcoreBoseFS{N,M}(vals::Integer...) where {N,M} = HardcoreBoseFS{N,M}(vals)

# Sparse constructors
HardcoreBoseFS(M::Integer, pairs::Pair...) = HardcoreBoseFS(M, pairs)
HardcoreBoseFS(M::Integer, pairs) = HardcoreBoseFS(sparse_to_onr(M, pairs))
HardcoreBoseFS{N,M}(pairs::Vararg{Pair,N}) where {N,M} = HardcoreBoseFS{N,M}(pairs)
HardcoreBoseFS{N,M}(pairs) where {N,M} = HardcoreBoseFS{N,M}(sparse_to_onr(M, pairs))
HardcoreBoseFS(pairs::Pair...) = throw(ArgumentError("number of modes must be provided"))

function print_address(io::IO, f::HardcoreBoseFS{N,M}; compact=false) where {N,M}
    if compact && f.bs isa SortedParticleList
        print(io, "|f ", M, ": ", join(Int.(f.bs.storage), ' '), "⟩")
    elseif compact
        print(io, "|", join(map(o -> o == 0 ? '⋅' : '↑', onr(f))), "⟩")
    elseif f.bs isa SortedParticleList
        print(io, "HardcoreBoseFS{$N,$M}(", onr_sparse_string(onr(f)), ")")
    else
        print(io, "HardcoreBoseFS{$N,$M}", tuple(onr(f)...))
    end
end

Base.bitstring(a::HardcoreBoseFS) = bitstring(a.bs)
Base.isless(a::HardcoreBoseFS, b::HardcoreBoseFS) = isless(a.bs, b.bs)
Base.hash(a::HardcoreBoseFS,  h::UInt) = hash(a.bs, h)
Base.:(==)(a::HardcoreBoseFS, b::HardcoreBoseFS) = a.bs == b.bs
num_occupied_modes(::HardcoreBoseFS{N}) where {N} = N
occupied_modes(a::HardcoreBoseFS{N,<:Any,S}) where {N,S} = FermiOccupiedModes{N,S}(a.bs)

num_unoccupied_modes(::HardcoreBoseFS{N,M}) where {N,M} = M - N
unoccupied_modes(a::HardcoreBoseFS{N,M,S}) where {N,M,S} = FermiUnoccupiedModes{M - N,S}(a.bs)

"""
    unoccupied_mode_map(addr::HardcoreBoseFS) <: AbstractVector

Get a map of unoccupied modes in [`HardcoreBoseFS`](@ref) address as an `AbstractVector`
of indices compatible with [`excitation`](@ref).

`unoccupied_mode_map(addr)[i]` contains the index for the `i`-th unoccupied mode.
This is useful because unoccupied modes is required in some cases.
`unoccupied_mode_map(addr)` is an eager version of the iterator returned by
[`unoccupied_modes`](@ref). It is similar to [`onr`](@ref) but contains more information.

Note that this function is only implemented for addresses of type [`HardcoreBoseFS`](@ref).

# Example

```jldoctest
julia> f = HardcoreBoseFS(1,1,0,0)
HardcoreBoseFS{2,4}(1, 1, 0, 0)

julia> mf = unoccupied_mode_map(f)
2-element Rimu.BitStringAddresses.ModeMap{2, FermiFSIndex}:
 FermiFSIndex(occnum=0, mode=3, offset=2)
 FermiFSIndex(occnum=0, mode=4, offset=3)

julia> mf == collect(unoccupied_modes(f))
true

```
See also [`occupied_mode_map`](@ref).
"""
function unoccupied_mode_map(addr::HardcoreBoseFS{N,M}) where {N,M}
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

function near_uniform(::Type{HardcoreBoseFS{N,M}}) where {N,M}
    return HardcoreBoseFS([fill(1, N); fill(0, M - N)])
end

find_mode(a::HardcoreBoseFS, i, occ=nothing) = fermi_find_mode(a.bs, i)

@inline function onr(a::HardcoreBoseFS{<:Any,M}) where {M}
    result = zero(MVector{M,Int32})
    @inbounds for (_, mode) in occupied_modes(a)
        result[mode] = 1
    end
    return SVector(result)
end

function Base.reverse(f::HardcoreBoseFS)
    return typeof(f)(reverse(f.bs))
end

function excitation(a::HardcoreBoseFS{N,M,S}, creations, destructions) where {N,M,S}
    new_bs, value = fermi_excitation(a.bs, creations, destructions)
    return HardcoreBoseFS{N,M,S}(new_bs), abs(value)
end
