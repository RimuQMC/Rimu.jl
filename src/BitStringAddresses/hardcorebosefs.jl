"""
    HardcoreBoseFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` hardcore bosons in `M` modes, with
occupancies restricted to 0 or 1 per mode, by wrapping a [`BitString`](@ref) or a
[`SortedParticleList`](@ref). Which is wrapped is chosen automatically based on the
properties of the address. Set `N` to `missing` if the number of particles is not known at
compile time and can be changed by excitations.

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

See the examples below.

# Examples

```jldoctest
julia> HardcoreBoseFS(1, 1, 1, 0, 0)
HardcoreBoseFS{3,5}(1, 1, 1, 0, 0)

julia> HardcoreBoseFS([abs(i - 2) ≤ 1 for i in 1:5])
HardcoreBoseFS{3,5}(1, 1, 1, 0, 0)

julia> HardcoreBoseFS(5, 1 => 1, 2 => 1, 3 => 1)
HardcoreBoseFS{3,5}(1, 1, 1, 0, 0)

julia> HardcoreBoseFS{3,5}(i => 1 for i in 1:3)
HardcoreBoseFS{3,5}(1, 1, 1, 0, 0)

julia> fs"|●●●∘∘⟩" # \\mdlgblkcircle(tab) -> ●, \\circ(tab) -> ∘, \\rangle(tab) -> ⟩
HardcoreBoseFS{3,5}(1, 1, 1, 0, 0)

julia> HardcoreBoseFS{missing}(1, 1, 1, 0, 0) == fs"|●●●∘∘⟩{}" # missing particle number
true

julia> fs"|h 5: 1 2 3⟩"
HardcoreBoseFS{3,5}(1, 1, 1, 0, 0)
```

See also: [`SingleComponentFockAddress`](@ref), [`BoseFS`](@ref), [`CompositeFS`](@ref),
[`FermiFS`](@ref), [`BitString`](@ref), [`OccupationNumberFS`](@ref), [`@fs_str`](@ref).
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
    if ismissing(N)
        S = typeof(BitString{M}(0))
        return HardcoreBoseFS{N,M,S}(from_fermi_onr(S, onr))
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
    return HardcoreBoseFS{N,M,S}(from_fermi_onr(S, onr))
end
function HardcoreBoseFS(onr)
    onr = Tuple(onr)
    M = length(onr)
    N = sum(onr)
    return HardcoreBoseFS{N,M}(onr)
end
function HardcoreBoseFS{N}(onr) where {N}
    onr = Tuple(onr)
    M = length(onr)
    return HardcoreBoseFS{N,M}(onr)
end

HardcoreBoseFS(vals::Integer...) = HardcoreBoseFS(vals) # list occupation numbers
HardcoreBoseFS(val::Integer) = HardcoreBoseFS((val,)) # single mode
HardcoreBoseFS{N}(vals::Integer...) where N = HardcoreBoseFS{N}(vals) # list occupation numbers
HardcoreBoseFS{N}(val::Integer) where {N} = HardcoreBoseFS{N}((val,)) # single mode
HardcoreBoseFS{N,M}(vals::Integer...) where {N,M} = HardcoreBoseFS{N,M}(vals)

# Sparse constructors
HardcoreBoseFS(M::Integer, pairs::Pair...) = HardcoreBoseFS(M, pairs)
HardcoreBoseFS{N}(M::Integer, pairs::Pair...) where {N} = HardcoreBoseFS{N}(M, pairs)
HardcoreBoseFS(M::Integer, pairs) = HardcoreBoseFS(sparse_to_onr(M, pairs))
HardcoreBoseFS{N}(M::Integer, pairs) where {N} = HardcoreBoseFS{N}(sparse_to_onr(M, pairs))
HardcoreBoseFS{N,M}(pairs::Vararg{Pair}) where {N,M} = HardcoreBoseFS{N,M}(pairs)
HardcoreBoseFS{N,M}(pairs) where {N,M} = HardcoreBoseFS{N}(sparse_to_onr(M, pairs))
HardcoreBoseFS(pairs::Pair...) = throw(ArgumentError("number of modes must be provided"))

function print_address(io::IO, f::HardcoreBoseFS{N,M}; compact=false) where {N,M}
    if compact && f.bs isa SortedParticleList
        print(io, "|h ", M, ": ", join(Int.(f.bs.storage), ' '), "⟩")
    elseif compact && ismissing(N)
        print(io, "|", join(map(o -> o == 0 ? '∘' : '●', onr(f))), "⟩{}")
    elseif compact
        print(io, "|", join(map(o -> o == 0 ? '∘' : '●', onr(f))), "⟩")
    elseif f.bs isa SortedParticleList
        print(io, "HardcoreBoseFS{$N,$M}(", onr_sparse_string(onr(f)), ")")
    else
        print(io, "HardcoreBoseFS{$N,$M}", tuple(onr(f)...))
    end
end

function excitation(
    a::HardcoreBoseFS{N,M,S}, creations::NTuple{NC}, destructions::NTuple{ND}
) where {N,M,S,NC,ND}
    if NC != ND && !ismissing(N)
        throw(ArgumentError("number of creations and destructions must be equal, got $NC and $ND"))
    end
    new_bs, value = fermi_excitation(a.bs, creations, destructions)
    return HardcoreBoseFS{N,M,S}(new_bs), abs(value) # different from FermiFS, no sign
end

# See "fermifs.jl" for other function definitions for HardcoreBoseFS.
