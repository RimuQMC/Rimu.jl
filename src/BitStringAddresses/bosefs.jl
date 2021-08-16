"""
    BoseFS{N,M,S} <: AbstractFockAddress

Address type that represents a Fock state of `N` spinless bosons in `M` orbitals
by wrapping a bitstring of type `S <: BitString`.

# Constructors

* `BoseFS{N,M}(bs::BitString)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`.

* `BoseFS(::BitString)`: Automatically determine `N` and `M`. This constructor is not type
  stable!

* `BoseFS{[N,M,S]}(onr)`: Create `BoseFS{N,M}` from [`onr`](@ref) representation. This is
  efficient as long as at least `N` is provided.

See also: [`FermiFS`](@ref), [`BitString`](@ref).
"""
struct BoseFS{N,M,S<:BitString} <: AbstractFockAddress
    bs::S
end

function BoseFS{N,M}(bs::BitString{B}) where {N,M,B}
    # Check for consistency between parameter, but NOT for the correct number of bits.
    N + M - 1 == B || throw(ArgumentError("type parameter mismatch"))
    return BoseFS{N,M,typeof(bs)}(bs)
end

function BoseFS(bs::BitString{B}) where B
    N = count_ones(bs)
    M = B - N + 1
    return BoseFS{N,M}(bs)
end

@inline function BoseFS{N,M,S}(
    onr::Union{SVector{M},NTuple{M}}
) where {N,M,S<:BitString{<:Any,1}}
    @boundscheck sum(onr) == N || error("invalid ONR")
    T = chunk_type(S)
    result = zero(T)
    for i in M:-1:1
        curr_occnum = T(onr[i])
        result <<= curr_occnum + T(1)
        result |= one(T) << curr_occnum - T(1)
    end
    return BoseFS{N,M,S}(S(SVector(result)))
end

@inline function BoseFS{N,M,S}(onr::Union{SVector{M},NTuple{M}}) where {N,M,S<:BitString}
    @boundscheck sum(onr) == N || error("invalid ONR")
    K = num_chunks(S)
    result = zeros(MVector{K,UInt64})
    offset = 0
    bits_left = chunk_bits(S, K)
    i = 1
    j = K
    while true
        # Write number to result
        curr_occnum = onr[i]
        while curr_occnum > 0
            x = min(curr_occnum, bits_left)
            mask = (one(UInt64) << x - 1) << offset
            @inbounds result[j] |= mask
            bits_left -= x
            offset += x
            curr_occnum -= x

            if bits_left == 0
                j -= 1
                offset = 0
                bits_left = chunk_bits(S, j)
            end
        end
        offset += 1
        bits_left -= 1

        if bits_left == 0
            j -= 1
            offset = 0
            bits_left = chunk_bits(S, j)
        end
        i += 1
        i > M && break
    end
    return BoseFS{N,M}(S(SVector(result)))
end
function BoseFS{N,M}(onr::Union{AbstractVector,Tuple}) where {N,M}
    S = typeof(BitString{N + M - 1}(0))
    return BoseFS{N,M,S}(SVector{M}(onr))
end
function BoseFS{N}(onr::Union{SVector{M},NTuple{M}}) where {N,M}
    return BoseFS{N,M}(onr)
end
function BoseFS(onr::Union{AbstractVector,Tuple})
    M = length(onr)
    N = sum(onr)
    return BoseFS{N,M}(onr)
end

function Base.show(io::IO, b::BoseFS{N,M,S}) where {N,M,S}
    print(io, "BoseFS{$N,$M}(", tuple(onr(b)...), ")")
end
Base.bitstring(b::BoseFS) = bitstring(b.bs)

Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.:(==)(a::BoseFS, b::BoseFS) = a.bs == b.bs
num_particles(::Type{BoseFS{N,M,S}}) where {N,M,S} = N
num_modes(::Type{BoseFS{N,M,S}}) where {N,M,S} = M

"""
    near_uniform_onr(N, M) -> onr::SVector{M,Int}

Create occupation number representation `onr` distributing `N` particles in `M`
modes in a close-to-uniform fashion with each orbital filled with at least
`N ÷ M` particles and at most with `N ÷ M + 1` particles.
"""
function near_uniform_onr(n::Number, m::Number)
    return near_uniform_onr(Val(n),Val(m))
end
function near_uniform_onr(::Val{N}, ::Val{M}) where {N, M}
    fillingfactor, extras = divrem(N, M)
    # startonr = fill(fillingfactor,M)
    startonr = fillingfactor * @MVector ones(Int,M)
    startonr[1:extras] .+= 1
    return SVector{M}(startonr)
end

"""
    near_uniform(BoseFS{N,M})
    near_uniform(BoseFS{N,M,S}) -> bfs::BoseFS{N,M,S}

Create bosonic Fock state with near uniform occupation number of `M` modes with
a total of `N` particles. Specifying the bit address type `S` is optional.

# Examples
```jldoctest
julia> near_uniform(BoseFS{7,5,BitString{14}})
BoseFS((2,2,1,1,1))

julia> near_uniform(BoseFS{7,5})
BoseFS((2,2,1,1,1))
```
"""
function near_uniform(::Type{<:BoseFS{N,M}}) where {N,M}
    return BoseFS{N,M}(near_uniform_onr(Val(N),Val(M)))
end
near_uniform(b::AbstractFockAddress) = near_uniform(typeof(b))

@deprecate nearUniform near_uniform

"""
    onr(bs)

Compute and return the occupation number representation of the bit string
address `bs` as an `SVector{M,Int32}`, where `M` is the number of orbitals.
"""
onr(bba::BoseFS) = SVector(m_onr(bba))

"""
    m_onr(bs)

Compute and return the occupation number representation of the bit string
address `bs` as an `MVector{M,Int32}`, where `M` is the number of orbitals.
"""
@inline m_onr(bba::BoseFS) = m_onr(Val(num_chunks(bba.bs)), bba)

# Version specialized for single-chunk addresses.
@inline function m_onr(::Val{1}, bba::BoseFS{N,M}) where {N,M}
    result = zeros(MVector{M,Int32})
    address = bba.bs
    for orbital in 1:M
        bosons = Int32(trailing_ones(address))
        @inbounds result[orbital] = bosons
        address >>>= (bosons + 1) % UInt
        iszero(address) && break
    end
    return result
end

# Version specialized for multi-chunk addresses. This is quite a bit faster for large
# addresses.
@inline function m_onr(::Val{K}, bba::BoseFS{N,M}) where {K,N,M}
    B = num_bits(bba.bs)
    result = zeros(MVector{M,Int32})
    address = bba.bs
    orbital = 1
    i = K
    while true
        chunk = chunks(address)[i]
        bits_left = chunk_bits(address, i)
        while !iszero(chunk)
            bosons = trailing_ones(chunk)
            @inbounds result[orbital] += unsafe_trunc(Int32, bosons)
            chunk >>>= bosons % UInt
            empty_modes = trailing_zeros(chunk)
            orbital += empty_modes
            chunk >>>= empty_modes % UInt
            bits_left -= bosons + empty_modes
        end
        i == 1 && break
        i -= 1
        orbital += bits_left
    end
    return result
end

"""
    BoseFSIndex

Convenience struct for indexing into a fock state.

## Fields:

* `occnum`: the occupation number.
* `site`: the index of the site.
* `offset`: the bit offset of the site.
"""
struct BoseFSIndex <: FieldVector{3,Int}
    occnum::Int
    site::Int
    offset::Int
end

struct OccupiedOrbitalIterator{C,S}
    address::S
end

"""
    occupied_orbitals(b::BoseFS)

Iterate over occupied orbitals in `BoseFS` address. Iterates values of type
[`BoseFSIndex`](@ref).

# Example

```jldoctest
julia> b = BoseFS((1,5,0,4))
julia> for (n, i, o) in occupied_orbitals(b)
    @show n, i, o
end
(n, i, o) = (1, 1, 0)
(n, i, o) = (5, 2, 2)
(n, i, o) = (4, 4, 9)
```
"""
function occupied_orbitals(b::BoseFS{<:Any,<:Any,S}) where {S}
    return OccupiedOrbitalIterator{num_chunks(S),S}(b.bs)
end

# Single chunk versions are simpler.
@inline function Base.iterate(osi::OccupiedOrbitalIterator{1})
    chunk = osi.address.chunks[1]
    empty_orbitals = trailing_zeros(chunk)
    return iterate(
        osi, (chunk >> (empty_orbitals % UInt), empty_orbitals, 1 + empty_orbitals)
    )
end
@inline function Base.iterate(osi::OccupiedOrbitalIterator{1}, (chunk, bit, orbital))
    if iszero(chunk)
        return nothing
    else
        bosons = trailing_ones(chunk)
        chunk >>>= (bosons % UInt)
        empty_orbitals = trailing_zeros(chunk)
        chunk >>>= (empty_orbitals % UInt)
        next_bit = bit + bosons + empty_orbitals
        next_orbital = orbital + empty_orbitals
        return BoseFSIndex(bosons, orbital, bit), (chunk, next_bit, next_orbital)
    end
end

# Multi-chunk version
@inline function Base.iterate(osi::OccupiedOrbitalIterator)
    address = osi.address
    i = num_chunks(address)
    chunk = chunks(address)[i]
    bits_left = chunk_bits(address, i)
    orbital = 1
    return iterate(osi, (i, chunk, bits_left, orbital))
end
@inline function Base.iterate(osi::OccupiedOrbitalIterator, (i, chunk, bits_left, orbital))
    i < 1 && return nothing
    address = osi.address
    S = typeof(address)
    bit_position = 0

    # Remove and count trailing zeros.
    empty_orbitals = min(trailing_zeros(chunk), bits_left)
    chunk >>>= empty_orbitals % UInt
    bits_left -= empty_orbitals
    orbital += empty_orbitals
    while bits_left < 1
        i -= 1
        i < 1 && return nothing
        @inbounds chunk = chunks(address)[i]
        bits_left = chunk_bits(S, i)
        empty_orbitals = min(bits_left, trailing_zeros(chunk))
        orbital += empty_orbitals
        bits_left -= empty_orbitals
        chunk >>>= empty_orbitals % UInt
    end

    bit_position = chunk_bits(S, i) - bits_left + 64 * (num_chunks(address) - i)

    # Remove and count trailing ones.
    result = 0
    bosons = trailing_ones(chunk)
    bits_left -= bosons
    chunk >>>= bosons % UInt
    result += bosons
    while bits_left < 1
        i -= 1
        i < 1 && break
        @inbounds chunk = chunks(address)[i]
        bits_left = chunk_bits(S, i)

        bosons = trailing_ones(chunk)
        bits_left -= bosons
        result += bosons
        chunk >>>= bosons % UInt
    end
    return BoseFSIndex(result, orbital, bit_position), (i, chunk, bits_left, orbital)
end

function find_site(b::BoseFS, index)
    last_occnum = last_site = last_offset = 0
    for (occnum, site, offset) in occupied_orbitals(b)
        dist = index - site
        if dist == 0
            return BoseFSIndex(occnum, index, offset)
        elseif dist < 0
            return BoseFSIndex(0, index, offset + dist)
        end
        last_occnum = occnum
        last_site = site
        last_offset = offset
    end
    offset = last_offset + last_occnum + index - last_site
    return BoseFSIndex(0, index, offset)
end

function find_particle(b::BoseFS, index, n=1)
    for (occnum, site, offset) in occupied_orbitals(b)
        index -= occnum ≥ n
        if index == 0
            return BoseFSIndex(occnum, site, offset)
        end
    end
    return BoseFSIndex(0, 0, 0)
end

"""
    move_particle(b::BoseFS, from::BoseFSIndex, to::BoseFSIndex)

Move particle from [`BoseFSIndex`](@ref) `from` to the [`BoseFSIndex`](@ref) `to`.

This is equivalent to applying a destruction operator followed by a creation operator to the
address.

Return the new Fock state and the product of the occupation numbers.
"""
function move_particle(b::BoseFS, from::BoseFSIndex, to::BoseFSIndex)
    occ1 = from.occnum
    occ2 = to.occnum
    if from == to
        return b, occ1 * (occ1 - 1)
    else
        return _move_particle(b, from.offset, to.offset), occ1 * (occ2 + 1)
    end
end

function _move_particle(b::BoseFS, from, to)
    if to < from
        return typeof(b)(partial_left_shift(b.bs, to, from))
    else
        return typeof(b)(partial_right_shift(b.bs, from, to - 1))
    end
end
