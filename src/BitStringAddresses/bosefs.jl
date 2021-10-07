"""
    BoseFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` spinless bosons in `M` modes
by wrapping a bitstring of type `S <: BitString`.

# Constructors

* `BoseFS{N,M}(bs::BitString)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`.

* `BoseFS(::BitString)`: Automatically determine `N` and `M`. This constructor is not type
  stable!

* `BoseFS{[N,M,S]}(onr)`: Create `BoseFS{N,M}` from [`onr`](@ref) representation. This is
  efficient as long as at least `N` is provided.

See also: [`SingleComponentFockAddress`](@ref), [`FermiFS`](@ref), [`BitString`](@ref).
"""
struct BoseFS{N,M,S<:BitString} <: SingleComponentFockAddress{N,M}
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

"""
    near_uniform_onr(N, M) -> onr::SVector{M,Int}

Create occupation number representation `onr` distributing `N` particles in `M`
modes in a close-to-uniform fashion with each mode filled with at least
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
BoseFS{7,5}((2, 2, 1, 1, 1))

julia> near_uniform(BoseFS{7,5})
BoseFS{7,5}((2, 2, 1, 1, 1))
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
address `bs` as an `SVector{M,Int32}`, where `M` is the number of modes.
"""
onr(bba::BoseFS) = SVector(m_onr(bba))

"""
    m_onr(bs)

Compute and return the occupation number representation of the bit string
address `bs` as an `MVector{M,Int32}`, where `M` is the number of modes.
"""
@inline m_onr(bba::BoseFS) = m_onr(Val(num_chunks(bba.bs)), bba)

# Version specialized for single-chunk addresses.
@inline function m_onr(::Val{1}, bba::BoseFS{N,M}) where {N,M}
    result = zeros(MVector{M,Int32})
    address = bba.bs
    for mode in 1:M
        bosons = Int32(trailing_ones(address))
        @inbounds result[mode] = bosons
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
    mode = 1
    i = K
    while true
        chunk = chunks(address)[i]
        bits_left = chunk_bits(address, i)
        while !iszero(chunk)
            bosons = trailing_ones(chunk)
            @inbounds result[mode] += unsafe_trunc(Int32, bosons)
            chunk >>>= bosons % UInt
            empty_modes = trailing_zeros(chunk)
            mode += empty_modes
            chunk >>>= empty_modes % UInt
            bits_left -= bosons + empty_modes
        end
        i == 1 && break
        i -= 1
        mode += bits_left
    end
    return result
end

function num_occupied_modes(b::BoseFS{<:Any,<:Any,S}) where S
    return num_occupied_modes(Val(num_chunks(S)), b)
end
# For vacuum state
function num_occupied_modes(b::BoseFS{0})
    return 0
end

@inline function num_occupied_modes(::Val{1}, b::BoseFS)
    chunk = b.bs.chunks[1]
    result = 0
    while true
        chunk >>= (trailing_zeros(chunk) % UInt)
        chunk >>= (trailing_ones(chunk) % UInt)
        result += 1
        iszero(chunk) && break
    end
    return result
end

@inline function num_occupied_modes(_, b::BoseFS)
    # This version is faster than using the occupied_mode iterator
    address = b.bs
    result = 0
    K = num_chunks(address)
    last_mask = UInt64(1) << 63 # = 0b100000...
    prev_top_bit = false
    # This loop compiles away for address<:BSAdd*
    for i in K:-1:1
        chunk = chunks(address)[i]
        # This part handles modes that span across chunk boundaries.
        # If the previous top bit and the current bottom bit are both 1, we have to subtract
        # 1 from the result or the mode will be counted twice.
        result -= (chunk & prev_top_bit) % Int
        prev_top_bit = (chunk & last_mask) > 0
        while !iszero(chunk)
            chunk >>>= trailing_zeros(chunk)
            chunk >>>= trailing_ones(chunk)
            result += 1
        end
    end
    return result
end


"""
    BoseFSIndex

Convenience struct for indexing into a [`BoseFS`](@ref).

## Fields:

* `occnum`: the occupation number.
* `mode`: the index of the mode.
* `offset`: the bit offset of the mode.
"""
struct BoseFSIndex<:FieldVector{3,Int}
    occnum::Int
    mode::Int
    offset::Int
end

function Base.show(io::IO, i::BoseFSIndex)
    @unpack occnum, mode, offset = i
    print(io, "BoseFSIndex(occnum=$occnum, mode=$mode, offset=$offset)")
end
Base.show(io::IO, ::MIME"text/plain", i::BoseFSIndex) = show(io, i)

"""
    BoseOccupiedModes{C,S<:BoseFS}
Iterator for occupied modes. `C` is the number of chunks. See [`occupied_modes`](@ref).
"""
struct BoseOccupiedModes{C,S<:BoseFS}
    address::S
end

function occupied_modes(b::BoseFS{<:Any,<:Any,S}) where {S}
    return BoseOccupiedModes{num_chunks(S),typeof(b)}(b)
end

function is_occupied(::BoseFS, i::BoseFSIndex)
    return i.occnum > 0
end

Base.length(o::BoseOccupiedModes) = num_occupied_modes(o.address)
Base.eltype(::BoseOccupiedModes) = BoseFSIndex

# Single chunk versions are simpler.
@inline function Base.iterate(osi::BoseOccupiedModes{1})
    chunk = osi.address.bs.chunks[1]
    empty_modes = trailing_zeros(chunk)
    return iterate(
        osi, (chunk >> (empty_modes % UInt), empty_modes, 1 + empty_modes)
    )
end
@inline function Base.iterate(osi::BoseOccupiedModes{1}, (chunk, bit, mode))
    if iszero(chunk)
        return nothing
    else
        bosons = trailing_ones(chunk)
        chunk >>>= (bosons % UInt)
        empty_modes = trailing_zeros(chunk)
        chunk >>>= (empty_modes % UInt)
        next_bit = bit + bosons + empty_modes
        next_mode = mode + empty_modes
        return BoseFSIndex(bosons, mode, bit), (chunk, next_bit, next_mode)
    end
end

# Multi-chunk version
@inline function Base.iterate(osi::BoseOccupiedModes)
    bitstring = osi.address.bs
    i = num_chunks(bitstring)
    chunk = chunks(bitstring)[i]
    bits_left = chunk_bits(bitstring, i)
    mode = 1
    return iterate(osi, (i, chunk, bits_left, mode))
end
@inline function Base.iterate(osi::BoseOccupiedModes, (i, chunk, bits_left, mode))
    i < 1 && return nothing
    bitstring = osi.address.bs
    S = typeof(bitstring)
    bit_position = 0

    # Remove and count trailing zeros.
    empty_modes = min(trailing_zeros(chunk), bits_left)
    chunk >>>= empty_modes % UInt
    bits_left -= empty_modes
    mode += empty_modes
    while bits_left < 1
        i -= 1
        i < 1 && return nothing
        @inbounds chunk = chunks(bitstring)[i]
        bits_left = chunk_bits(S, i)
        empty_modes = min(bits_left, trailing_zeros(chunk))
        mode += empty_modes
        bits_left -= empty_modes
        chunk >>>= empty_modes % UInt
    end

    bit_position = chunk_bits(S, i) - bits_left + 64 * (num_chunks(bitstring) - i)

    # Remove and count trailing ones.
    result = 0
    bosons = trailing_ones(chunk)
    bits_left -= bosons
    chunk >>>= bosons % UInt
    result += bosons
    while bits_left < 1
        i -= 1
        i < 1 && break
        @inbounds chunk = chunks(bitstring)[i]
        bits_left = chunk_bits(S, i)

        bosons = trailing_ones(chunk)
        bits_left -= bosons
        result += bosons
        chunk >>>= bosons % UInt
    end
    return BoseFSIndex(result, mode, bit_position), (i, chunk, bits_left, mode)
end

function find_mode(b::BoseFS, index)
    last_occnum = last_mode = last_offset = 0
    for (occnum, mode, offset) in occupied_modes(b)
        dist = index - mode
        if dist == 0
            return BoseFSIndex(occnum, index, offset)
        elseif dist < 0
            return BoseFSIndex(0, index, offset + dist)
        end
        last_occnum = occnum
        last_mode = mode
        last_offset = offset
    end
    offset = last_offset + last_occnum + index - last_mode
    return BoseFSIndex(0, index, offset)
end
# Multiple in a single pass
function find_mode(b::BoseFS, indices::NTuple{N}) where {N}
    # Idea: find permutation, then use the permutation to find indices in order even though
    # they are not sorted.
    perm = sortperm(SVector(indices))
    # perm_i is the index in permutation and goes from 1:N.
    perm_i = 1
    # curr_i points to indices and result
    curr_i = perm[1]
    # index is the current index we are looking for.
    index = indices[curr_i]

    result = ntuple(_ -> BoseFSIndex(0, 0, 0), Val(N))
    last_occnum = last_mode = last_offset = 0
    @inbounds for (occnum, mode, offset) in occupied_modes(b)
        dist = index - mode
        # While loop handles duplicate entries in indices.
        while dist ≤ 0
            if dist == 0
                @set! result[curr_i] = BoseFSIndex(occnum, mode, offset)
            else
                @set! result[curr_i] = BoseFSIndex(0, index, offset + dist)
            end
            perm_i += 1
            perm_i > N && return result
            curr_i = perm[perm_i]
            index = indices[curr_i]
            dist = index - mode
        end
        last_occnum = occnum
        last_mode = mode
        last_offset = offset
    end
    # Now we have to find all indices that appear after the last occupied site.
    # While true because we break out of the loop early anyway.
    @inbounds while true
        offset = last_offset + last_occnum + index - last_mode
        @set! result[curr_i] = BoseFSIndex(0, index, offset)
        perm_i += 1
        perm_i > N && return result
        curr_i = perm[perm_i]
        index = indices[curr_i]
    end
    return result # not reached
end

function find_occupied_mode(b::BoseFS, index::Integer, n=1)
    for (occnum, mode, offset) in occupied_modes(b)
        index -= occnum ≥ n
        if index == 0
            return BoseFSIndex(occnum, mode, offset)
        end
    end
    return BoseFSIndex(0, 0, 0)
end
# Find multiple in a single pass.
function find_occupied_mode(b::BoseFS, indices::NTuple{N}, n=1) where {N}
    # Idea is similar to find_mode, i.e. find permutation, then use the permutation to find
    # indices
    perm = sortperm(SVector(indices))
    # Index into permutation, is in 1:N
    perm_i = 1
    # Points to result and indices
    curr_i = perm[1]
    # Current occupied mode index.
    index = 0
    result = ntuple(_ -> BoseFSIndex(0, 0, 0), Val(N))
    @inbounds for (occnum, mode, offset) in occupied_modes(b)
        index += occnum ≥ n
        # While loop handles duplicates in indices
        while index == indices[curr_i]
            @set! result[curr_i] = BoseFSIndex(occnum, mode, offset)
            perm_i += 1
            perm_i > N && return result
            curr_i = perm[perm_i]
        end
    end
    return result # not reached if all modes asked for exist
end

function move_particle(b::BoseFS, from::BoseFSIndex, to::BoseFSIndex)
    occ1 = from.occnum
    occ2 = to.occnum
    if from == to
        return b, Float64(occ1)
    else
        return _move_particle(b, from.offset, to.offset), √(occ1 * (occ2 + 1))
    end
end
function _move_particle(b::BoseFS, from, to)
    if to == from
        return b
    elseif to < from
        return typeof(b)(partial_left_shift(b.bs, to, from))
    else
        return typeof(b)(partial_right_shift(b.bs, from, to - 1))
    end
end

###
### Multiple excitation stuff
###
# Fix offsets that changed after performing a move.
@inline function _fix_offset(pair, index::BoseFSIndex)
    fst, snd = pair[1], pair[2]
    if fst.offset < snd.offset
        return @set index.offset += fst.offset < index.offset ≤ snd.offset
    else
        return @set index.offset -= fst.offset > index.offset > snd.offset
    end
end
_fix_offset(pair) = Base.Fix1(_fix_offset, pair)

# Move multiple particles. This does not care about values, so it performs moves in an
# arbitrary order (from left to right in pairs).
@inline function _move_particles(b::BoseFS, (c,)::NTuple{1}, (d,)::NTuple{1})
    return _move_particle(b, d.offset, c.offset)
end
@inline function _move_particles(b::BoseFS, (c, cs...), (d, ds...))
    b = _move_particle(b, d.offset, c.offset)
    fix = _fix_offset(c => d)
    b = _move_particles(b, map(fix, cs), map(fix, ds))
    return b
end

# Apply destruction operator to BoseFSIndex.
@inline _destroy(d, index) = @set index.occnum -= (d.mode == index.mode)
@inline _destroy(d) = Base.Fix1(_destroy, d)
# Apply creation operator to BoseFSIndex.
@inline _create(c, index) = @set index.occnum += (c.mode == index.mode)
@inline _create(c) = Base.Fix1(_create, c)

# Compute the value of an excitation. Starts by applying all destruction operators, and
# then applying all creation operators. The operators must be given in reverse order.
# Will return 0 if move is illegal.
@inline _compute_value(::Tuple{}, ::Tuple{}) = 1
@inline function _compute_value((c, cs...), ::Tuple{})
    return _compute_value(map(_create(c), cs), ()) * (c.occnum + 1)
end
@inline function _compute_value(creations, (d, ds...))
    return _compute_value(map(_destroy(d), creations), map(_destroy(d), ds)) * d.occnum
end

function excitation(b::BoseFS, creations::NTuple{N}, destructions::NTuple{N}) where N
    # We start by computing the value. This is where the check if the move is even legal
    # is done.
    creations_rev = reverse(creations)
    value = _compute_value(creations_rev, reverse(destructions))
    if iszero(value)
        return b, 0.0
    else
        # Now that we know the value and that the move is legal, we can apply the moves
        # without worrying about doing something weird.
        return _move_particles(b, creations_rev, destructions), √value
    end
end

function find_occupied_modes_with_offsets(
    b::BoseFS, indices::NTuple{N}, offsets::NTuple{N}
) where {N}
    i = 0
    left = N
    result_idx = ntuple(_ -> BoseFSIndex(0, 0, 0), Val(N))
    for boseindex in occupied_modes(b)
        i += 1
        for j in 1:N
            if indices[j] == i
                @set! result_idx[j] = boseindex
                left -= 1
            end
        end
        left == 0 && break
    end
    left ≠ 0 && error("Some indices not found")

    left = N
    result_off = ntuple(_ -> BoseFSIndex(0, 0, 0), Val(N))
    target_modes = getindex.(result_idx, 2) .+ offsets
    last_occnum = last_mode = last_offset = 0
    for (occnum, mode, offset) in occupied_modes(b)
        dists = target_modes .- mode
        for j in 1:N
            !iszero(result_off[j]) && continue
            if dists[j] == 0
                @set! result_off[j] = BoseFSIndex(occnum, mode, offset)
                left -= 1
            elseif dists[j] < 0
                @set! result_off[j] = BoseFSIndex(0, target_modes[j], offset + dists[j])
                left -= 1
            end
        end
        left == 0 && break
    end
    left ≠ 0 && error("Some offsets not found")
    return result_idx, result_off
end
