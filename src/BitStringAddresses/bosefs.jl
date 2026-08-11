"""
    BoseFS{N,M,S} <: SingleComponentFockAddress

Address type that represents a Fock state of `N` spinless bosons in `M` modes. The
particle number `N` can be set to `missing`. In the latter case the number of particles is
not known at compile time and can be changed by excitations.

# Constructors

* `BoseFS{[N,M]}(val::Integer...)`: Create `BoseFS{N,M}` from occupation numbers. This is
  type-stable if the number of modes `M` and the number of particles `N` are provided.
  Otherwise, `M` and `N` are inferred from the arguments.

* `BoseFS{missing}(arg; type=nothing)`: Create `BoseFS{missing,M}` from occupation numbers.
  The number of particles is not known at compile time and can be changed by excitations.
  The keyword argument `type` can be used to specify the type of the occupation numbers. It
  must be an unsigned integer type. If unspecified, the smallest unsigned integer type that
  can hold the maximum occupation number is chosen automatically.

* `BoseFS{[N,M]}(onr)`: Create `BoseFS{N,M}` from occupation number representation, see
  [`onr`](@ref). This is efficient if `N` and `M` are provided, and `onr` is a
  statically-sized collection, such as a `Tuple` or `SVector`.

* `BoseFS{[N,M]}([M, ]pairs...)`: Provide the number of modes `M` and `mode =>
  occupation_number` pairs. If `M` is provided as a type parameter, it should not be
  provided as the first argument.  Useful for creating sparse addresses. `pairs` can be
  multiple arguments or an iterator of pairs.

* `BoseFS{N,M,S}(bs::S)`: Unsafe constructor. Does not check whether the number of
  particles in `bs` is equal to `N`.

* [`@fs_str`](@ref): Addresses are sometimes printed in a compact manner. This
  representation can also be used as a constructor. See the examples below.

# Examples

```jldoctest
julia> address = BoseFS(0, 1, 2, 3, 0)
BoseFS(0, 1, 2, 3, 0)

julia> num_modes(address), num_particles(address)
(5, 6)

julia> BoseFS(abs(i - 3) ≤ 1 ? i - 1 : 0 for i in 1:5)
BoseFS(0, 1, 2, 3, 0)

julia> BoseFS(5, 2 => 1, 3 => 2, 4 => 3) # sparse constructor
BoseFS(0, 1, 2, 3, 0)

julia> BoseFS{6,5}(i => i - 1 for i in 2:4)
BoseFS(0, 1, 2, 3, 0)

julia> fs"|0 1 2 3 0⟩" # \\rangle(tab) -> ⟩
BoseFS(0, 1, 2, 3, 0)

julia> fs"|b 5: 2 3 3 4 4 4⟩" # compact sparse constructor
BoseFS(0, 1, 2, 3, 0)

julia> BoseFS{missing}(0, 1, 2, 3, 0) === fs"|0 1 2 3 0⟩{}" # missing particle number UInt8
true

julia> BoseFS{missing}(0, 1, 2, 3, 0; type=UInt16) === fs"|0 1 2 3 0⟩{UInt16}"
true
```

See also: [`SingleComponentFockAddress`](@ref),
[`FermiFS`](@ref), [`CompositeFS`](@ref), [`FermiFS2C`](@ref), [`@fs_str`](@ref).

# Extended Help

The particle number `N` is a type parameter of the address and will be inferred by default,
unless it is specifically set to `missing`. Having the particle number as a type parameter
allows for type-stable code and optimizations. It should be used by default for
number-conserving Hamiltonians.

Internally, there are three different storage types for `BoseFS`. The type `S` is
chosen automatically based on the properties of the address. The storage types
[`BitString`](@ref) and [`SortedParticleList`](@ref) are used for dense and sparse
representations, respectively, when the number of particles is a type parameter and thus
known at compile time.

When the number of particles is set to `missing` and not known at compile time, the
occupation numbers are stored in a statically-sized vector of type `SVector{M,T}` where
`T` is an unsigned integer type. For a type-stable constructor with missing particle number,
use
```julia
BoseFS{missing}(v::SVector{M,T}) where {M,T<:Unsigned}
```
"""
struct BoseFS{N,M,S} <: SingleComponentFockAddress{N,M}
    bs::S

    function BoseFS{N,M,S}(bs::S) where {N,M,S}
        new{N,M,S}(bs)
    end
    function BoseFS{missing,M,S}(bs::S) where {M,S<:SVector{M,<:Unsigned}}
        new{missing,M,S}(bs)
    end
end

@inline function BoseFS{N,M,S}(onr) where {N,M,S}
    onr isa Union{SVector{M},MVector{M},NTuple{M}} || throw(ArgumentError(
        "invalid occupation number representation: expected SVector{$M}, MVector{$M}, or NTuple{$M}; got $(typeof(onr))"
    ))
    @boundscheck begin
        sum(onr) == N || throw(ArgumentError(
            "invalid ONR: $N particles expected, $(sum(onr)) given"
        ))
        if S <: BitString
            B = num_bits(S)
            M + N - 1 == B || throw(ArgumentError(
                "invalid ONR: $B-bit BitString does not fit $N particles in $M modes"
            ))
        elseif S <: SortedParticleList
            N == num_particles(S) && M == num_modes(S) || throw(ArgumentError(
                "invalid ONR: $S does not fit $N particles in $M modes"
            ))
        end
    end
    return BoseFS{N,M,S}(from_bose_onr(S, onr))
end

function BoseFS{N,M}(onr::Union{AbstractArray{<:Integer},NTuple{M,<:Integer}}) where {N,M}
    @boundscheck begin
        sum(onr) == N || throw(ArgumentError(
            "invalid ONR: $N particles expected, $(sum(onr)) given"
        ))
        length(onr) == M || throw(ArgumentError(
            "invalid ONR: $M modes expected, $(length(onr)) given"
        ))
    end
    spl_type = select_int_type(M)

    # Pick smaller address type, but prefer sparse.
    # Alway pick dense if it fits into one chunk.

    # Compute the size of container in words
    sparse_sizeof = ceil(Int, N * sizeof(spl_type) / 8)
    dense_sizeof = ceil(Int, (N + M - 1) / 64)
    if dense_sizeof == 1 || dense_sizeof < sparse_sizeof
        S = typeof(BitString{M + N - 1}(0))
    else
        S = SortedParticleList{N,M,spl_type}
    end
    return BoseFS{N,M,S}(from_bose_onr(S, onr))
end
function BoseFS(onr) # single argument constructor
    onr = Tuple(onr)
    M = length(onr)
    N = sum(onr)
    return BoseFS{N,M}(onr)
end
BoseFS(vals::Integer...) = BoseFS(vals) # specify occupation numbers
BoseFS(val::Integer) = BoseFS((val,)) # single mode address
BoseFS{N,M}(vals::Integer...) where {N,M} = BoseFS{N,M}(vals)

# Sparse constructors
BoseFS(M::Integer, pairs::Pair...) = BoseFS(M, pairs)
BoseFS(M::Integer, pairs) = BoseFS(sparse_to_onr(M, pairs))
BoseFS{N,M}(pairs::Pair...) where {N,M} = BoseFS{N,M}(pairs)
BoseFS{N,M}(pairs) where {N,M} = BoseFS{N,M}(sparse_to_onr(M, pairs))
BoseFS{N,M}() where {N,M} = BoseFS{N,M}(sparse_to_onr(M, ())) # vacuum state
BoseFS(pairs::Pair...) = throw(ArgumentError("number of modes must be provided"))


function print_address(io::IO, b::BoseFS{N,M}; compact=false) where {N,M}
    if compact && b.bs isa SortedParticleList
        print(io, "|b ", M, ": ", join(Int.(b.bs.storage), ' '), "⟩")
    elseif compact
        print(io, "|", join(onr(b), ' '), "⟩")
    elseif b.bs isa SortedParticleList
        print(io, "BoseFS{$N,$M}(", onr_sparse_string(onr(b)), ")")
    else
        print(io, "BoseFS", tuple(onr(b)...))
    end
end

Base.bitstring(b::BoseFS) = bitstring(b.bs) # TODO rename?

Base.isless(a::BoseFS, b::BoseFS) = isless(a.bs, b.bs)
function Base.isless(a::BoseFS{missing,M}, b::BoseFS{missing,M}) where {M}
    # equivalent to `isless(reverse(a.bs), reverse(b.bs))`
    # reversing the order here to make it consistent with BoseFS
    i = M
    @inbounds while i > 1 && a.bs[i] == b.bs[i]
        i -= 1
    end
    return isless(a.bs[i], b.bs[i])
end

Base.hash(bba::BoseFS,  h::UInt) = hash(bba.bs, h)
Base.:(==)(a::BoseFS, b::BoseFS) = a.bs == b.bs

"""
    near_uniform_onr(::Val{N}, ::Val{M}) -> onr::SVector{M,Int}

Create occupation number representation `onr` distributing `N` particles in `M`
modes in a close-to-uniform fashion with each mode filled with at least
`N ÷ M` particles and at most with `N ÷ M + 1` particles.
"""
function near_uniform_onr(::Val{N}, ::Val{M}) where {N, M}
    fillingfactor, extras = divrem(N, M)
    # startonr = fill(fillingfactor,M)
    startonr = fillingfactor * @MVector ones(Int,M)
    startonr[1:extras] .+= 1
    return SVector{M}(startonr)
end

"""
    near_uniform(T::Type{<:SingleComponentFockAddress{N,M}}) → address::T
    near_uniform(T::Type{<:SingleComponentFockAddress}, N::Integer, M::Integer) → address::T
    near_uniform(address::SingleComponentFockAddress) → address::typeof(address)

Create a single component Fock state with `M` modes and `N` particles with near uniform
occupation numbers.

# Examples
```jldoctest
julia> near_uniform(BoseFS{7,5})
BoseFS(2, 2, 1, 1, 1)

julia> near_uniform(FermiFS{3,5})
FermiFS(1, 1, 1, 0, 0)

julia> near_uniform(HardcoreBoseFS{missing}, 3, 5)
HardcoreBoseFS{missing}(1, 1, 1, 0, 0)

julia> near_uniform(BoseFS(10,0,0,0))
BoseFS(3, 3, 2, 2)
```
"""
function near_uniform(T::Type{<:SingleComponentFockAddress{N,M}}) where {N,M}
    return T(near_uniform_onr(Val(N), Val(M)))
end
function near_uniform(T::Type{<:SingleComponentFockAddress}, N::Integer, M::Integer)
    return near_uniform(T, Val(N), Val(M))
end
function near_uniform(T::Type{<:SingleComponentFockAddress}, ::Val{N}, ::Val{M}) where {N,M}
    return T(near_uniform_onr(Val(N), Val(M)))
end
near_uniform(b::SingleComponentFockAddress) = near_uniform(typeof(b))
function near_uniform(b::SingleComponentFockAddress{missing})
    N = num_particles(b)
    M = num_modes(b)
    return near_uniform(typeof(b), Val(N), Val(M))
end

onr(b::BoseFS{<:Any,M}) where {M} = to_bose_onr(b.bs, Val(M))
const occupation_number_representation = onr # resides here because `onr` has to be defined

function Base.reverse(b::BoseFS)
    return typeof(b)(reverse(b.bs))
end

# For vacuum state
function num_occupied_modes(b::BoseFS{0})
    return 0
end
function num_occupied_modes(b::BoseFS)
    return bose_num_occupied_modes(b.bs)
end
function occupied_modes(b::BoseFS{N,M,S}) where {N,M,S}
    return BoseOccupiedModes{N,M,S}(b.bs)
end

function find_mode(b::BoseFS, index, occ=occupied_modes(b))
    last_occnum = last_mode = last_offset = 0
    for (occnum, mode, offset) in occ
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
function find_mode(b::BoseFS, indices::NTuple{N}, occ=occupied_modes(b)) where {N}
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
    @inbounds for (occnum, mode, offset) in occ
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

# Specialised version of each_mode for iterating modes in BoseFS with BitString storage.
# This is necessary because find_mode in a BitString-backed BoseFS is inefficient.
struct BoseBitStringEachMode{M,A<:BoseFS{<:Any,M,<:BitString}}
    address::A
end
Base.eltype(::BoseBitStringEachMode) = BoseFSIndex
Base.length(::BoseBitStringEachMode{M}) where {M} = M

function Base.iterate(em::BoseBitStringEachMode{M}, state=(0, 1, em.address.bs)) where {M}
    offset, mode, bitstring = state
    if mode > M
        return nothing
    else
        bosons = Int32(trailing_ones(bitstring))
        bitstring >>>= (bosons + 1) % UInt

        return BoseFSIndex(bosons, mode, offset), (offset + bosons + 1, mode + 1, bitstring)
    end
end

function each_mode(addr::BoseFS{<:Any,<:Any,<:BitString})
    return BoseBitStringEachMode(addr)
end

# find_occupied_mode provided by generic implementation

function excitation(b::B, creations::NTuple{C}, destructions::NTuple{C}) where {B<:BoseFS, C}
    new_bs, val = bose_excitation(b.bs, creations, destructions)
    return B(new_bs), val # type doesn't change
end
function excitation(b::BoseFS, c::NTuple{C}, d::NTuple{D}) where {C, D}
    throw(ArgumentError("number of creations and destructions must be equal, got $C and $D"))
end

"""
    new_address, value = hopnextneighbour(add, chosen, boundary_condition)

Compute the new address of a hopping event for the Hubbard model. Returns the new
address and the square root of product of occupation numbers of the involved modes
multiplied by a term consistent with boundary condition as the `value`.
The following boundary conditions are supported:

* `:periodic`: hopping over the boundary gives does not change the `value`.
* `:twisted`: hopping over the boundary flips the sign of the `value`.
* `:hard_wall`: hopping over the boundary gives a `value` of zero.
* `θ <: Number`: hopping over the boundary gives a `value` multiplied by ``\\exp(iθ)`` or ``\\exp(−iθ)`` depending on the direction of hopping.

The off-diagonals are indexed as follows:

* `(chosen + 1) ÷ 2` selects the hopping site.
* Even `chosen` indicates a hop to the left.
* Odd `chosen` indicates a hop to the right.

# Example

```jldoctest
julia> using Rimu.Hamiltonians: hopnextneighbour

julia> hopnextneighbour(BoseFS(1, 0, 1), 3)
(BoseFS(2, 0, 0), 1.4142135623730951)

julia> hopnextneighbour(BoseFS(1, 0, 1), 4)
(BoseFS(1, 1, 0), 1.0)

julia> hopnextneighbour(BoseFS(1, 0, 1), 3, :twisted)
(BoseFS(2, 0, 0), -1.4142135623730951)

julia> hopnextneighbour(BoseFS(1, 0, 1), 3, :hard_wall)
(BoseFS(2, 0, 0), 0.0)

julia> hopnextneighbour(BoseFS(1, 0, 1), 3, π/4)
(BoseFS(2, 0, 0), 1.0000000000000002 + 1.0im)
```
"""
function hopnextneighbour(b::BoseFS{N,M,A}, chosen) where {N,M,A<:BitString}
    address = b.bs
    T = chunk_type(address)
    site = (chosen + 1) >>> 0x1
    if isodd(chosen) # Hopping to the right
        next = 0
        curr = 0
        offset = 0
        sc = 0
        reached_end = false
        for (i, (num, sn, bit)) in enumerate(occupied_modes(b))
            next = num * (sn == sc + 1) # only set next to > 0 if sites are neighbours
            reached_end = i == site + 1
            reached_end && break
            curr = num
            offset = bit + num
            sc = sn
        end
        if sc == M
            new_address = (address << 0x1) | A(T(1))
            prod = curr * (trailing_ones(address) + 1) # mul occupation num of first obital
        else
            next *= reached_end
            new_address = address ⊻ A(T(3)) << ((offset - 1) % T)
            prod = curr * (next + 1)
        end
    else # Hopping to the left
        if site == 1 && isodd(address)
            # For leftmost site, we shift the whole address circularly by one bit.
            new_address = (address >>> 0x1) | A(T(1)) << ((N + M - 2) % T)
            prod = trailing_ones(address) * leading_ones(new_address)
        else
            prev = 0
            curr = 0
            offset = 0
            sp = 0
            for (i, (num, sc, bit)) in enumerate(occupied_modes(b))
                prev = curr * (sc == sp + 1) # only set prev to > 0 if sites are neighbours
                curr = num
                offset = bit
                i == site && break
                sp = sc
            end
            new_address = address ⊻ A(T(3)) << ((offset - 1) % T)
            prod = curr * (prev + 1)
        end
    end
    return BoseFS{N,M,A}(new_address), √prod
end

function hopnextneighbour(b::SingleComponentFockAddress, i)
    src = find_occupied_mode(b, (i + 1) >>> 0x1)
    dst = find_mode(b, mod1(src.mode + ifelse(isodd(i), 1, -1), num_modes(b)))

    new_b, val = excitation(b, (dst,), (src,))
    return new_b, val
end

function hopnextneighbour(
    b::SingleComponentFockAddress, i, boundary_condition::Symbol)
    src = find_occupied_mode(b, (i + 1) >>> 0x1)
    dir = ifelse(isodd(i), 1, -1)
    dst = find_mode(b, mod1(src.mode + dir, num_modes(b)))
    new_b, val = excitation(b, (dst,), (src,))
    on_boundary = src.mode == 1 && dir == -1 || src.mode == num_modes(b) && dir == 1
    if boundary_condition == :twisted && on_boundary
        return new_b, -val
    elseif boundary_condition == :hard_wall && on_boundary
        return new_b, 0.0
    else
        return new_b, val
    end
end

function hopnextneighbour(b::SingleComponentFockAddress, i, boundary_condition::Number)
    src = find_occupied_mode(b, (i + 1) >>> 0x1)
    dir = ifelse(isodd(i), 1, -1)
    dst = find_mode(b, mod1(src.mode + dir, num_modes(b)))
    new_b, val = excitation(b, (dst,), (src,))
    if (src.mode == 1 && dir == -1)
        return new_b, val*exp(-im*boundary_condition)
    elseif (src.mode == num_modes(b) && dir == 1)
        return new_b, val*exp(im*boundary_condition)
    else
        return new_b, complex(val)
    end
end

"""
    bose_hubbard_interaction(address)

Return ``Σ_i n_i (n_i-1)`` for computing the Bose-Hubbard on-site interaction (without the
``U`` prefactor.)

# Example

```jldoctest
julia> Hamiltonians.bose_hubbard_interaction(BoseFS{4,4}((2,1,1,0)))
2
julia> Hamiltonians.bose_hubbard_interaction(BoseFS{4,4}((3,0,1,0)))
6
```
"""
function bose_hubbard_interaction(b::BoseFS{<:Any,<:Any,A}) where {A<:BitString}
    return bose_hubbard_interaction(Val(num_chunks(A)), b)
end
function bose_hubbard_interaction(b::SingleComponentFockAddress)
    return bose_hubbard_interaction(nothing, b)
end

@inline function bose_hubbard_interaction(_, b::SingleComponentFockAddress)
    result = 0
    for (n, _, _) in occupied_modes(b)
        result += n * (n - 1)
    end
    return result
end

@inline function bose_hubbard_interaction(::Val{1}, b::BoseFS{<:Any,<:Any,<:BitString})
    # currently this ammounts to counting occupation numbers of modes
    chunk = chunks(b.bs)[1]
    matrixelementint = 0
    while !iszero(chunk)
        chunk >>>= (trailing_zeros(chunk) % UInt) # proceed to next occupied mode
        bosonnumber = trailing_ones(chunk) # count how many bosons inside
        # surpsingly it is faster to not check whether this is nonzero and do the
        # following operations anyway
        chunk >>>= (bosonnumber % UInt) # remove the counted mode
        matrixelementint += bosonnumber * (bosonnumber - 1)
    end
    return matrixelementint
end

###
### Variable particle number with N = missing
###
smallest_uint_type(n::Integer) = begin
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    n <= typemax(UInt8) && return UInt8
    n <= typemax(UInt16) && return UInt16
    n <= typemax(UInt32) && return UInt32
    n <= typemax(UInt64) && return UInt64
    n <= typemax(UInt128) && return UInt128
    throw(OverflowError("n is too large for fixed-width unsigned integers"))
end

function BoseFS{missing,M}(onr::SVector{M,T}) where {M,T<:Unsigned}
    return @inbounds BoseFS{missing,M,typeof(onr)}(onr)
end
function BoseFS{missing}(onr::SVector{M,T}) where {M,T<:Unsigned}
    return @inbounds BoseFS{missing,M,typeof(onr)}(onr)
end
function BoseFS{missing,M,S}(onr) where {M,S}
    S <: SVector{M,<:Unsigned} || throw(ArgumentError(
        "invalid container type: expected SVector{$M,<:Unsigned}; got $(S)"
    ))
    return BoseFS{missing,M,S}(S(onr))
end
function BoseFS{missing,M}(v::AbstractVector{<:Integer}; type=nothing) where {M}
    BoseFS{missing,M}(v...; type)
end
function BoseFS{missing}(arg; type=nothing) # single argument constructor
    isnothing(type) && (type = smallest_uint_type(maximum(arg)))
    type <: Unsigned || throw(ArgumentError("type must be an unsigned integer type"))
    onr = SVector{length(arg),type}(arg)
    return @inbounds BoseFS{missing}(onr)
end
BoseFS{missing}(args::Integer...; type=nothing) = BoseFS{missing}(Tuple(args); type)
BoseFS{missing}(arg::Integer; type=nothing) = BoseFS{missing}((arg,); type) # single mode address
function BoseFS{missing,M}(args::Integer...; type=nothing) where {M}
    BoseFS{missing,M}(Tuple(args); type)
end
function BoseFS{missing,M}(t::NTuple{M,T}; type=nothing) where {M,T<:Integer}
    BoseFS{missing}(SVector{M}(t); type)
end

# BoseFS from BoseFS
function BoseFS{N,M}(fs::BoseFS) where {N,M}
    M === num_modes(fs) || throw(ArgumentError(
        "number of modes must match: $M != $(num_modes(fs))"
    ))
    ons = occupation_number_representation(fs)
    return BoseFS{N,M}(ons)
end
function BoseFS{N}(fs::BoseFS) where {N}
    M = num_modes(fs)
    ons = occupation_number_representation(fs)
    return BoseFS{N,M}(ons)
end
function BoseFS{missing}(fs::BoseFS{N,M}; type=nothing) where {N,M}
    type === nothing && (type = smallest_uint_type(N))
    ons = occupation_number_representation(fs)
    return BoseFS{missing,M}(ons...; type)
end
function BoseFS{missing,M}(fs::BoseFS) where {M}
    M === num_modes(fs) || throw(ArgumentError(
        "number of modes must match: $M != $(num_modes(fs))"
    ))
    BoseFS{missing}(fs)
end
BoseFS(fs::BoseFS) = fs

# sparse constructors
function BoseFS{missing}(M::Integer, pairs::Pair...; type=nothing)
    BoseFS{missing}(M, pairs; type=type)
end
function BoseFS{missing}(M::Integer, pair::Pair; type=nothing)
    BoseFS{missing}(M, (pair,); type=type)
end
function BoseFS{missing}(M::Integer, pairs; type=nothing)
    BoseFS{missing,M}(pairs; type)
end
function BoseFS{missing,M}(pairs; type=nothing) where {M}
    BoseFS{missing}(sparse_to_onr(M, pairs); type)
end
function BoseFS{missing,M}(pair::Pair; type=nothing) where {M}
    BoseFS{missing}(sparse_to_onr(M, (pair,)); type)
end
BoseFS{missing}(pairs::Pair...; _...) = throw(ArgumentError("number of modes must be provided"))

function from_bose_onr(::Type{S}, onr) where {M,T<:Unsigned,S<:SVector{M,T}}
    return S(onr)
end
function to_bose_onr(onr::S, ::Val{M}) where {M, S <: SVector{M,<:Unsigned}}
    return onr
end
function print_address(io::IO, b::BoseFS{missing,M,S}; compact=false) where {T,M,S<:SVector{M,T}}
    if T === UInt8
        if compact
            print(io, "|", join(onr(b), ' '), "⟩{}")
        else
            print(io, "BoseFS{missing}", Int.(tuple(onr(b)...)))
        end
    else
        if compact
            print(io, "|", join(onr(b), ' '), "⟩{", string(T), "}")
        else
            print(io, "BoseFS{missing}(", Int(onr(b)[1]))
            foreach(i -> print(io, ", ", Int(onr(b)[i])), 2:M)
            print(io, "; type=", string(T), ")")
        end
    end
end

Interfaces.num_particles(a::BoseFS{missing}) = sum(Int, onr(a))
function Interfaces.maximum_mode_occupation(::Type{<:BoseFS{missing,M,S}}) where {T,M,S<:SVector{M,T}}
    return typemax(T)
end
Interfaces.maximum_mode_occupation(::Type{<:BoseFS{N}}) where {N} = N

@inline function _destroy(onr::SVector{M,T}, mode::Integer) where {M,T}
    val = onr[mode]
    @set! onr[mode] = val - one(T)
    return onr, val
end

@inline function _create(onr::SVector{M,T}, mode::Integer) where {M,T}
    val = onr[mode] + one(T)
    @set! onr[mode] = val
    return onr, val
end

function excitation(
    fs::BoseFS{missing,M,S},
    c::NTuple{<:Any,Int},
    d::NTuple{<:Any,Int}
) where {M,S<:SVector{M,<:Unsigned}}
    onr = fs.bs
    accumulator = 1.0 # to avoid overflow
    for i in d
        onr, val = _destroy(onr, i)
        iszero(val) && return fs, 0.0 # return early if invalid; efficient according to benchmarks
        accumulator *= val
    end
    for i in c
        onr, val = _create(onr, i)
        accumulator *= val
        iszero(val) && return fs, 0.0
    end
    return typeof(fs)(onr), √accumulator
end
function excitation(
    fs::BoseFS{missing},
    c::NTuple{N1,BoseFSIndex},
    d::NTuple{N2,BoseFSIndex}
) where {N1,N2}
    creations = ntuple(i -> c[i].mode, Val(N1)) # convert BoseFSIndex to mode number
    destructions = ntuple(i -> d[i].mode, Val(N2))
    return excitation(fs, creations, destructions)
end

# `SingleComponentFockAddress` interface for BoseFS{missing}

find_mode(fs::BoseFS{missing}, n::Integer, occ=nothing) = BoseFSIndex(fs.bs[n], n, n)
function find_mode(fs::BoseFS{missing}, ns::NTuple{N,Integer}, occ=nothing) where N
    return ntuple(i -> find_mode(fs, ns[i]), Val(N))
end

num_occupied_modes(fs::BoseFS{missing}) = count(!iszero, fs.bs)

# for the lazy iterator `occupied_modes` we adapt the `BoseOccupiedModes` type
function occupied_modes(fs::BoseFS{missing,M}) where {M}
    return BoseOccupiedModes{missing,M,typeof(fs)}(fs)
end

function Base.length(bom::BoseOccupiedModes{<:Any,<:Any,<:BoseFS{missing}})
    return num_occupied_modes(bom.storage)
end

function Base.iterate(bom::BoseOccupiedModes{<:Any,<:Any,<:BoseFS{missing,M}}, i=1) where M
    s = onr(bom.storage) # is an SVector with the onr
    while true
        i > length(s) && return nothing
        iszero(s[i]) || return BoseFSIndex(s[i], i, i), i + 1
        i += 1
    end
end
