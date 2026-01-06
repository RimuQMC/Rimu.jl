
"""
    ColumnStats{T,A,TT}()
Create an empty struct to hold the data and stats of `AbstractOperatorColumn`.

See also [`update_column_stats`](@ref).
"""
struct ColumnStats{T,A,TT<:AbstractFloat}
    addresses::Vector{A}
    values::Vector{T}
    mod_cumsum::Vector{TT}
    column_sum::Base.RefValue{T}
    column_length::Base.RefValue{Int}
end
function ColumnStats{T,A,TT}() where {T,A,TT}
    addresses = A[]
    values = T[]
    mod_cumsum = TT[]
    column_sum = zero(T)
    column_length = 0
    return ColumnStats(addresses, values, mod_cumsum, Ref(column_sum), Ref(column_length))
end
Base.length(stats::ColumnStats) = stats.column_length[]

function Base.show(io::IO, cs::ColumnStats{T,A,TT}) where {A,T,TT}
    if iszero(length(cs))
        print(io, "ColumnStats{$T,$A,$TT}()")
    else
        print(io, "ColumnStats{$T,$A,$TT} of length $(length(cs))")
    end
end

import Base: ==, hash
function ==(a::ColumnStats, b::ColumnStats)
    res = a.column_length[] == b.column_length[] && a.column_sum[] == b.column_sum[]
    res = res && a.addresses == b.addresses && a.values == b.values
    res = res && a.mod_cumsum == b.mod_cumsum
    return res
end
function hash(a::ColumnStats, h::UInt64)
    res = hash(a.addresses, h)
    res = hash(a.values, hash(a.mod_cumsum, res))
    res = hash(a.column_length[], hash(a.column_sum[], res))
    return res
end

"""
    IsWalkerDiffusion(hamiltonian_or_column; kwargs...) <: StochasticStyle{T}

Stochastic propagation with floating point walker numbers using the
walker diffusion algorithm. An instance can be used with the `style` keyword argument of
[`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem).

## Arguments:
* `hamiltonian_or_column`: An `AbstractHamiltonian` or an `AbstractOperatorColumn` to
  determine the types used in the stochastic style.

## Keyword arguments:
* `splitting_threshold = 1.0`: Values above twice this number are split into multiple
  walkers.
* `projection_threshold = 0.9 * splitting_threshold`: Values below this number are
    stochastically projected to this value or zero.
* `target_threshold = 1.1 * splitting_threshold`: Values above this number are
    reduced to this value by merging walkers.
* `rel_spawning_threshold = 1.0`: If the walker number on a configuration is greater than
  or equal to the number of offdiagonals times this threshold, spawning is done
  deterministically. Should be set to 1 or smaller for best performance.
* `abs_spawning_threshold = Inf`: If the walker number on a configuration is greater than
  this value, spawning is done deterministically. Can be set to e.g.
  `abs_spawning_threshold = 0.1 * target_walkers`.

In order to minimize the stochastic projection events, it is recommended to set
`projection_threshold < splitting_threshold < target_threshold`.

See also [`StochasticStyle`](@ref).
"""
struct IsWalkerDiffusion{T<:Number,A,TT<:AbstractFloat} <: StochasticStyle{T}
    splitting_threshold::TT
    projection_threshold::TT
    target_threshold::TT
    rel_spawning_threshold::TT
    abs_spawning_threshold::TT
    column_stats::ColumnStats{T,A,TT}
end
function IsWalkerDiffusion{T,A,TT}(;
    splitting_threshold=1.0,
    projection_threshold=0.9 * splitting_threshold,
    target_threshold=1.1 * splitting_threshold,
    rel_spawning_threshold=1.0,
    abs_spawning_threshold=Inf,
) where {T,A,TT}
    return IsWalkerDiffusion{T,A,TT}(
        TT(splitting_threshold),
        TT(projection_threshold),
        TT(target_threshold),
        TT(rel_spawning_threshold),
        TT(abs_spawning_threshold),
        ColumnStats{T,A,TT}()
    )
end
function IsWalkerDiffusion(column::AbstractOperatorColumn; kwargs...)
    T = eltype(parent_operator(column))
    A = typeof(starting_address(column))
    TT = promote_type(Float64, real(T))
    return IsWalkerDiffusion{T,A,TT}(; kwargs...)
end

function IsWalkerDiffusion(hamiltonian::AbstractHamiltonian; kwargs...)
    IsWalkerDiffusion(hamiltonian * starting_address(hamiltonian); kwargs...)
end
function Base.show(io::IO, iwd::IsWalkerDiffusion{T,A,TT}) where {T,A,TT}
    println(io, "IsWalkerDiffusion{$T,$A,$TT}(")
    println(io, "    splitting_threshold = $(iwd.splitting_threshold), ")
    println(io, "    projection_threshold = $(iwd.projection_threshold), ")
    println(io, "    target_threshold = $(iwd.target_threshold), ")
    println(io, "    rel_spawning_threshold = $(iwd.rel_spawning_threshold), ")
    print(io, "    abs_spawning_threshold = $(iwd.abs_spawning_threshold)\n)")
end
function ==(
    a::IsWalkerDiffusion{T,A,TT}, b::IsWalkerDiffusion{T,A,TT}
) where {T,A,TT}
    res = a.splitting_threshold == b.splitting_threshold &&
        a.projection_threshold == b.projection_threshold &&
        a.target_threshold == b.target_threshold &&
        a.rel_spawning_threshold == b.rel_spawning_threshold &&
        a.abs_spawning_threshold == b.abs_spawning_threshold &&
        a.column_stats == b.column_stats
    return res
end
function hash(iwd::IsWalkerDiffusion{T,A,TT}, h::UInt64) where {T,A,TT}
    res = hash(iwd.splitting_threshold, h)
    res = hash(iwd.projection_threshold, res)
    res = hash(iwd.target_threshold, res)
    res = hash(iwd.rel_spawning_threshold, res)
    res = hash(iwd.abs_spawning_threshold, res)
    res = hash(iwd.column_stats, res)
    return res
end

function update_column_stats!(stats::ColumnStats, oc::AbstractOperatorColumn)
    @unpack addresses, values, mod_cumsum = stats

    # prepare vectors
    length_estimate = num_offdiagonals(oc) + 1 # expected upper bound
    resize!(addresses, length_estimate)
    resize!(values, length_estimate)
    resize!(mod_cumsum, length_estimate)

    # zero accumulators
    mod_sum = zero(eltype(mod_cumsum))
    column_sum = zero(stats.column_sum[])
    i = 0

    # iterate through operator column
    out = iterate(oc)
    while out !== nothing
        (address, value), status = out
        i += 1
        addresses[i] = address
        values[i] = value
        column_sum += value
        mod_sum += abs(value)
        mod_cumsum[i] = mod_sum
        out = iterate(oc, status)
    end
    column_length = i
    resize!(addresses, column_length)
    resize!(values, column_length)
    resize!(mod_cumsum, column_length)

    stats.column_sum[] = column_sum
    stats.column_length[] = column_length
    return stats
end
function update_column_stats!(s::IsWalkerDiffusion, column)
    update_column_stats!(s.column_stats, column)
    return s
end

"""
    RandomFromCumsumIterator(n, cumsum)
Create an iterator that yields `n` random indices sampled from the cumulative sum `cumsum`.
The indices are sampled according to the probabilities defined by `cumsum`, assuming
`cumsum` contains a list of growing positive values. The `n` samples are correlated and
maximally spread out. Useful for spawning walkers in the walker diffusion algorithm.

See also [`IsWalkerDiffusion`](@ref).
"""
struct RandomFromCumsumIterator{T}
    cumsum::Vector{T}
    weight::T
    n::Int
end
function RandomFromCumsumIterator(n, cumsum)
    weight = last(cumsum) / n
    return RandomFromCumsumIterator(cumsum, weight, n)
end
Base.length(ri::RandomFromCumsumIterator) = ri.n

function Base.iterate(ri::RandomFromCumsumIterator, state=(0, 0.0, 1))
    @unpack cumsum, weight, n = ri
    number_returned, offset, current_index = state
    number_returned < n || return nothing
    chosen = rand() * weight
    @inbounds while true
        if chosen + offset < cumsum[current_index]
            return current_index, (number_returned + 1, offset + weight, current_index)
        end
        current_index += 1
    end
end

function step_stats(::IsWalkerDiffusion{T}) where {T}
    z = zero(T) # the local energy carries full signs and can be complex
    rz = real(zero(T)) # the stoquastic energy is non-negative and real
    names = (:local_energy, :coefficient_sum, :stoquastic_energy, :deaths, :walkers,
        :single_walkers, :exact_steps, :inexact_steps)
    values = MultiScalar(z, z, rz, 0, 0, 0, 0, 0)
    return names, values
end
function apply_column!(s::IsWalkerDiffusion{T}, w, column, val, boost=1) where {T}
    # boost is currently unused
    RT = real(T)

    @unpack splitting_threshold, projection_threshold, target_threshold,
    rel_spawning_threshold, abs_spawning_threshold = s

    # separate modulus and sign of value
    val_mod, val_sign = abs(val), sign(val)

    # stochastic projection
    if val_mod < projection_threshold
        if val_mod < rand() * target_threshold
            deaths = 1
            return (T(0), T(0), RT(0), deaths, 0, 0, 0, 0)
        else
            val_mod = target_threshold
        end
    end

    # instantiate column and compute stats
    stats = update_column_stats!(s.column_stats, column)
    mod_cumsum = stats.mod_cumsum
    column_sum = stats.column_sum[]
    column_norm = last(mod_cumsum)
    column_length = stats.column_length[]

    # decide how many walkers to propagate
    total_val_mod = column_norm * val_mod
    num_walkers = ceil(Int, total_val_mod / splitting_threshold)

    # decide about exact step
    if num_walkers > abs_spawning_threshold || num_walkers > column_length * rel_spawning_threshold
        exact_steps = 1
        for i in eachindex(stats.addresses)
            deposit!(
                w, stats.addresses[i], stats.values[i] * val_mod * val_sign,
                starting_address(column) => val
            )
        end
        return (
            T(column_sum * val_mod * val_sign), T(val_mod * val_sign),
            RT(column_norm * val_mod), 0, 0, 0, exact_steps, 0
        )
    else
        inexact_steps = 1
        val_mod_per_walker = total_val_mod / num_walkers

        for i in RandomFromCumsumIterator(num_walkers, mod_cumsum)
            deposit!(
                w, stats.addresses[i],
                val_mod_per_walker * val_sign * sign(stats.values[i]),
                starting_address(column) => val
            )
        end
        deaths = 0
        return (
            T(column_sum * val_mod * val_sign), T(val_mod * val_sign),
            RT(column_norm * val_mod), deaths, num_walkers, Int(num_walkers == 1),
            0, inexact_steps
        )
    end
end
