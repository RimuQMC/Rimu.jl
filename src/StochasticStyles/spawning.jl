"""
    projected_deposit!(w, add, val, parent, threshold=0)

Like [`deposit!`](@ref), but performs threshold projection before spawning. If `eltype(w)`
is an `Integer`, values are stochastically rounded.

Returns the value deposited.
"""
@inline function projected_deposit!(w, add, val, parent, threshold=0)
    return projected_deposit!(valtype(w), w, add, val, parent, threshold)
end
# Non-integer
@inline function projected_deposit!(
    ::Type{T}, w, add, value, parent, threshold
) where {T<:Union{AbstractFloat, Complex{<:AbstractFloat}}}
    thresh = abs(T(threshold))
    val = T(value)
    absval = abs(val)
    if absval < thresh
        if rand() < absval / thresh
            val = sign(val) * thresh
        else
            val = zero(T)
        end
    end

    if !iszero(val)
        deposit!(w, add, val, parent)
    end

    return val
end
# Round to integer
@inline function projected_deposit!(
    ::Type{T}, w, add, val, parent, threshold=0
) where {T<:Integer}
    if !iszero(threshold)
        throw(ArgumentError("Thresholding not supported for integer spawns"))
    end

    new_val = T(sign(val)) * floor(T, abs(val) + rand())

    if !iszero(new_val)
        deposit!(w, add, new_val, parent)
    end
    return new_val
end
# Complex/Int
@inline function projected_deposit!(
    ::Type{T}, w, add, val, parent, threshold=0
) where {I<:Integer,T<:Complex{I}}
    if !iszero(threshold)
        throw(ArgumentError("Thresholding not supported for integer spawns"))
    end

    val_re, val_im = reim(val)

    new_val_re = I(sign(val_re)) * floor(I, abs(val_re) + rand())
    new_val_im = I(sign(val_im)) * floor(I, abs(val_im) + rand())
    new_val = new_val_re + im * new_val_im

    if !iszero(new_val)
        deposit!(w, add, new_val, parent)
    end
    return new_val_re + im * new_val_im
end

"""
    diagonal_step!(w, column, val, threshold=0) -> (clones, deaths, zombies)

Perform diagonal step on a walker `starting_address(column)`. Optional argument
`threshold` sets the projection threshold. If `eltype(w)` is an `Integer`, the `val` is
rounded to the nearest integer stochastically.
"""
@inline function diagonal_step!(w, column, val, threshold=0)
    new_val = diagonal_element(column) * val
    res = projected_deposit!(w, starting_address(column), new_val, column => val, threshold)
    return clones_deaths_zombies(res, typeof(res)(val))
end

@inline function clones_deaths_zombies(res::T, val::T) where {T<:Real}
    clones = deaths = zombies = zero(T)
    if res > val
        # walker number increased
        clones = abs(res - val)
    elseif sign(res) ≠ sign(val)
        # walker number decreased so much that sign changed
        deaths = abs(val)
        zombies = abs(res)
    else
        # walker number decreased, but not too much
        deaths = abs(res - val)
    end
    return (clones, deaths, zombies)
end
@inline function clones_deaths_zombies(res::Complex, val::Complex)
    res_re, res_im = reim(res)
    val_re, val_im = reim(val)

    clones_re, deaths_re, zombies_re = clones_deaths_zombies(res_re, val_re)
    clones_im, deaths_im, zombies_im = clones_deaths_zombies(res_im, val_im)
    clones = clones_re + im * clones_im
    deaths = deaths_re + im * deaths_im
    zombies = zombies_re + im * zombies_im

    return (clones, deaths, zombies)
end

"""
    SpawningStrategy

A `SpawningStrategy` is used to control how spawns (multiplies with off-diagonal part of the
column vector) are performed and can be passed to some of the [`StochasticStyle`](@ref)s as
keyword arguments.

The following concrete implementations are provided:

* [`Exact`](@ref): Perform exact spawns. Used by [`IsDeterministic`](@ref).

* [`WithReplacement`](@ref): The default stochastic spawning strategy. Spawns are chosen
  with replacement.

* [`DynamicSemistochastic`](@ref): Behave like [`Exact`](@ref) when the number of spawns
  performed is high, and like a different substrategy otherwise. Used by
  [`IsDynamicSemistochastic`](@ref).

* [`SingleSpawn`](@ref): Perform a single spawn only. Used as a building block for other
  strategies.

* [`WithoutReplacement`](@ref): Similar to [`WithReplacement`](@ref), but ensures each spawn
  is only performed once. Only to be used as a substrategy of
  [`DynamicSemistochastic`](@ref).

* [`Bernoulli`](@ref): Each spawn is attempted with a fixed probability. Only to be used as
  a substrategy of [`DynamicSemistochastic`](@ref).

## Interface

In order to implement a new `SpawningStrategy`, define a method for [`spawn!`](@ref).
"""
abstract type SpawningStrategy end

"""
    Exact(threshold=0.0) <: SpawningStrategy

Perform an exact spawning step.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct Exact{T} <: SpawningStrategy
    threshold::T

    Exact(threshold::T=0.0) where {T} = new{T}(threshold)
end

"""
    spawn!(s::SpawningStrategy, w, column, val, boost)

Perform stochastic spawns to `w` from `column` with `val` walkers. `val * boost`
controls the number of spawns performed.

See [`SpawningStrategy`](@ref).
"""
@inline function spawn!(s::Exact, w, column, val, boost=1)
    T = valtype(w)
    attempts = 0
    spawns = real(zero(T))
    for (new_add, mat_elem) in offdiagonals(column)
        attempts += 1
        spawns += abs(projected_deposit!(
            w, new_add, val * mat_elem, column => val, s.threshold
        ))
    end
    return (attempts, spawns)
end

"""
    SingleSpawn(threshold=0.0) <: SpawningStrategy

Perform a single spawn. Useful as a building block for other stochastic styles.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts (always 1)
and the number of spawns.
"""
struct SingleSpawn{T} <: SpawningStrategy
    threshold::T

    SingleSpawn(threshold::T=0.0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::SingleSpawn, w, column, val, boost=1)
    if iszero(val)
        return (1, real(zero(valtype(w))))
    else
        new_add, prob, mat_elem = random_offdiagonal(column)
        new_val = val * mat_elem / prob
        spawns = abs(projected_deposit!(w, new_add, new_val, column => val, s.threshold))
        return (1, spawns)
    end
end

"""
    WithReplacement(threshold=0.0) <: SpawningStrategy

[`SpawningStrategy`](@ref) where spawn targets are sampled with replacement. This is the
default spawning strategy for most of the [`StochasticStyle`](@ref)s.

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct WithReplacement{T} <: SpawningStrategy
    threshold::T

    WithReplacement(threshold::T=0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::WithReplacement, w, column, val, boost=1)
    spawns = real(zero(valtype(w)))
    num_attempts = max(floor(Int, abs(val) * boost), 1)
    magnitude = val / num_attempts

    for _ in 1:num_attempts
        new_add, prob, mat_elem = random_offdiagonal(column)
        new_val = mat_elem * magnitude / prob
        spawns += abs(projected_deposit!(w, new_add, new_val, column => val, s.threshold))
    end
    return (num_attempts, spawns)
end

"""
    WithoutReplacement(threshold=0.0) <: SpawningStrategy

[`SpawningStrategy`](@ref) where spawn targets are sampled without replacement. This
strategy needs to allocate a temporary array during spawning, which makes it significantly
less efficient than [`WithReplacement`](@ref).

If the number of spawn attempts is greater than the number of offdiagonals, this functions
like [`Exact`](@ref), but is less efficient. For best performance, this strategy is to be
used as a substrategy of [`DynamicSemistochastic`](@ref).

## Parameters

* `threshold` sets the projection threshold. If set to zero, no projection is performed.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct WithoutReplacement{T} <: SpawningStrategy
    threshold::T

    WithoutReplacement(threshold::T=0.0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::WithoutReplacement, w, column, val, boost=1)
    offdiags = offdiagonals(column)
    if !(offdiags isa AbstractVector)
        throw(ArgumentError(
            "The WithoutReplacement strategy requires offdiagonals to be an AbstractVector. Please use a different strategy."
        ))
    end
    spawns = zero(valtype(w))
    num_attempts = max(floor(Int, abs(val) * boost), 1)

    if abs(num_attempts) ≤ 1
        spawn!(SingleSpawn(s.threshold), w, column, val)
    else
        magnitude = val / num_attempts
        num_offdiags = num_offdiagonals(column)
        prob = 1 / num_offdiags
        for i in sample(1:num_offdiags, num_attempts; replace=false)
            new_add, mat_elem = offdiags[i]
            new_val = mat_elem * magnitude / prob
            spawns += abs(projected_deposit!(w, new_add, new_val, column => val, s.threshold))
        end
    end
    return (num_attempts, spawns)
end

"""
    Bernoulli(threshold=0.0) <: SpawningStrategy

Perform Bernoulli sampling. A spawn is attempted on each offdiagonal element with a
probability that results in an expected number of spawns equal to the number of walkers on
the spawning configuration. This is significantly less efficient than
[`WithReplacement`](@ref).

If the number of spawn attempts is greater than the number of offdiagonals, this functions
like [`Exact`](@ref), but is less efficient. For best performance, this strategy is to be
used as a substrategy of [`DynamicSemistochastic`](@ref).

## Parameters

* `threshold` sets the projection threshold.

[`spawn!`](@ref) with this strategy returns the number of spawn attempts and the
number of spawns.
"""
struct Bernoulli{T} <: SpawningStrategy
    threshold::T

    Bernoulli(threshold::T=0.0) where {T} = new{T}(threshold)
end

@inline function spawn!(s::Bernoulli, w, column, val, boost=1)
    offdiags = offdiagonals(column)
    if !(offdiags isa AbstractVector)
        throw(ArgumentError(
            "The Bernoulli strategy requires offdiagonals to be an AbstractVector. Please use a different strategy."
        ))
    end
    spawns = zero(valtype(w))
    # General case.
    num_offdiags = num_offdiagonals(column)
    prob = abs(val) * boost / num_offdiags
    num_attempts = 0
    for i in 1:num_offdiags
        if rand() < prob
            new_add, mat_elem = offdiags[i]
            new_val = mat_elem / prob * val
            spawns += abs(projected_deposit!(w, new_add, new_val, column => val, s.threshold))
            num_attempts += 1
        end
    end
    return (num_attempts, spawns)
end

"""
    DynamicSemistochastic(; strat, rel_threshold, abs_threshold) <: SpawningStrategy

[`SpawningStrategy`](@ref) that behaves like `strat` when the number of walkers is low, but
performs exact steps when it is high. What "high" means is controlled by the two thresholds
described below.

## Parameters

* `strat = WithReplacement()`: a [`SpawningStrategy`](@ref) to use when the multiplication
  is not performed exactly. If the `strat` has a `threshold` different from zero, all spawns
  will be projected to that threshold.

* `rel_threshold = 1.0`: If the walker number on a configuration (multiplied by the `boost`
  argument to [`spawn!`](@ref)) is greater than or equal to the number of offdiagonals
  times this threshold, spawning is done deterministically. Should be set to 1 or smaller
  for best performance.

* `abs_threshold = Inf`: If the walker number on a configuration (multiplied by the `boost`
  argument to [`spawn!`](@ref)) is greater than this value, spawning is done
  deterministically.

See e.g. [`WithoutReplacement`](@ref) for a description of the `strat.threshold` parameter.

[`spawn!`](@ref) with this strategy returns the numbers of exact and inexact spawns, the
number of spawn attempts and the number of spawns.
"""
Base.@kwdef struct DynamicSemistochastic{T,S<:SpawningStrategy} <: SpawningStrategy
    strat::S = WithReplacement()
    rel_threshold::T = 1.0
    abs_threshold::T = Inf
end

@inline function spawn!(s::DynamicSemistochastic, w, column, val, boost=1)
    # assumes that s.strat.threshold is defined
    # special-case substrategies that don't fit the pattern?
    amount = boost * abs(val)
    if amount ≥ num_offdiagonals(column) * s.rel_threshold || amount > s.abs_threshold
        # Exact multiplication.
        attempts, spawns = spawn!(Exact(s.strat.threshold), w, column, val)
        return (1, 0, attempts, spawns)
    else
        # Regular spawns.
        attempts, spawns = spawn!(s.strat, w, column, val, boost)
        return (0, 1, attempts, spawns)
    end
end

# bypass branching code for Exact() sub-strategy
@inline function spawn!(
    s::DynamicSemistochastic{<:Any,<:Exact}, w, column, args...
)
    return (1, 0, spawn!(s.strat, w, column, args...)...)
end
