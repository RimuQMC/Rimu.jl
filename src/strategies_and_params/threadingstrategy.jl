using SplittablesBase: halve, amount
using Base.Threads: @spawn, nthreads, threadid

"""
    abstract type ThreadingStrategy

Controls how threading is performed in [`lomc!`](@ref).

# Interface

* [`fciqmc_step!`](@ref)
* [`working_memory`](@ref)

# Implemented Strategies

* [`NoThreading`](@ref)
* [`ThreadsThreading`](@ref)
* [`SplittablesThreading`](@ref)
* [`ThreadsXSumThreading`](@ref)
* [`ThreadsXForeachThreading`](@ref)

"""
abstract type ThreadingStrategy end

"""
    ntuple_working_memory(dv::AbstractDVec)

Create a `NTuple{N}` of vectors that are `similar` to `dv`, where `N = Threads.nthreads()`.
"""
ntuple_working_memory(dv) = ntuple_working_memory(localpart(dv))
function ntuple_working_memory(v::AbstractDVec)
    return Tuple(similar(v) for _ in 1:nthreads())
end
function ntuple_working_memory(v::AbstractVector)
    return Tuple(similar(v) for _ in 1:nthreads())
end

"""
    fciqmc_step!(t_strat::ThreadingStrategy, w, ham, v, shift, dτ) -> stat_names, stats

Perform a single matrix(/operator)-vector multiplication:

```math
v^{(n + 1)} = [1 - dτ(\\hat{H} - S)]⋅v^n ,
```

where `Ĥ == ham` and `S == shift`.

Whether the operation is performed in stochastic, semistochastic, or determistic way is
controlled by the trait `StochasticStyle(w)`. See [`StochasticStyle`](@ref).

Whether the multiplication is performed on multiple threads is controlled by `t_strat` (see
[`ThreadingStrategy`](@ref)). `w` is the working memory corresponding to `t_strat` (see
[`working_memory`](@ref)).

Returns the step stats generated by the `StochasticStyle`.
"""
fciqmc_step!

"""
    working_memory(t_strat::ThreadingStrategy, dv)

Create a working memory instance compatible with `t_strat`. The working memory must be
compatible with [`sort_into_targets!`](@ref).
"""
working_memory

"""
    NoThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) that disables threading.
"""
struct NoThreading <: ThreadingStrategy end

working_memory(::NoThreading, dv) = similar(localpart(dv))

function fciqmc_step!(::NoThreading, w, ham, dv, shift, dτ)
    # single-threaded version suitable for MPI
    v = localpart(dv)
    @assert w ≢ v "`w` and `v` must not be the same object"
    zero!(w) # clear working memory

    stat_names, stats = step_stats(v, Val(1))
    for (add, val) in pairs(v)
        stats += fciqmc_col!(w, ham, add, val, shift, dτ)
    end
    return stat_names, stats
end

"""
    ThreadsThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) based on `Base.Threads`.
"""
struct ThreadsThreading <: ThreadingStrategy end

working_memory(::ThreadsThreading, dv) = ntuple_working_memory(dv)

function fciqmc_step!(::ThreadsThreading, ws::NTuple{N}, ham, dv, shift, dτ) where {N}
    nthreads = Threads.nthreads()
    batchsize = max(100, min(length(dv) ÷ nthreads, round(Int, sqrt(length(dv)) * 10)))
    @assert N == nthreads "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    stat_names, stats = step_stats(v, Val(N))
    zero!.(ws)
    @sync for btr in Iterators.partition(pairs(v), batchsize)
        Threads.@spawn for (add, num) in btr
            ss = fciqmc_col!(ws[Threads.threadid()], ham, add, num, shift, dτ)
            stats[Threads.threadid()] += ss
        end
    end
    return stat_names, stats
end

"""
    SplittablesThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) based on [SplittablesBase](https://github.com/JuliaFolds/SplittablesBase.jl).
"""
struct SplittablesThreading <: ThreadingStrategy end

working_memory(::SplittablesThreading, dv) = ntuple_working_memory(dv)

@inline function _loop_configs!(ws, stats, ham, pairs, shift, dτ, batchsize)
    if amount(pairs) > batchsize
        two_halves = halve(pairs)
        first_half = @spawn _loop_configs!(
            ws, stats, ham, two_halves[1], shift, dτ, batchsize
        )
        _loop_configs!(
            ws, stats, ham, two_halves[2], shift, dτ, batchsize
        )
        wait(first_half)
    else
        for (add, num) in pairs
            stats[threadid()] += fciqmc_col!(ws[threadid()], ham, add, num, shift, dτ)
        end
    end
    return nothing
end

function fciqmc_step!(::SplittablesThreading, ws::NTuple{N}, ham, dv, shift, dτ) where {N}
    @assert N > 1 "attempted to run threaded code with one thread"
    v = localpart(dv)
    stat_names, stats = step_stats(v, Val(N))
    batchsize = max(100.0, min(amount(pairs(v))/N, sqrt(amount(pairs(v))) * 10))

    zero!.(ws)
    _loop_configs!(ws, stats, ham, pairs(v), shift, dτ, batchsize)

    return stat_names, stats
end

"""
    ThreadsXSumThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) based on [`ThreadsX`](https://github.com/tkf/ThreadsX.jl)`.sum`.
"""
struct ThreadsXSumThreading <: ThreadingStrategy end

working_memory(::ThreadsXSumThreading, dv) = ntuple_working_memory(dv)

function fciqmc_step!(::ThreadsXSumThreading, ws::NTuple{N}, ham, dv, shift, dτ) where {N}
    @assert N == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    zero!.(ws)

    stat_names, _ = step_stats(v, Val(1))

    stats = ThreadsX.sum(pairs(v)) do (add, val)
        MultiScalar(fciqmc_col!(ws[Threads.threadid()], ham, add, val, shift, dτ))
    end
    return stat_names, stats
end

"""
    ThreadsXForeachThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) based on [`ThreadsX`](https://github.com/tkf/ThreadsX.jl)`.foreach`.
"""
struct ThreadsXForeachThreading <: ThreadingStrategy end

working_memory(::ThreadsXForeachThreading, dv) = ntuple_working_memory(dv)

function fciqmc_step!(::ThreadsXForeachThreading, ws::NTuple{N}, ham, dv, shift, dτ) where {N}
    # multithreaded version; should also work with MPI
    @assert N == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)

    stat_names, stats = step_stats(v, Val(N))
    zero!.(ws) # clear working memory

    # parallel execution happens here:
    ThreadsX.foreach(pairs(v)) do (add, val)
        tid = Threads.threadid()
        stats[tid] += fciqmc_col!(ws[tid], ham, add, val, shift, dτ)
    end
    return stat_names, stats
end

"""
    select_threading_strategy(threading, targetwalkers)

Select a [`ThreadingStrategy`](@ref) to control threading in [`lomc!`](@ref).

`threading` can be:

* `:auto`: Decide whether threading should be done or not based on `targetwalkers` and
  whether threads are available.
* `:threadlocal`: Use [`SplittablesThreading`](@ref) with thread-local working memory. This
  minimises allocations and can be faster than the default, but it is not guaranteed to be
  safe.
* `true` or `false`: Use [`ReproducibleThreading`](@ref) as the default
  [`ThreadingStrategy`](@ref), or [`NoThreading`](@ref).
* Any [`ThreadingStrategy`](@ref).
"""
select_threading_strategy(t::ThreadingStrategy, _) = t

function select_threading_strategy(threading::Symbol, targetwalkers)
    if threading == :auto
        t = targetwalkers > 500 && Threads.nthreads() > 1
        return select_threading_strategy(t, targetwalkers)
    elseif threading == :threadlocal
        if Threads.nthreads() == 1
            @warn "threading was requested, but only one thread is available"
            return NoThreading()
        else
            return SplittablesThreading()
        end
    else
        throw(ArgumentError("invalid threading strategy `$threading`"))
    end
end
function select_threading_strategy(threading::Bool, _)
    if threading
        if Threads.nthreads() == 1
            @warn "threading was requested, but only one thread is available"
            return NoThreading()
        else
            return ReproducibleThreading()
        end
    else
        return NoThreading()
    end
end

"""
    ReproducibleThreading(; batch_base = Threads.nthreads()) <: ThreadingStrategy
Use threads in a way that reproducible Monte Carlo results are obtained when the
random number generator is seeded before calling [`lomc!`](@ref). When the keyword argument
`batch_base` is set, the results will be reproducible independently of the number of
available threads.

Notes:
- This [`ThreadingStrategy`](@ref) leads to appreciable memory allocations and it is
  typically a little (but not much) slower than [`SplittablesThreading`](@ref).
- Doesn't return spawning stats.

See [`ThreadingStrategy`](@ref).
"""
struct ReproducibleThreading{N} <: ThreadingStrategy
    n::N
end
ReproducibleThreading(; batch_base=Threads.nthreads()) = ReproducibleThreading(batch_base)

working_memory(::ReproducibleThreading, dv) = zero(localpart(dv))

@inline function _rt_loop_configs!(w, ham, pairs, shift, dτ, batchsize)
    if amount(pairs) > batchsize # recursively halve `pairs` iterator
        two_halves = halve(pairs)

        # first_half gets new working memory; is spawned off to another task/thread
        w_fh = DictVectors.SkinnyDVec(keytype(w)[], valtype(w)[], StochasticStyle(w))

        first_half_task = Threads.@spawn _rt_loop_configs!(
            w_fh, ham, two_halves[1], shift, dτ, batchsize
        )

        # second half uses the passed-in working memory
        w = _rt_loop_configs!(
            w, ham, two_halves[2], shift, dτ, batchsize
        )

        # harvest the results from the spawned task and combine them
        w_fh = fetch(first_half_task)
        add!(w, w_fh) # combine the DVecs
    else # amount of `pairs` too small, process them on this thread and task
        # stats = sum(pairs) do (add, val)
        #     MultiScalar(fciqmc_col!(w, ham, add, val, shift, dτ))
        # end
        # # Returning stats this way is allocating and costs time. Therefore we skip it.
        for (add, num) in pairs
            fciqmc_col!(w, ham, add, num, shift, dτ)
        end
    end
    return w # results are returned in the passed-in working memory `w`
end

function fciqmc_step!(t::ReproducibleThreading, wm, ham, dv, shift, dτ)
    v = localpart(dv)
    stat_names, stats_def = step_stats(StochasticStyle(v))
    batchsize = max(100.0, min(amount(pairs(v))/t.n, sqrt(amount(pairs(v))) * 10))

    zero!(wm)
    wm = _rt_loop_configs!(wm, ham, pairs(v), shift, dτ, batchsize)

    return stat_names, stats_def
end
