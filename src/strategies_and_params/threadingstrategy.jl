using SplittablesBase: halve, amount
using Base.Threads: @spawn, nthreads, threadid

"""
    abstract type ThreadingStrategy

Controls how threading is performed in [`lomc!`](@ref). Must overload the following
functions:

* [`fciqmc_step!`](@ref)
* [`working_memory`](@ref)
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
    fciqmc_step!(
        t_strat::ThreadingStrategy, w, ham, v, shift, dτ, pnorm, m_strat)
    ) -> v, w, stats

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

Returns new references to the modified `v` and `w` and the step statistics.
"""
fciqmc_step!

"""
    working_memory(t_strat::ThreadingStrategy, dv)

Create a working memory instance compatible with `t_strat`.
"""
working_memory

"""
    NoThreading <: ThreadingStrategy

`ThreadingStrategy` that disables threading.
"""
struct NoThreading <: ThreadingStrategy end

working_memory(::NoThreading, dv) = similar(localpart(dv))

function fciqmc_step!(::NoThreading, w, ham, dv, shift, dτ, pnorm, m_strat)
    # single-threaded version suitable for MPI
    v = localpart(dv)
    @assert w ≢ v "`w` and `v` must not be the same object"
    zero!(w) # clear working memory

    stat_names, stats = step_stats(v, Val(1))
    for (add, val) in pairs(v)
        stats += SVector(fciqmc_col!(w, ham, add, val, shift, dτ))
    end
    r = apply_memory_noise!(w, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, w, stats)..., stat_names, r) # MPI syncronizing
end

"""
    ThreadsThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) based on `Base.Threads`.
"""
struct ThreadsThreading <: ThreadingStrategy end

working_memory(::ThreadsThreading, dv) = ntuple_working_memory(dv)

function fciqmc_step!(
    ::ThreadsThreading, ws::NTuple{N}, ham, dv, shift, dτ, pnorm, m_strat
) where {N}
    nthreads = Threads.nthreads()
    batchsize = max(100, min(length(dv)÷nthreads, round(Int,sqrt(length(dv))*10)))
    @assert N == nthreads "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    stat_names, stats = step_stats(v, Val(N))
    zero!.(ws)
    @sync for btr in Iterators.partition(pairs(v), batchsize)
        Threads.@spawn for (add, num) in btr
            stats[Threads.threadid()] .+= fciqmc_col!(
                ws[Threads.threadid()], ham, add, num, shift, dτ
            )
        end
    end
    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat)
    return (sort_into_targets!(dv, ws, stats)..., stat_names, r)
end

"""
    SplittablesThreading <: TheradingStrategy

`ThreadingStrategy` based on [SplittablesBase](https://github.com/JuliaFolds/SplittablesBase.jl).
"""
struct SplittablesThreading <: ThreadingStrategy end

working_memory(::SplittablesThreading, dv) = ntuple_working_memory(dv)

function fciqmc_step!(
    ::SplittablesThreading, ws::NTuple{N}, ham, dv, shift, dτ, pnorm, m_strat
) where {N}
    @assert N == nthreads() "`nthreads()` not matching dimension of `ws`"
    @assert N > 1 "attempted to run threaded code with one thread"
    v = localpart(dv)
    stat_names, stats = step_stats(v, Val(N))

    batchsize = max(100.0, min(amount(pairs(v))/N, sqrt(amount(pairs(v)))*10))

    # define recursive dispatch function that loops two halves in parallel
    function loop_configs!(ps) # recursively spawn threads
        if amount(ps) > batchsize
            two_halves = halve(ps) #
            fh = @spawn loop_configs!(two_halves[1]) # runs in parallel
            loop_configs!(two_halves[2])           # with second half
            wait(fh)                             # wait for fist half to finish
        else # run serial
            # id = threadid() # specialise to which thread we are running on here
            # serial_loop_configs!(ps, ws[id], statss[id], trng())
            for (add, num) in ps
                ss = fciqmc_col!(ws[threadid()], ham, add, num, shift, dτ)
                stats[threadid()] += SVector(ss)
            end
        end
        return nothing
    end

    zero!.(ws)
    loop_configs!(pairs(v))

    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat)
    return (sort_into_targets!(dv, ws, stats)... , stat_names, r)
end

"""
    ThreadsXSumThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) based on [`ThreadsX`](https://github.com/tkf/ThreadsX.jl)`.sum`.
"""
struct ThreadsXSumThreading <: ThreadingStrategy end

working_memory(::ThreadsXSumThreading, dv) = ntuple_working_memory(dv)

function fciqmc_step!(
    ::ThreadsXSumThreading, ws::NTuple{N}, ham, dv, shift, dτ, pnorm, m_strat
) where {N}
    @assert N == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)
    zero!.(ws)

    stats = ThreadsX.sum(
        SVector(fciqmc_col!(ws[Threads.threadid()], ham, add, val, shift, dτ)) for (add, val) in pairs(v)
    )
    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat)
    return (sort_into_targets!(dv, ws, stats)... , r)
end

"""
    ThreadsXMapThreading <: ThreadingStrategy

[`ThreadingStrategy`](@ref) based on [`ThreadsX`](https://github.com/tkf/ThreadsX.jl)`.map`.
"""
struct ThreadsXMapThreading <: ThreadingStrategy end

working_memory(::ThreadsXMapThreading, dv) = ntuple_working_memory(dv)

function fciqmc_step!(
    ::ThreadsXMapThreading, ws::NTuple{N}, ham, dv, shift, dτ, pnorm, m_strat
) where {N}
    # multithreaded version; should also work with MPI
    @assert N == Threads.nthreads() "`nthreads()` not matching dimension of `ws`"
    v = localpart(dv)

    statss = step_stats(v, Val(N))
    zero!.(ws) # clear working memory

    function col!(p) # take a pair address -> value and run `fciqmc_col!()` on it
        statss[threadid()] .+= fciqmc_col!(
            ws[threadid()], ham, p.first, p.second, shift, dτ
        )
        return nothing
    end

    # parallel execution happens here:
    ThreadsX.map(col!, pairs(v))

    # return ws, stats
    r = apply_memory_noise!(ws, v, shift, dτ, pnorm, m_strat) # memory noise
    return (sort_into_targets!(dv, ws, statss)... , r) # MPI syncronizing
end

"""
    select_threading_strategy(threading, targetwalkers)

Select a [`ThreadingStrategy`](@ref) to control threading in [`lomc!`](@ref).

`threading` can be:

* `:auto`: decide whether threading should be done or not based on `targetwalkers` and
  whether threads are available.
* `true` or `false`: use the default [`ThreadingStrategy`](@ref) or [`NoThreading`](@ref).
* Any [`ThreadingStrategy`](@ref).

The default [`ThreadingStrategy`](@ref) is currently [`SplittablesThreading`](@ref).
"""
select_threading_strategy(t::ThreadingStrategy, _) = t

function select_threading_strategy(threading::Symbol, targetwalkers)
    if threading == :auto
        t = targetwalkers > 500 && Threads.nthreads() > 1
        return select_threading_strategy(t, targetwalkers)
    else
        error("invalid threading strategy `$threading`")
    end
end
function select_threading_strategy(threading::Bool, _)
    if threading
        if Threads.nthreads() == 1
            @warn "threading was requested, but only one thread is available"
            return NoThreading()
        else
            return SplittablesThreading()
        end
    else
        return NoThreading()
    end
end
