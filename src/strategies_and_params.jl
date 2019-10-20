# define parameters and strategies for fciqmc as well as methods that use them
#
# FciqmcRunStrategy, RunTillLastStep
#
# TimeStepStrategy, update_dτ()
#
# ShiftStrategy, update_shift()

"""
Abstract type representing the strategy for running and terminating
[`fciqmc!()`](@ref). Implemented strategies:

   * [`RunTillLastStep`](@ref)
"""
abstract type FciqmcRunStrategy end

"""
    RunTillLastStep(step::Int = 0 # number of current/starting timestep
                 laststep::Int = 50 # number of final timestep
                 shiftMode::Bool = false # whether to adjust shift
                 shift::Float64 = 0.0 # starting/current value of shift
                 dτ::Float64 = 0.01 # current value of time step
    ) <: FciqmcRunStrategy
Parameters for running [`fciqmc!()`](@ref) for a fixed number of time steps.
"""
@with_kw mutable struct RunTillLastStep <: FciqmcRunStrategy
    step::Int = 0 # number of current/starting timestep
    laststep::Int = 50 # number of final timestep
    shiftMode::Bool = false # whether to adjust shift
    shift::Float64 = 0.0 # starting/current value of shift
    dτ::Float64 = 0.01 # time step
end

"""
Abstract type for strategies for reporting data in a DataFrame with
[`report!()`](@ref). Implemented strategies:
   * [`EveryTimeStep`](@ref)
   * [`EveryKthStep`](@ref)
   * [`ReportDFAndInfo`](@ref)
"""
abstract type ReportingStrategy end

"Report every time step."
struct EveryTimeStep <: ReportingStrategy end

"""
    EveryKthStep(;k = 10)
Report every `k`th step.
"""
@with_kw struct EveryKthStep <: ReportingStrategy
    k::Int = 10
end

"""
    ReportDFAndInfo(; k=10, i=100, io=stdout, writeinfo=true)
Report every `k`th step in DataFrame and write info message to `io` every `i`th
step (unless `writeinfo == false`). The flag `writeinfo` is useful for
controlling info messages in MPI codes.
"""
@with_kw struct ReportDFAndInfo <: ReportingStrategy
    k::Int = 10 # how often to write to DataFrame
    i::Int = 100 # how often to write info message
    io::IO = stdout # IO stream for info messages
    writeinfo::Bool = true # write info only if true - useful for MPI codes
end

"""
    report!(df::DataFrame, t::Tuple, s<:ReportingStrategy)
Record results in `df` and write informational messages according to strategy
`s`. See [`ReportingStrategy`](@ref).
"""
report!(df::DataFrame,t::Tuple,s::EveryTimeStep) = push!(df,t)

function report!(df::DataFrame,t::Tuple,s::EveryKthStep)
    step = t[1]
    step % s.k == 0 && push!(df,t) # only push to df if step is multiple of s.k
    return df
end

function report!(df::DataFrame,t::Tuple,s::ReportDFAndInfo)
    step = t[1]
    step % s.k == 0 && push!(df,t) # only push to df if step is multiple of s.k
    if s.writeinfo && step % s.i == 0
        println(s.io, "Step ", step)
        flush(s.io)
    end
    return df
end

"""
Abstract type for strategies for load balancing distributed computation with
[`load_balance!()`](@ref). Implemented strategies:
   * [`NoLB`](@ref)
   * [`ReportLoad`](@ref)
"""
abstract type LoadBalanceStrategy end

"Ignore load balancing. See [`LoadBalanceStrategy`](@ref)."
struct NoLB <: LoadBalanceStrategy end

"""
Report the load of each worker but do not perform active load balancing.
See [`LoadBalanceStrategy`](@ref)."""
struct ReportLoad <: LoadBalanceStrategy end

"""
    load_balance_initial(v, step, llen, lnorm, s::LoadBalanceStrategy)
    -> ldf
Initiate load balancing for data structure `v`. Returns `DataFrame` or `missing`
if the load balancing strategy `s` does not record data.
See [`LoadBalanceStrategy`](@ref).
"""
load_balance_initial(v, step, llen, lnorm, s::NoLB) = missing

"""
    load_balance!(ldf, step, llen, lnorm, s::LoadBalanceStrategy)
Perform load balancing and reporting through `ldf` according to stratgy `s`.
Initialise with [`load_balance_initial()`](@ref).
See [`LoadBalanceStrategy`](@ref).
"""
load_balance!(ldf, v, step, llen, lnorm, s::NoLB) = missing

function load_balance_initial(v, step, llen, lnorm, s::ReportLoad)
    ldf = DataFrame(steps = Int[], id = Int[], llen=Int[], lnorm = Float64[])
    return ldf
end

function load_balance!(ldf::DataFrame, v, step, llen, lnorm, s::ReportLoad)
    push!(ldf, (step, get_id(v), llen, lnorm))
    return ldf
end

"""
Abstract type for strategies for updating the time step with
[`update_dτ()`](@ref). Implemented
strategies:

   * [`ConstantTimeStep`](@ref)
"""
abstract type TimeStepStrategy end

"Keep `dτ` constant."
struct ConstantTimeStep <: TimeStepStrategy end

"""
    update_dτ(s<:TimeStepStrategy, dτ, args...) -> new dτ
Update the time step according to the strategy `s`.
"""
update_dτ(::ConstantTimeStep, dτ, args...) = dτ
# here we implement the trivial strategy: don't change dτ

"""
Abstract type for defining the strategy for updating the `shift` with
[`update_shift()`](@ref). Implemented strategies:

   * [`DontUpdate`](@ref)
   * [`LogUpdate`](@ref)
   * [`DelayedLogUpdate`](@ref)
   * [`LogUpdateAfterTargetWalkers`](@ref)
   * [`DelayedLogUpdateAfterTargetWalkers`](@ref)
   * [`HistoryLogUpdate`](@ref)
"""
abstract type ShiftStrategy end

"""
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.
See [`LogUpdate`](@ref).
"""
@with_kw struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
end

"""
    DelayedLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3, a = 10) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and delay of
`a` steps. See [`DelayedLogUpdate`](@ref).
"""
@with_kw struct DelayedLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end

"""
    LogUpdate(ζ = 0.3) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)
```
"""
@with_kw struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
end

"""
    DelayedLogUpdate(ζ = 0.3, a = 10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and delay of `a` steps.

```math
S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+a}}{\\|Ψ\\|_1^n}\\right)
```
"""
@with_kw struct DelayedLogUpdate <: ShiftStrategy
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end

"`DontUpdate() <: ShiftStrategy` Don't update the `shift`."
struct DontUpdate <: ShiftStrategy end

"""
    HistoryLogUpdate(df::DataFrame; d = 100, k=1, ζ= 0.3)
Strategy for updating the shift according to log formula but with walker
numbers accumulated from `k` samples of the history with delay `d`. A
recent history has to be passed with the data frame `df` for initialisation.
```math
N_w^{n} = \\sum_{i=0}^{k-1} \\|Ψ\\|_1^{n-i} \\\\
S^{n+a} = S^n -\\frac{ζ}{a dτ}\\ln\\left(\\frac{N_w^{n+1}}{N_w^n}\\right)
```
"""
mutable struct HistoryLogUpdate{T} <: ShiftStrategy
    ζ::Float64 # damping parameter, best left at value of 0.3
    d::Int # delay for time window
    k::Int # number of samples to take from history
    n_w::T # for remembering last time step's sum of walker numbers
end
function HistoryLogUpdate(df::DataFrame; d = 100, k = 1, ζ= 0.3)
    size(df,1) ≤ d*k && @error "insufficient history for `HistoryLogUpdate`"
    n_w = sum(df[end-i*d, :norm] for i in 0:(k-1))
    return HistoryLogUpdate(ζ, d, k, n_w)
end

"""
    update_shift(s <: ShiftStrategy, shift, shiftMode, tnorm, pnorm, dτ, step, df)
Update the shift according to strategy `s`. See [`ShiftStrategy`](@ref).
"""
@inline function update_shift(s::HistoryLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    prev_n_w = s.n_w # previous sum of walker numbers
    # compute sum of walker numbers from history
    s.n_w = sum([df[end-i*s.d, :norm] for i in 0:(s.k-1)])
    # note that this will fail with a BoundsError if the history is not long enough
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(s.n_w/prev_n_w), true, tnorm
end

@inline function update_shift(s::LogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(tnorm/pnorm), true, tnorm
end

@inline function update_shift(s::DelayedLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    if step % s.a == 0 && size(df,1) > s.a
        prevnorm = df[end-s.a+1,:norm]
        return shift - s.ζ/(dτ * s.a) * log(tnorm/prevnorm), true, tnorm
        # return shift - s.ζ/(dτ * s.a) * log(tnorm/pnorm), true, tnorm
    else
        return shift, true, pnorm # important: return the old norm - not updated
    end
end

@inline function update_shift(s::LogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, args...)
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(LogUpdate(s.ζ), shift, true, tnorm, args...)
    end
    return shift, false, tnorm
end

@inline function update_shift(s::DelayedLogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, pnorm, args...)
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(DelayedLogUpdate(s.ζ,s.a), shift, true, tnorm, pnorm, args...)
    end
    return shift, false, pnorm
end

@inline update_shift(::DontUpdate, shift, tnorm, args...) = (shift, false, tnorm)
