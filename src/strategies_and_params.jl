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
Abstract type for defining the strategy for updating the time step with
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
N_w^{n} = \\sum_{i=0}^{k-1} \\|Ψ\\|_1^{n-i}
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

# """
#     HistoryLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3, d = 20, k = 5, n_w)
# Strategy for updating the shift according to [`HistoryLogUpdate`](@ref) after
# `targetwalkers` has been reached.
# """
# @with_kw mutable struct HistoryLogUpdateAfterTargetWalkers{T} <: ShiftStrategy
#     targetwalkers::Int # only start updating after this has been reached
#     ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
#     d::Int = 20 # delay for time window
#     k::Int = 5 # number of samples to take from history
#     n_w::T # for remembering last time step's sum of walker numbers
# end

"""
    update_shift(s <: ShiftStrategy, shift, shiftMode, tnorm, pnorm, dτ, step, df)
Update the shift according to strategy `s`. See [`ShiftStrategy`](@ref).
"""
# @inline function update_shift(s::HistoryLogUpdateAfterTargetWalkers,
#                         shift, shiftMode,
#                         tnorm, pnorm, dτ, step, df)
#     if shiftMode || tnorm > s.targetwalkers
#         prev_n_w = s.n_w # previous sum of walker numbers
#         # compute sum of walker numbers from history
#         this_n_w = tnorm
#         nhistory = size(df,1) # length of available history
#         for i in 1:(s.k-1)
#             i*s.d > nhistory && break
#             this_n_w += df[end - i*s.d,:norm]
#         end
#         s.n_w = this_n_w # store the result for next time
#         # this_n_w = sum([df[end - i*s.d] for i in 0:(s.k-1)])
#         # return new shift and new shiftMode
#         return shift - s.ζ/dτ * log(this_n_w/prev_n_w), true
#     end
#     return shift, false
# end

@inline function update_shift(s::HistoryLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    prev_n_w = s.n_w # previous sum of walker numbers
    # compute sum of walker numbers from history
    s.n_w = sum([df[end-i*s.d, :norm] for i in 0:(s.k-1)])
    # note that this will fail with a BoundsError if the history is not long enough

    # this_n_w = tnorm
    # nhistory = size(df,1)
    # for i in 1:(s.k-1)
    #     i*s.d > nhistory && break
    #     this_n_w += df[end - i*s.d,:norm]
    # end
    # s.n_w = this_n_w # store the result for next time
    # this_n_w = sum([df[end - i*s.d] for i in 0:(s.k-1)])
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(s.n_w/prev_n_w), true
end

@inline function update_shift(s::LogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(tnorm/pnorm), true
end

@inline function update_shift(s::DelayedLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    if step % s.a == 0 && size(df,1) > s.a
        prevnorm = df[end-s.a+1,:norm]
        return shift - s.ζ/(dτ * s.a) * log(tnorm/prevnorm), true
    else
        return shift, true
    end
end

@inline function update_shift(s::LogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, args...)
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(LogUpdate(s.ζ), shift, true, tnorm, args...)
    end
    return shift, false
end

@inline function update_shift(s::DelayedLogUpdateAfterTargetWalkers,
                        shift, shiftMode, tnorm, args...)
    if shiftMode || tnorm > s.targetwalkers
        return update_shift(DelayedLogUpdate(s.ζ,s.a), shift, true, tnorm, args...)
    end
    return shift, false
end

@inline update_shift(::DontUpdate, shift, args...) = (shift, false)
