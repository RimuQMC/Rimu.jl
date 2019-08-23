# define parameters and strategies for fciqmc as well as methods that use them
#
# FCIQMCParams
#
# TimeStepStrategy, update_dτ()
#
# ShiftStrategy, update_shift()



"""
    FCIQMCParams(step::Int = 0 # number of current/starting timestep
                 laststep::Int = 50 # number of final timestep
                 shiftMode::Bool = false # whether to adjust shift
                 shift::Float64 = 0.0 # starting/current value of shift
                 dτ::Float64 = 0.01 # current value of time step
    )
Parameters for `fciqmc!()`.
"""
@with_kw mutable struct FCIQMCParams
    step::Int = 0 # number of current/starting timestep
    laststep::Int = 50 # number of final timestep
    shiftMode::Bool = false # whether to adjust shift
    shift::Float64 = 0.0 # starting/current value of shift
    dτ::Float64 = 0.01 # time step
end

"""
Abstract type for defining the strategy for updating the time step.
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

   * [`DonTUpdate`](@ref)
   * [`LogUpdate`](@ref)
   * [`DelayedLogUpdate`](@ref)
   * [`LogUpdateAfterTargetWalkers`](@ref)
   * [`DelayedLogUpdateAfterTargetWalkers`](@ref)
"""
abstract type ShiftStrategy end

"""
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.
"""
@with_kw struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
end

"""
    DelayedLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.3, a = 10) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and delay of
`a` steps.
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
"""
@with_kw struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
end

"""
    DelayedLogUpdate(ζ = 0.3, a = 10) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and delay of `a` steps.
"""
@with_kw struct DelayedLogUpdate <: ShiftStrategy
    ζ::Float64 = 0.3 # damping parameter, best left at value of 0.3
    a::Int = 10 # delay for updating shift
end

"`DonTUpdate() <: ShiftStrategy` Don't update the `shift`."
struct DonTUpdate <: ShiftStrategy end

"""
    update_shift(s <: ShiftStrategy, shift, shiftMode, tnorm, pnorm, dτ, step, df)
Update the shift according to strategy `s`.
"""
@inline function update_shift(s::LogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    return shift - s.ζ/dτ * log(tnorm/pnorm), true
end

function update_shift(s::DelayedLogUpdate,
                        shift, shiftMode,
                        tnorm, pnorm, dτ, step, df)
    # return new shift and new shiftMode
    if shift % a == 0 && size(df,1) > a
        prevnorm = df[end-a+1,:norm]
        return shift - s.ζ/(dτ*a) * log(tnorm/prevnorm), true
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

@inline update_shift(::DonTUpdate, shift, args...) = (shift, false)
