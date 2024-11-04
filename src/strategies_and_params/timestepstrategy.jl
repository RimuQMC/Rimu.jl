"""
    TimeStepStrategy

Abstract type for strategies for updating the time step with
[`update_time_step()`](@ref). Implemented strategies:

   * [`ConstantTimeStep`](@ref)
"""
abstract type TimeStepStrategy end

"""
    ConstantTimeStep <: TimeStepStrategy

Keep `time_step` constant.
"""
struct ConstantTimeStep <: TimeStepStrategy end

"""
    update_time_step(s<:TimeStepStrategy, time_step, deaths, clones, zombies, tnorm, len)
    -> new_time_step
Update the time step according to the strategy `s`.
"""
update_time_step(::ConstantTimeStep, time_step, args...) = time_step

"""
    AdaptiveTimeStep <: TimeStepStrategy

Adapt the time step to avoid zombies.
"""
struct AdaptiveTimeStep <: TimeStepStrategy end

function update_time_step(::AdaptiveTimeStep, time_step, _, _, zombies, _...)
    if  zombies > 0
        return time_step * 0.9^zombies
    else
        return time_step * 1.01 # increase by 1%
    end
end
