"""
    TimeStepStrategy

Abstract type for strategies for updating the time step with
[`update_time_step()`](@ref). Implemented strategies:

* [`ConstantTimeStep`](@ref)
* [`AdaptiveTimeStep`](@ref)

See also [`FCIQMC`](@ref).
"""
abstract type TimeStepStrategy end

"""
    ConstantTimeStep() <: TimeStepStrategy

Keep the `time_step` constant.

See also [`TimeStepStrategy`](@ref), [`FCIQMC`](@ref).
"""
struct ConstantTimeStep <: TimeStepStrategy end

"""
    update_time_step(s<:TimeStepStrategy, time_step, deaths, clones, zombies, tnorm, len)
    -> new_time_step
Update the time step according to the strategy `s`.

See also [`TimeStepStrategy`](@ref).
"""
update_time_step(::ConstantTimeStep, time_step, args...) = time_step

"""
    AdaptiveTimeStep(; damp_zombies=0.9, grow=1.01) <: TimeStepStrategy

Adapt the time step to avoid zombies.

## Parameters
* `damp_zombies`: factor by which to decrease the time step for each zombie.
* `grow`: factor by which to increase the time step when there are no zombies.

See also [`TimeStepStrategy`](@ref), [`FCIQMC`](@ref).
"""
struct AdaptiveTimeStep <: TimeStepStrategy
    damp_zombies::Float64
    grow::Float64
end
function AdaptiveTimeStep(; damp_zombies=0.9, grow=1.01)
    return AdaptiveTimeStep(damp_zombies, grow)
end

function update_time_step(s::AdaptiveTimeStep, time_step, _, _, zombies, _...)
    if  zombies > 0
        return time_step * s.damp_zombies^zombies # decrease time step
    else
        return time_step * s.grow # increase time step
    end
end
