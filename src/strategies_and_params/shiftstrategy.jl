"""
Abstract type for defining the strategy for controlling the norm, potentially by updating
the `shift`. The concrete types are typically stateful and store the necessary information.
Passed as a parameter to [`lomc!`](@ref).

## Implemented strategies:

* [`DontUpdate`](@ref)
* [`DoubleLogUpdate`](@ref) - default in [`lomc!()`](@ref)
* [`LogUpdate`](@ref)
* [`LogUpdateAfterTargetWalkers`](@ref) - FCIQMC standard
* [`DoubleLogUpdateAfterTargetWalkers`](@ref)
"""
abstract type ShiftStrategy end

"""
    DefaultShiftParameters
Default mutable struct for storing the shift parameters.

See [`shift_parameters`](@ref).
"""
mutable struct DefaultShiftParameters{S, N}
    shift::S # for current time step
    pnorm::N # norm from previous time step
    time_step::Float64
    counter::Int
    shift_mode::Bool
end

"""
    initialise_shift_parameters(s::ShiftStrategy, shift, norm, time_step, counter=0, shift_mode=false)
Initiatlise a struct to store the shift parameters.
"""
function initialise_shift_parameters(
    ::ShiftStrategy, shift, norm, time_step,
    counter=0, shift_mode=false
)
    return DefaultShiftParameters(shift, norm, time_step, counter, shift_mode)
end

"""
    update_shift!(
        s <: ShiftStrategy,
        shift_parameters,
        tnorm,
        v_new,
        v_old,
        step,
        report
    ) -> shift_stats, proceed
Update the `shift_parameters` according to strategy `s`. See [`ShiftStrategy`](@ref).
Returns a named tuple of the shift statistics and a boolean `proceed` indicating whether
the simulation should proceed.
"""
update_shift!

"""
    DontUpdate(; targetwalkers = 1_000_000) <: ShiftStrategy
Don't update the `shift`.  Return when `targetwalkers` is reached.

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct DontUpdate <: ShiftStrategy
    targetwalkers::Int = 1_000_000
end

function update_shift!(s::DontUpdate, sp, tnorm, _...)
    return (; shift=sp.shift, norm=tnorm), tnorm < s.targetwalkers
end

"""
    LogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.

See [`LogUpdate`](@ref), [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end

function update_shift!(s::LogUpdateAfterTargetWalkers, sp, tnorm, _...)
    @unpack shift, pnorm, time_step, shift_mode = sp
    if shift_mode || real(tnorm) > s.targetwalkers
        shift_mode = true
        dτ = time_step
        shift -= s.ζ / dτ * log(tnorm / pnorm)
    end
    pnorm = tnorm
    @pack! sp = shift, pnorm, shift_mode
    return (; shift, norm=tnorm, shift_mode), true
end

"""
    LogUpdate(ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)
```

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end

function update_shift!(s::LogUpdate, sp, tnorm, _...)
    @unpack shift, pnorm, time_step = sp
    dτ = time_step
    shift -= s.ζ / dτ * log(tnorm / pnorm)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm), true
end

"""
    DoubleLogUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
When ξ = ζ^2/4 this corresponds to critical damping with a damping time scale
T = 2/ζ.

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
struct DoubleLogUpdate{T} <: ShiftStrategy
    targetwalkers::T
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64  # restoring force to bring walker number to the target
end
function DoubleLogUpdate(;targetwalkers = 1000,  ζ = 0.08, ξ = ζ^2/4)
    return DoubleLogUpdate(targetwalkers, ζ, ξ)
end

function update_shift!(s::DoubleLogUpdate, sp, tnorm, _...)
    @unpack shift, pnorm, time_step = sp
    dτ = time_step
    shift -= s.ξ / dτ * log(tnorm / s.targetwalkers) + s.ζ / dτ * log(tnorm / pnorm)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm), true
end

"""
    DoubleLogUpdateAfterTargetWalkers(targetwalkers, ζ = 0.08, ξ = 0.0016) <: ShiftStrategy
Strategy for updating the shift: After `targetwalkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`.

See [`DoubleLogUpdate`](@ref), [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
Base.@kwdef struct DoubleLogUpdateAfterTargetWalkers <: ShiftStrategy
    targetwalkers::Int
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
    ξ::Float64 = 0.0016 # restoring force to bring walker number to the target
end

function update_shift!(s::DoubleLogUpdateAfterTargetWalkers, sp, tnorm, _...)
    @unpack shift, pnorm, time_step, shift_mode = sp
    if shift_mode || real(tnorm) > s.targetwalkers
        shift_mode = true
        dτ = time_step
        shift -= s.ξ / dτ * log(tnorm / s.targetwalkers) + s.ζ / dτ * log(tnorm / pnorm)
    end
    pnorm = tnorm
    @pack! sp = shift, pnorm, shift_mode
    return (; shift, norm=tnorm, shift_mode), true
end

# more experimental strategies from here on:

"""
    DoubleLogSumUpdate(; targetwalkers = 1000, ζ = 0.08, ξ = ζ^2/4, α = 1/2) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameters `ζ` and `ξ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^n}\\right)
- \\frac{ξ}{dτ}\\ln\\left(\\frac{N_\\mathrm{w}^{n+1}}{N_\\mathrm{w}^\\text{target}}\\right),
```
where ``N_\\mathrm{w} =`` `(1-α)*walkernumber() + α*UniformProjector()⋅ψ` computed with
[`walkernumber()`](@ref) and [`UniformProjector()`](@ref).
When ξ = ζ^2/4 this corresponds to critical damping with a damping time scale
T = 2/ζ.


See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
struct DoubleLogSumUpdate{T} <: ShiftStrategy
    targetwalkers::T
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64  # restoring force to bring walker number to the target
    α::Float64  # mixing angle for (1-α)*walkernumber + α*UniformProjector()⋅ψ
end
function DoubleLogSumUpdate(;targetwalkers = 1000,  ζ = 0.08, ξ = ζ^2/4, α = 1/2)
    DoubleLogSumUpdate(targetwalkers,  ζ, ξ, α)
end

function update_shift!(s::DoubleLogSumUpdate, sp, tnorm, v_new, v_old, _...)
    @unpack shift, pnorm, time_step = sp
    dτ = time_step
    tp = DictVectors.UniformProjector() ⋅ v_new
    pp = DictVectors.UniformProjector() ⋅ v_old # could be cached
    twn = (1 - s.α) * tnorm + s.α * tp
    pwn = (1 - s.α) * pnorm + s.α * pp
    # return new shift
    shift -= s.ξ / dτ * log(twn / s.targetwalkers) + s.ζ / dτ * log(twn / pwn)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm, up=tp), true
end


"""
    DoubleLogProjected(; target, projector, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` after projecting onto `projector`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{P⋅Ψ^{(n)}}\\right)-\\frac{ξ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{\\text{target}}\\right)
```

Note that adjusting the keyword `maxlength` in [`lomc!`](@ref) is advised as the
default may not be appropriate.

See [`ShiftStrategy`](@ref), [`lomc!`](@ref).
"""
struct DoubleLogProjected{T,P} <: ShiftStrategy
    target::T
    projector::P
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64 # restoring force to bring walker number to the target
end
function DoubleLogProjected(; target, projector, ζ = 0.08, ξ = ζ^2/4)
    return DoubleLogProjected(target, freeze(projector), ζ, ξ)
end

function update_shift!(s::DoubleLogProjected, sp, tnorm, v_new, v_old, _...)
    @unpack shift, pnorm, time_step = sp
    dτ = time_step
    tp = s.projector ⋅ v_new
    pp = s.projector ⋅ v_old
    # return new shift
    shift -= s.ξ / dτ * log(tp / s.target) + s.ζ / dτ * log(tp / pp)
    pnorm = tnorm
    @pack! sp = shift, pnorm
    return (; shift, norm=tnorm, tp, pp), true
end
