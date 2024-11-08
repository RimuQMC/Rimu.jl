"""
    ShiftStrategy
Abstract type for defining the strategy for controlling the norm, potentially by updating
the `shift`. Passed as a parameter to [`ProjectorMonteCarloProblem`](@ref) or to
[`FCIQMC`](@ref).

## Implemented strategies:

* [`DontUpdate`](@ref)
* [`DoubleLogUpdate`](@ref) - default in [`ProjectorMonteCarloProblem()`](@ref)
* [`LogUpdate`](@ref)
* [`LogUpdateAfterTargetWalkers`](@ref) - FCIQMC standard
* [`DoubleLogUpdateAfterTargetWalkers`](@ref)

# Extended help

Internally
To implement a custom strategy, define a new subtype of `Rimu.ShiftStrategy` and implement
methods for:
- [`Rimu.update_shift_parameters!`](@ref) - to update the `shift_parameters`
- [`Rimu.initialise_shift_parameters`](@ref) - (optional) to initialise and construct a
    custom implementation of the `shift_parameters`. The default implementation is
    [`Rimu.DefaultShiftParameters`](@ref).
"""
abstract type ShiftStrategy end

"""
    DefaultShiftParameters
Default mutable struct for storing the shift parameters.

See [`ShiftStrategy`](@ref), [`initialise_shift_parameters`](@ref),
[`update_shift_paramters!`](@ref).
"""
mutable struct DefaultShiftParameters{S, N}
    shift::S # for current time step
    pnorm::N # norm from previous time step
    time_step::Float64
    boost::Float64 # boost factor
    counter::Int
    shift_mode::Bool
end

"""
    initialise_shift_parameters(
        s::ShiftStrategy, shift, norm, time_step, boost=1.0, counter=0, shift_mode=false
    ) -> DefaultShiftParameters
Initiatlise a struct to store the shift parameters.

See [`ShiftStrategy`](@ref), [`update_shift_parameters!`](@ref),
[`DefaultShiftParameters`](@ref).
"""
function initialise_shift_parameters(
    ::ShiftStrategy, shift, norm, time_step,
    boost=1.0, counter=0, shift_mode=false
)
    return DefaultShiftParameters(
        shift, norm, Float64(time_step), Float64(boost), counter, shift_mode
    )
end

"""
    update_shift_parameters!(
        s <: ShiftStrategy,
        shift_parameters,
        new_time_step,
        tnorm,
        single_state,
        step
    ) -> shift_stats, proceed
Update the `shift_parameters` according to strategy `s`. See [`ShiftStrategy`](@ref).
Returns a named tuple of the shift statistics and a boolean `proceed` indicating whether
the simulation should proceed.

See [`initialise_shift_parameters`](@ref), [`ShiftStrategy`](@ref),
[`DefaultShiftParameters`](@ref).
"""
update_shift_parameters!

"""
    DontUpdate(; target_walkers = 1_000) <: ShiftStrategy
Don't update the `shift`.  Return when `target_walkers` is reached.

See [`ShiftStrategy`](@ref), [`ProjectorMonteCarloProblem`](@ref).
"""
struct DontUpdate <: ShiftStrategy
    target_walkers::Int
end
function DontUpdate(; targetwalkers = nothing, target_walkers = 1_000)
    if !isnothing(targetwalkers)
        @warn "The keyword targetwalkers is deprecated. Use target_walkers instead."
        target_walkers = targetwalkers
    end
    return DontUpdate(target_walkers)
end


function update_shift_parameters!(s::DontUpdate, sp, new_time_step, tnorm, _...)
    sp.time_step = new_time_step
    return (; shift=sp.shift, norm=tnorm), tnorm < s.target_walkers
end

"""
    LogUpdateAfterTargetWalkers(target_walkers = 1_000, ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift: After `target_walkers` is reached, update the
shift according to the log formula with damping parameter `ζ`.

See [`LogUpdate`](@ref), [`ShiftStrategy`](@ref), [`ProjectorMonteCarloProblem`](@ref).
"""
struct LogUpdateAfterTargetWalkers <: ShiftStrategy
    target_walkers::Int
    ζ::Float64
end
function LogUpdateAfterTargetWalkers(; targetwalkers=nothing, target_walkers = 1_000, ζ = 0.08)
    if !isnothing(targetwalkers)
        @warn "The keyword targetwalkers is deprecated. Use target_walkers instead."
        target_walkers = targetwalkers
    end
    return LogUpdateAfterTargetWalkers(target_walkers, ζ)
end

function update_shift_parameters!(
    s::LogUpdateAfterTargetWalkers, sp, new_time_step, tnorm, _...
)
    @unpack shift, pnorm, time_step, shift_mode = sp
    if shift_mode || real(tnorm) > s.target_walkers
        shift_mode = true
        dτ = time_step
        shift -= s.ζ / dτ * log(tnorm / pnorm)
    end
    pnorm = tnorm
    time_step = new_time_step
    @pack! sp = shift, pnorm, shift_mode, time_step
    return (; shift, norm=tnorm, shift_mode), true
end

"""
    LogUpdate(ζ = 0.08) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)
```

See [`ShiftStrategy`](@ref), [`ProjectorMonteCarloProblem`](@ref).
"""
Base.@kwdef struct LogUpdate <: ShiftStrategy
    ζ::Float64 = 0.08 # damping parameter, best left at value of 0.3
end

function update_shift_parameters!(s::LogUpdate, sp, new_time_step, tnorm, _...)
    @unpack shift, pnorm, time_step = sp
    dτ = time_step
    shift -= s.ζ / dτ * log(tnorm / pnorm)
    pnorm = tnorm
    time_step = new_time_step
    @pack! sp = shift, pnorm, time_step
    return (; shift, norm=tnorm), true
end

"""
    DoubleLogUpdate(; target_walkers = 1_000, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^n}\\right)-\\frac{ξ}{dτ}\\ln\\left(\\frac{\\|Ψ\\|_1^{n+1}}{\\|Ψ\\|_1^\\text{target}}\\right)
```
When ξ = ζ^2/4 this corresponds to critical damping with a damping time scale
T = 2/ζ.

See [`ShiftStrategy`](@ref), [`ProjectorMonteCarloProblem`](@ref).
"""
struct DoubleLogUpdate{T} <: ShiftStrategy
    target_walkers::T
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64  # restoring force to bring walker number to the target
end
function DoubleLogUpdate(;targetwalkers = nothing, target_walkers = 1_000,  ζ = 0.08, ξ = ζ^2/4)
    if !isnothing(targetwalkers)
        @warn "The keyword targetwalkers is deprecated. Use target_walkers instead."
        target_walkers = targetwalkers
    end
    return DoubleLogUpdate(target_walkers, ζ, ξ)
end

function update_shift_parameters!(s::DoubleLogUpdate, sp, new_time_step, tnorm, _...)
    @unpack shift, pnorm, time_step = sp
    dτ = time_step
    shift -= s.ξ / dτ * log(tnorm / s.target_walkers) + s.ζ / dτ * log(tnorm / pnorm)
    pnorm = tnorm
    time_step = new_time_step
    @pack! sp = shift, pnorm, time_step
    return (; shift, norm=tnorm), true
end

"""
    DoubleLogUpdateAfterTargetWalkers(target_walkers = 1_000, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift: After `target_walkers` is reached, update the
shift according to the log formula with damping parameter `ζ` and `ξ`.

See [`DoubleLogUpdate`](@ref), [`ShiftStrategy`](@ref), [`ProjectorMonteCarloProblem`](@ref).
"""
struct DoubleLogUpdateAfterTargetWalkers <: ShiftStrategy
    target_walkers::Int
    ζ::Float64 # damping parameter
    ξ::Float64 # restoring force to bring walker number to the target
end
function DoubleLogUpdateAfterTargetWalkers(;
    targetwalkers=nothing, target_walkers = 1_000, ζ = 0.08, ξ = ζ^2/4
)
    if !isnothing(targetwalkers)
        @warn "The keyword targetwalkers is deprecated. Use target_walkers instead."
        target_walkers = targetwalkers
    end
    return DoubleLogUpdateAfterTargetWalkers(target_walkers, ζ, ξ)
end

function update_shift_parameters!(
    s::DoubleLogUpdateAfterTargetWalkers, sp, new_time_step, tnorm, _...
)
    @unpack shift, pnorm, time_step, shift_mode = sp
    if shift_mode || real(tnorm) > s.target_walkers
        shift_mode = true
        dτ = time_step
        shift -= s.ξ / dτ * log(tnorm / s.target_walkers) + s.ζ / dτ * log(tnorm / pnorm)
    end
    pnorm = tnorm
    time_step = new_time_step
    @pack! sp = shift, pnorm, shift_mode, time_step
    return (; shift, norm=tnorm, shift_mode), true
end

# more experimental strategies from here on:

"""
    DoubleLogSumUpdate(; target_walkers = 1000, ζ = 0.08, ξ = ζ^2/4, α = 1/2) <: ShiftStrategy
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


See [`ShiftStrategy`](@ref), [`ProjectorMonteCarloProblem`](@ref).
"""
struct DoubleLogSumUpdate{T} <: ShiftStrategy
    target_walkers::T
    ζ::Float64 # damping parameter, best left at value of 0.08
    ξ::Float64  # restoring force to bring walker number to the target
    α::Float64  # mixing angle for (1-α)*walkernumber + α*UniformProjector()⋅ψ
end
function DoubleLogSumUpdate(;
    targetwalkers = nothing, target_walkers = 1000,  ζ = 0.08, ξ = ζ^2/4, α = 1/2)
    if !isnothing(targetwalkers)
        @warn "The keyword targetwalkers is deprecated. Use target_walkers instead."
        target_walkers = targetwalkers
    end
    DoubleLogSumUpdate(target_walkers,  ζ, ξ, α)
end

function update_shift_parameters!(
    s::DoubleLogSumUpdate, sp, new_time_step, tnorm, s_state, _...
)
    @unpack shift, pnorm, time_step = sp
    @unpack v, pv = s_state
    dτ = time_step
    tp = DictVectors.UniformProjector() ⋅ v
    pp = DictVectors.UniformProjector() ⋅ pv # could be cached
    twn = (1 - s.α) * tnorm + s.α * tp
    pwn = (1 - s.α) * pnorm + s.α * pp
    # return new shift
    shift -= s.ξ / dτ * log(twn / s.target_walkers) + s.ζ / dτ * log(twn / pwn)
    pnorm = tnorm
    time_step = new_time_step
    @pack! sp = shift, pnorm, time_step
    return (; shift, norm=tnorm, up=tp), true
end


"""
    DoubleLogProjected(; target, projector, ζ = 0.08, ξ = ζ^2/4) <: ShiftStrategy
Strategy for updating the shift according to the log formula with damping
parameter `ζ` and `ξ` after projecting onto `projector`.

```math
S^{n+1} = S^n -\\frac{ζ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{P⋅Ψ^{(n)}}\\right)-\\frac{ξ}{dτ}\\ln\\left(\\frac{P⋅Ψ^{(n+1)}}{\\text{target}}\\right)
```

Note that adjusting the keyword `maxlength` in [`ProjectorMonteCarloProblem`](@ref) is
advised as the default may not be appropriate.

See [`ShiftStrategy`](@ref), [`ProjectorMonteCarloProblem`](@ref).
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

function update_shift_parameters!(
    s::DoubleLogProjected, sp, new_time_step, tnorm, s_state, _...
)
    @unpack shift, pnorm, time_step = sp
    @unpack v, pv = s_state
    dτ = time_step
    tp = s.projector ⋅ v
    pp = s.projector ⋅ pv
    # return new shift
    shift -= s.ξ / dτ * log(tp / s.target) + s.ζ / dτ * log(tp / pp)
    pnorm = tnorm
    time_step = new_time_step
    @pack! sp = shift, pnorm, time_step
    return (; shift, norm=tnorm, tp, pp), true
end
