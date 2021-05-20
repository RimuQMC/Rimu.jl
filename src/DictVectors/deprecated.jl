# This file should be deleted at some point and is only here to ease the transition to the new
# version.
export copytight, DVec2, capacity, IsStochastic

function copytight(args...; kwargs...)
    @warn "`copytight` is deprecated. Use `copy`" maxlog=1
    return copy(args...; kwargs...)
end

function DVec2(args...; kwargs...)
    @warn "`DVec2` is deprecated. Use `DVec`" maxlog=1
    return DVec(args...; kwargs...)
end

function capacity(args...; kwargs...)
    error("`capacity` has been removed")
end

function IsStochastic()
    @warn "`IsStochastic` has been renamed to `IsStochasticInteger`." maxlog=1
    return IsStochasticInteger()
end
