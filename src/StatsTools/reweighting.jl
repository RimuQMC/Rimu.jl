# reweighting functions

"""
    w_exp(shift, h, dτ; E_r = mean(shift), skip = 0)
Compute the weights for reweighting over `h` time steps with reference energy `E_r` from
the exponential formula
```math
w_h^{(n)} = \\prod_{j=1}^h \\exp[-dτ(S^{(q+n-j)}-E_r)] ,
```
where `q = skip`.

See also [`w_lin()`](@ref), [`growth_estimator()`](@ref),
[`mixed_estimator()`](@ref).
"""
@inline function w_exp(shift, h, dτ; E_r = mean(shift), skip = 0)
    T = promote_type(eltype(shift),typeof(E_r))
    len = length(shift)-skip
    accu = Vector{T}(undef, len)
    @inbounds for n in 1:len
        a = zero(T)
        look_back = min(h,skip+n-1)
        @simd ivdep for j in 1:look_back # makes it very fast
            a += shift[skip+n-j] - E_r
        end
        accu[n] = exp(-dτ*a)
    end
    return accu
end

"""
    w_lin(shift, h, dτ; E_r = mean(shift), skip = 0)
Compute the weights for reweighting over `h` time steps with reference energy `E_r` from
the linearised formula
```math
w_h^{(n)} = \\prod_{j=1}^h [1-dτ(S^{(q+n-j)}-E_r)] ,
```
where `q = skip`.

See also [`w_exp()`](@ref), [`growth_estimator()`](@ref),
[`mixed_estimator()`](@ref).
"""
@inline function w_lin(shift, h, dτ; E_r = mean(shift), skip = 0)
    T = promote_type(eltype(shift),typeof(E_r))
    len = length(shift)-skip
    accu = ones(T, len)
    @inbounds for n in 1:len
        a = one(T)
        look_back = min(h,skip+n-1)
        @simd ivdep for j in 1:look_back
            a *= 1 - dτ*(shift[skip+n-j] - E_r)
        end
        accu[n] = a
    end
    return accu
end

"""
    growth_estimator(
        shift, wn, h, dτ;
        skip = 0,
        E_r = mean(shift[skip+1:end]),
        weights = w_exp,
        change_type = identity,
        kwargs...,
    ) -> r::RatioBlockingResult
    growth_estimator(df::DataFrame, h; kwargs ...)
Compute the growth estimator with reference energy `E_r` by the reweighting
technique described in [Umirgar *et al.* (1993)](http://dx.doi.org/10.1063/1.465195),
see Eq. (20).
`shift` and `wn` are equal length
vectors containing the shift and walker number time series, respectively.
Reweighting is done over `h`
time steps and `length(shift) - skip` time steps are used for the blocking analysis done
with [`ratio_of_means()`](@ref). `dτ` is the time step and `weights` a function that
calulates the weights. See [`w_exp()`](@ref) and [`w_lin()`](@ref).
```math
E_{gr} = E_r - \\frac{1}{dτ}\\ln
    \\frac{\\sum_n w_{h+1}^{(n+1)} N_\\mathrm{w}^{(n+1)}}
        {\\sum_m w_{h}^{(m)} N_\\mathrm{w}^{(m)}}
```
When `h` is greater than the autocorrelation time scale of the `shift`,
then `E_gr` (returned as `r.ratio`) is an unbiased but approximate estimator for the ground
state energy ``E_0`` with an error ``\\mathcal{O}(dτ^2)`` and potentially increased
confidence intervals compared to the (biased) shift estimator.
Error propagation is done with [`MonteCarloMeasurements`](@ref). Progagation through the
logarithm can be modified by setting `change_type` to [`to_measurement`](@ref) in order
to avoid `NaN` results from negative outliers.

If `success==true` the
blocking analysis was successful in `k-1` steps, using `blocks` uncorrelated data points.

See also [`mixed_estimator()`](@ref) and [`RatioBlockingResult`](@ref).
"""
function growth_estimator(
    shift, wn, h, dτ;
    skip = 0,
    E_r = mean(shift[skip+1:end]),
    weights = w_exp,
    change_type = identity,
    mc_samples = nothing,
    kwargs...,
)
    T = promote_type(eltype(shift), eltype(wn))
    # W_{t+1}^{(n+1)} .* wn^{(n+1)}
    numerator = weights(shift[2:end], h+1, dτ; E_r, skip) .* wn[skip+2:end]
    # W_{t}^{(n)} .* wn^{(n)}
    denominator = weights(shift[1:end-1], h, dτ; E_r, skip) .* wn[skip+1:end-1]
    rbr = ratio_of_means(numerator, denominator; mc_samples, kwargs...)
    r = rbr.ratio::MonteCarloMeasurements.Particles{T,<:Any}
    r = change_type(r)
    E_gr = E_r - log(r)/dτ # MonteCarloMeasurements propagates the uncertainty
    E_gr_f = E_r - log(Measurements.measurement(rbr.f, rbr.σ_f))/dτ # linear error prop
    return RatioBlockingResult(
        particles(mc_samples, E_gr),
        Measurements.value(E_gr_f),
        Measurements.uncertainty(E_gr_f),
        rbr.δ_y,
        rbr.k,
        rbr.blocks,
        rbr.success
    )
    # return (; E_gr, k=rbr.k, blocks = rbr.blocks, success = rbr.success)
end
function growth_estimator(df::DataFrame, h; kwargs...)
    return growth_estimator(df.shift, df.norm, h, df.dτ[end]; kwargs...)
end

"""
    mixed_estimator(
        hproj, vproj, shift, h, dτ;
        skip = 0,
        E_r = mean(shift[skip+1:end]),
        weights = w_exp,
        kwargs...,
    ) -> r::RatioBlockingResult
    mixed_estimator(df::DataFrame, h; kwargs...)
Compute the mixed estimator by the reweighting
technique described in [Umirgar *et al.* (1993)](http://dx.doi.org/10.1063/1.465195),
Eq. (19)
```math
E_\\mathrm{mix} = \\frac{\\sum_n w_{h}^{(n)}  (Ĥ'\\mathrm{v})⋅\\mathrm{c}^{(n)}}
        {\\sum_m w_{h}^{(m)}  \\mathrm{v}⋅\\mathrm{c}^{(m)}} ,
```
where the time series `hproj ==` ``(Ĥ'\\mathrm{v})⋅\\mathrm{c}^{(n)}`` and
`vproj ==` ``\\mathrm{v}⋅\\mathrm{c}^{(m)}`` have the same length as `shift`
(See [`ProjectedEnergy`](@ref) on how to set these up).
Reweighting is done over `h`
time steps and `length(shift) - skip` time steps are used for the blocking analysis done
with [`ratio_of_means()`](@ref). `dτ` is the time step and `weights` a function that
calulates the weights. See [`w_exp()`](@ref) and [`w_lin()`](@ref).
Additional keyword arguments are passed on to [`ratio_of_means()`](@ref).

When `h` is greater than the autocorrelation time scale of the `shift`,
then `r.ratio` is an unbiased but approximate estimator for the ground state energy
``E_0`` with an error ``\\mathcal{O}(dτ^2)`` and potentially increased confidence intervals
compared to the unweighted ratio.
Error propagation is done with [`MonteCarloMeasurements`](@ref).
Results are returned as [`RatioBlockingResult`](@ref).

See also [`growth_estimator()`](@ref).
"""
function mixed_estimator(
    hproj, vproj, shift, h, dτ;
    skip = 0,
    E_r = mean(shift[skip+1:end]),
    weights = w_exp,
    kwargs...,
)
    num = weights(shift, h, dτ; E_r, skip) .* hproj[skip+1:end]
    denom = weights(shift, h, dτ; E_r, skip) .* vproj[skip+1:end]
    rbs = ratio_of_means(num, denom; kwargs...)
    return rbs
end

function mixed_estimator(df::DataFrame, h; kwargs...)
    return mixed_estimator(df.hproj, df.vproj, df.shift, h, df.dτ[end]; kwargs...)
end

"""
    projected_energy(df::DataFrame; skip=0, kwargs...) -> r::RatioBlockingResult
Compute the projected energy estimator
```math
E_\\mathrm{p} = \\frac{\\sum_n  \\mathrm{v}⋅Ĥ\\mathrm{c}^{(n)}}
        {\\sum_m \\mathrm{v}⋅\\mathrm{c}^{(m)}} ,
```
where the time series `df.hproj ==` ``\\mathrm{v}⋅Ĥ\\mathrm{c}^{(n)}`` and
`df.vproj ==` ``\\mathrm{v}⋅\\mathrm{c}^{(m)}`` are taken from `df`, skipping the first
`skip` entries (use `post_step = `[`ProjectedEnergy()`](@ref) to set these up in
[`lomc!()`](@ref)).

`projected_energy` is equivalent to [`mixed_estimator`](@ref) with `h=0`. `kwargs` are
passed on to [`ratio_of_means`](@ref).

Returns a [`RatioBlockingResult`](@ref).
See [`to_nt`](@ref), [`val_and_errs`](@ref), [`val`](@ref), [`errs`](@ref) for processing
results.
"""
projected_energy(df::DataFrame; kwargs...) = mixed_estimator(df, 0; kwargs...)

"""
    shift_estimator(df::DataFrame; kwargs...) -> r::BlockingResult
Return the shift estimator from the data in `df.shift`. Keyword arguments are passed on to
[`blocking_analysis`](@ref). Returns a [`BlockingResult`](@ref).

See also [`growth_estimator`](@ref), [`projected_energy`](@ref).
"""
function shift_estimator(df::DataFrame; kwargs...)
    return blocking_analysis(df.shift; kwargs...)
end
