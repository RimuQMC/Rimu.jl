"""
    variational_energy_estimator(shifts, overlaps; kwargs...)
    variational_energy_estimator(df::DataFrame; max_replicas=:all, spectral_state=1, kwargs...)
    variational_energy_estimator(sim::PMCSimulation; kwargs...)
    -> r::RatioBlockingResult

Compute the variational energy estimator from the replica time series of the `shifts` and
coefficient vector `overlaps` by blocking analysis.
The keyword argument `max_replicas` can be used to constrain the number of replicas
processed to be smaller than all available in `df`.
The keyword argument `spectral_state` determines which spectral state's energy is computed.
Other keyword arguments are passed on to [`ratio_of_means()`](@ref).
Returns a [`RatioBlockingResult`](@ref).

An estimator for the variational energy
```math
\\frac{⟨\\mathbf{c}⟩^† \\mathbf{H}⟨\\mathbf{c}⟩}{⟨\\mathbf{c}⟩^†⟨\\mathbf{c}⟩}
```
is calculated from
```math
Ē_{v}  =  \\frac{\\sum_{a<b}^R \\overline{(S_a+S_b) \\mathbf{c}_a^† \\mathbf{c}_b}}
               {2\\sum_{a<b}^R \\overline{\\mathbf{c}_a^† \\mathbf{c}_b}} ,
```
where the sum goes over distinct pairs out of the ``R`` replicas. See
[arXiv:2103.07800](http://arxiv.org/abs/2103.07800).

The `DataFrame` and [`PMCSimulation`](@ref Main.Rimu.PMCSimulation) versions can extract
the relevant information from the result of
[`solve`](@ref CommonSolve.solve(::ProjectorMonteCarloProblem)).
Set up the [`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem) with the
keyword argument `replica_strategy = AllOverlaps(R)` and `R ≥ 2`. If passing `shifts` and
`overlaps`, the data has to be arranged in the correct order (as provided in the `DataFrame`
version).

See [`AllOverlaps`](@ref Main.AllOverlaps).
"""
function variational_energy_estimator(shifts, overlaps; kwargs...)
    num_replicas = length(shifts)
    if length(overlaps) ≠ binomial(num_replicas, 2)
        throw(ArgumentError(
            "The number of `overlaps` needs to be `binomial(length(shifts),2)`."
        ))
    end
    denominator = sum(overlaps)
    numerator = zero(shifts[1])
    count_overlaps = 0
    for i in 1:num_replicas, j in i+1:num_replicas
        count_overlaps += 1
        @. numerator += 1 / 2 * (shifts[i] + shifts[j]) * overlaps[count_overlaps]
    end
    return ratio_of_means(numerator, denominator; kwargs...)
end

function variational_energy_estimator(sim; max_replicas=:all, spectral_state=1, kwargs...)
    df = DataFrame(sim)
    num_reps = num_replicas(df)
    if num_reps == 1
        throw(ArgumentError(
            "No replicas found. Use keyword \
            `replica_strategy=AllOverlaps(n)` with n≥2 in `ProjectorMonteCarloProblem` to set up replicas!"
        ))
    end
    @assert num_reps ≥ 2 "At least two replicas are needed, found $num_replicas"

    num_olaps = num_overlaps(df)
    if num_olaps == 0
        throw(ArgumentError(
            "No overlaps found. Use keyword \
            `replica_strategy=AllOverlaps(n)` with n≥2 in `ProjectorMonteCarloProblem` to set up replicas!"
        ))
    end
    @assert num_olaps == binomial(num_reps, 2) "Unexpected number of overlaps."

    # process at most `max_replicas` but at least 2 replicas
    if max_replicas isa Integer
        num_reps = max(2, min(max_replicas, num_reps))
    end

    shiftnames = [Symbol("shift_r$(i)s$(spectral_state)") for i in 1:num_reps]
    shifts = map(name -> getproperty(df, name), shiftnames)
    @assert length(shifts) == num_reps

    overlap_names = [
        Symbol("r$(i)s$(spectral_state)_dot_r$(j)s$(spectral_state)") for i in 1:num_reps for j in i+1:num_reps
    ]
    overlaps = map(name -> getproperty(df, name), overlap_names)
    @assert length(overlaps) ≤ num_olaps

    return variational_energy_estimator(shifts, overlaps; kwargs...)
end
