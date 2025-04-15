module IterativeSolversExt

using IterativeSolvers: IterativeSolvers, lobpcg, LOBPCGResults
using CommonSolve: CommonSolve, solve
using NamedTupleTools: delete
using Rimu: Rimu, DVec, replace_keys, split_keys, clean_and_warn_if_others_present
using Rimu.ExactDiagonalization: IterativeEDSolver, LOBPCGSolver,
    LazyDVecs, EDResult

struct LOBPCGConvergenceInfo
    tolerance::Float64
    iterations::Int
    converged::Bool
    maxiter::Int
    residual_norms::Vector{Float64}
end
function Base.show(io::IO, info::LOBPCGConvergenceInfo)
    print(io, "tolerance = $(info.tolerance), ")
    print(io, "iterations = $(info.iterations), ")
    print(io, "converged = $(info.converged), ")
    print(io, "maxiter = $(info.maxiter), ")
    print(io, "residual_norms ≤ ")
    show(io, maximum(info.residual_norms))
end

function CommonSolve.solve(s::IterativeEDSolver{<:LOBPCGSolver}; kwargs...)
    # Combine keyword arguments and set defaults for `howmany` and `which` and split out
    # the arguments the are accepted by `lobpcg`
    kwargs = (; howmany=1, which=:SR, s.solver_kwargs..., kwargs...)
    kwargs = replace_keys(
        kwargs, (:abstol => :tol, :maxiters => :maxiter, :howmany => :nev)
    )
    lobpcg_kwargs, rest = split_keys(kwargs, :log, :P, :C, :maxiter, :tol)

    verbose = get(rest, :verbose, false)
    rest = delete(rest, :verbose)

    # Check that only `which` and `nev` remain in `rest` and extract them.
    (; nev, which) = clean_and_warn_if_others_present(
        rest, (:nev, :which)
    )
    if which == :SR
        largest = false
    elseif which == :LR
        largest = true
    else
        throw(ArgumentError("unsupported `which` argument! Only `:SR` and `:LR` are supported."))
    end

    results = lobpcg(s.linear_map, largest, nev; lobpcg_kwargs...)
    success = results.converged

    if success
        verbose && @info "IterativeSolvers.lobpcg: $nev requested eigenvalue(s) converged in $(results.iterations) iterations, norm(s) of residuals = $(results.residual_norms)"
    else
        @warn "IterativeSolvers.lobpcg did not converge for all requested eigenvalues."
    end

    coefficient_vectors = eachcol(results.X)
    info = LOBPCGConvergenceInfo(
        results.tolerance,
        results.iterations,
        results.converged,
        results.maxiter,
        results.residual_norms
    )
    # create the result object
    return EDResult(
        s.algorithm,
        s.problem,
        results.λ,
        LazyDVecs(coefficient_vectors, s.basis),
        coefficient_vectors,
        s.basis,
        info,
        nev,
        results,
        success
    )
end

end # module
