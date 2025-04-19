module ArpackExt

using Arpack: Arpack, eigs
using CommonSolve: CommonSolve, solve
using NamedTupleTools: delete
using LinearAlgebra: norm

using Rimu: Rimu, DVec, replace_keys, split_keys, clean_and_warn_if_others_present
using Rimu.ExactDiagonalization: ArpackSolver, IterativeEDSolver,
    LazyDVecs, EDResult

struct ArpackConvergenceInfo{T}
    converged::Int
    numiter::Int
    numops::Int
    residual::Vector{T}
end
function Base.show(io::IO, info::ArpackConvergenceInfo)
    print(io, "converged = $(info.converged), ")
    print(io, "numiter = $(info.numiter), ")
    print(io, "numops = $(info.numops), ")
    print(io, "residual norm = ")
    show(io, norm(info.residual))
end

function CommonSolve.solve(s::IterativeEDSolver{<:ArpackSolver}; kwargs...)
    # Combine keyword arguments and set defaults for `howmany` and `which` and rename
    # arguments to fit Arpack's interface. Setting `check=2` turns off errors and warnings
    # on non-convergence since we already check for that later on.
    kwargs = (; howmany=1, which=:SR, check=2, s.solver_kwargs..., kwargs...)
    kwargs = replace_keys(kwargs, (:abstol => :tol, :maxiters => :maxiter, :howmany => :nev))

    verbose = get(kwargs, :verbose, false)
    kwargs = delete(kwargs, :verbose)

    arpack_kwargs, rest = split_keys(
        kwargs,
        :nev, :ncv, :which, :tol, :maxiter, :sigma, :ritzvec, :explicittransform, :check
    )
    clean_and_warn_if_others_present(rest, ())

    # set up the starting vector
    v0 = s.initial_vector

    # solve the problem
    eigenvalues, vec_matrix, nconv, niter, nmult, resid = eigs(
        s.linear_map; v0, arpack_kwargs...
    )

    verbose && @info "Arpack.eigs: $nconv converged out of $howmany requested eigenvalues,"*
        " $niter iterations," *
        " $nmult matrix vector multiplications, norm of residual = $(norm(resid))"
    howmany = arpack_kwargs.nev
    success = nconv ≥ howmany

    if success
        coefficient_vectors = eachcol(vec_matrix)
    else
        @warn "Arpack.eigs did not converge for all requested eigenvalues:" *
              " $nconv converged out of $howmany requested value(s)."
        coefficient_vectors = eachcol(vec_matrix[:,1:nconv])
    end

    info = ArpackConvergenceInfo(nconv, niter, nmult, resid)

    return EDResult(
        s.algorithm,
        s.problem,
        eigenvalues,
        LazyDVecs(coefficient_vectors, s.basis),
        coefficient_vectors,
        s.basis,
        info,
        howmany,
        vec_matrix,
        success
    )
end

end # module
