module ArpackExt

using Arpack: Arpack, eigs
using CommonSolve: CommonSolve, solve
using NamedTupleTools: delete
using LinearAlgebra: norm

using Rimu: Rimu, DVec, replace_keys
using Rimu.ExactDiagonalization: ArpackSolver, MatrixEDSolver,
    LazyDVecs, EDResult

struct ArpackConvergenceInfo
    converged::Int
    numiter::Int
    numops::Int
    residual::Vector{Float64}
end
function Base.show(io::IO, info::ArpackConvergenceInfo)
    print(io, "converged = $(info.converged), ")
    print(io, "numiter = $(info.numiter), ")
    print(io, "numops = $(info.numops), ")
    print(io, "residual norm = ")
    show(io, norm(info.residual))
end

function CommonSolve.solve(s::S; kwargs...
) where {S<:MatrixEDSolver{<:ArpackSolver}}
    # combine keyword arguments and set defaults for `howmany` and `which`
    kw_nt = (; howmany=1, which=:SR, s.kw_nt..., kwargs...)
    # check if universal keyword arguments are present
    kw_nt = replace_keys(kw_nt, (:abstol=>:tol, :maxiters=>:maxiter))
    verbose = get(kw_nt, :verbose, false)
    kw_nt = delete(kw_nt, (:verbose,))

    # Remove the `howmany` key from the kwargs.
    kw_nt = (; nev=kw_nt.howmany, kw_nt..., ritzvec=true)
    kw_nt = delete(kw_nt, (:howmany,))
    howmany = kw_nt.nev

    # set up the starting vector
    v0 = if isnothing(s.v0)
        zeros((0,))
    else
        # convert v0 to a DVec to use it like a dictionary
        dvec = DVec(s.v0)
        [dvec[a] for a in s.basissetrep.basis]
    end
    # solve the problem
    vals, vec_matrix, nconv, niter, nmult, resid = eigs(s.basissetrep.sparse_matrix; v0, kw_nt...)

    verbose && @info "Arpack.eigs: $nconv converged out of $howmany requested eigenvalues,"*
        " $niter iterations," *
        " $nmult matrix vector multiplications, norm of residual = $(norm(resid))"
    success = nconv ≥ howmany
    # vecs = [view(vec_matrix, :, i) for i in 1:length(vals)] # convert to array of vectors
    coefficient_vectors = eachcol(vec_matrix)
    vectors = LazyDVecs(coefficient_vectors, s.basissetrep.basis)
    info = ArpackConvergenceInfo(nconv, niter, nmult, resid)
    if !success
        @warn "Arpack.eigs did not converge for all requested eigenvalues:" *
              " $nconv converged out of $howmany requested value(s)."
    end
    return EDResult(
        s.algorithm,
        s.problem,
        vals,
        vectors,
        coefficient_vectors,
        s.basissetrep.basis,
        info,
        howmany,
        vec_matrix,
        success
    )
end

end # module
