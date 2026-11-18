module KrylovKitExt

using KrylovKit: KrylovKit, EigSorter, eigsolve
using LinearAlgebra: LinearAlgebra, mul!, ishermitian, issymmetric
using CommonSolve: CommonSolve
using NamedTupleTools: delete
using LinearMaps: LinearMap

using Rimu: Rimu, AbstractDVec, AbstractOperator, IsDeterministic,
    starting_address, PDVec, DVec,
    scale!!, zerovector, replace_keys, split_keys,
    clean_and_warn_if_others_present

using Rimu.DictVectors: PDWorkingMemory

using Rimu.ExactDiagonalization: IterativeEDSolver, KrylovKitSolver,
    LazyDVecs, EDResult, LazyCoefficientVectorsDVecs

const U = Union{Symbol,EigSorter}

"""
    OperatorWithWorkingMemory

A struct that holds the working memory for repeatedly multiplying vectors with an operator.
"""
struct OperatorWithWorkingMemory{H,W<:PDWorkingMemory}
    hamiltonian::H
    working_memory::W
end
function OperatorWithWorkingMemory(hamiltonian, vector::PDVec)
    return OperatorWithWorkingMemory(hamiltonian, PDWorkingMemory(vector; style=IsDeterministic()))
end

function (o::OperatorWithWorkingMemory)(v)
    result = zerovector(v)
    return mul!(result, o.hamiltonian, v, o.working_memory)
end

function KrylovKit.eigsolve(
    ham::AbstractOperator, vec::PDVec, howmany::Int=1, which::U=:LR; kwargs...
)
    # Change the type of `vec` to float, if needed.
    v = scale!!(vec, 1.0)
    op = OperatorWithWorkingMemory(ham, v)
    return eigsolve(
        op, v, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end

# This method only exists to detect whether a Hamiltonian is Hermitian or not.
function KrylovKit.eigsolve(
    ham::AbstractOperator, vec::AbstractDVec, howmany::Int=1, which::U=:LR; kwargs...
)
    return @invoke eigsolve(
        ham::Any, vec::Any, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end

function KrylovKit.eigsolve(
    ham::AbstractOperator, vec::Vector, howmany::Int=1, which::U=:LR;
    basis=nothing, starting_address=starting_address(ham), full_basis=true, kwargs...
)
    # Change the type of `vec` to float, if needed.
    v = scale!!(vec, 1.0)
    linmap = LinearMap(ham; basis, starting_address, full_basis)
    return eigsolve(
        linmap, v, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end
function KrylovKit.eigsolve(
    ham::AbstractOperator, howmany::Int=1, which::U=:LR;
    basis=nothing, starting_address=starting_address(ham), full_basis=true, kwargs...
    )
    linmap = LinearMap(ham; basis, starting_address, full_basis)
    v = rand(eltype(linmap), size(linmap, 1))
    return eigsolve(
        linmap, v, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end

# solve for KrylovKit solvers: prepare arguments for `KrylovKit.eigsolve`
function CommonSolve.solve(s::IterativeEDSolver{<:KrylovKitSolver}; kwargs...)
    # Combine keyword arguments and set defaults for `howmany` and `which`
    kwargs = (; howmany=1, which=:SR, s.solver_kwargs..., kwargs...)
    kwargs = replace_keys(kwargs, (:abstol => :tol, :maxiters => :maxiter))

    # Set verbosity - added at the beginning so it can stille be manually set to 2 or 4 by
    # the user.
    if get(kwargs, :verbose, false)
        kwargs = (; verbosity=3, kwargs...)
    else
        kwargs = (; verbosity=0, kwargs...)
    end
    kwargs = delete(kwargs, :verbose)

    # Split kwargs into ones passed to KrylovKit and the rest. Add information regarding
    # hermiticity.
    kk_kwargs, rest = split_keys(
        kwargs, :tol, :maxiter, :krylovdim, :orth, :eager, :verbosity
    )
    kk_kwargs = (
        ; ishermitian=ishermitian(s.linear_map), issymmetric=issymmetric(s.linear_map),
        kk_kwargs...
    )

    # Check for unused arguments and extract the `howmany` and `which` keys.
    (; howmany, which) = clean_and_warn_if_others_present(rest, (:howmany, :which))

    eigenvalues, coefficient_vectors, info = eigsolve(
        s.linear_map, s.initial_vector, howmany, which; kk_kwargs...
    )
    success = info.converged ≥ howmany

    if !success
        @warn "KrylovKit.eigsolve did not converge for all requested eigenvalues:" *
              " $(info.converged) converged out of $howmany requested value(s)."
    end

    return EDResult(
        s.algorithm,
        s.problem,
        eigenvalues,
        LazyDVecs(coefficient_vectors, s.basis),
        coefficient_vectors,
        s.basis,
        info,
        howmany,
        nothing,
        success,
    )
end

end # module
