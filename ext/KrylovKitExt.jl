module KrylovKitExt

using KrylovKit: KrylovKit, EigSorter, eigsolve
using LinearAlgebra: LinearAlgebra, mul!, ishermitian, issymmetric
using CommonSolve: CommonSolve
using Setfield: Setfield, @set
using NamedTupleTools: NamedTupleTools, delete
using LinearMaps: LinearMap

using Rimu: Rimu, AbstractDVec, AbstractHamiltonian, AbstractOperator, IsDeterministic,
    starting_address, PDVec, DVec, PDWorkingMemory,
    scale!!, working_memory, zerovector, dimension, replace_keys

using Rimu.ExactDiagonalization: MatrixEDSolver, KrylovKitSolver,
    KrylovKitDirectEDSolver,
    LazyDVecs, EDResult, LazyCoefficientVectorsDVecs, build_basis

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

function _prepare_linear_map(
    ham, vec; starting_address=starting_address(ham), basis=nothing, full_basis=false
)
    if issymmetric(ham) && (isnothing(vec) || valtype(vec) <: Real)
        eltype = Float64
    else
        eltype = ComplexF64
    end
    if isnothing(basis) && full_basis
        basis = build_basis(starting_address)
    elseif isnothing(basis)
        basis = build_basis(ham, starting_address; sort=true)
    end
    return LinearMap(ham, basis)
end

function KrylovKit.eigsolve(
    ham::AbstractOperator, vec::Vector, howmany::Int=1, which::U=:LR;
    basis=nothing, starting_address=starting_address(ham), full_basis=true, kwargs...
)
    # Change the type of `vec` to float, if needed.
    v = scale!!(vec, 1.0)
    linmap = _prepare_linear_map(ham, v; basis, starting_address, full_basis)
    return eigsolve(
        linmap, v, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end
function KrylovKit.eigsolve(
    ham::AbstractOperator, howmany::Int=1, which::U=:LR;
    basis=nothing, starting_address=starting_address(ham), full_basis=true, kwargs...
    )
    v = rand(eltype(linmap), size(linmap, 1))
    linmap = _prepare_linear_map(ham, v; basis, starting_address, full_basis)
    return eigsolve(
        linmap, v, howmany, which;
        ishermitian=ishermitian(ham), issymmetric=issymmetric(ham), kwargs...
    )
end

# solve for KrylovKit solvers: prepare arguments for `KrylovKit.eigsolve`
function CommonSolve.solve(s::S; kwargs...
) where {S<:Union{MatrixEDSolver{<:KrylovKitSolver},KrylovKitDirectEDSolver}}
    # combine keyword arguments and set defaults for `howmany` and `which`
    kw_nt = (; howmany = 1, which = :SR, s.kw_nt..., kwargs...)
    # check if universal keyword arguments are present
    if isdefined(kw_nt, :verbose)
        if kw_nt.verbose
            kw_nt = (; kw_nt..., verbosity = 1)
        else
            kw_nt = (; kw_nt..., verbosity = 0)
        end
        kw_nt = delete(kw_nt, (:verbose,))
    end
    kw_nt = replace_keys(kw_nt, (:abstol => :tol, :maxiters => :maxiter))

    # Remove the `howmany` and `which` keys from the kwargs.
    howmany, which = kw_nt.howmany, kw_nt.which
    kw_nt = delete(kw_nt, (:howmany, :which))

    return _kk_eigsolve(s, howmany, which, kw_nt)
end

# solve with KrylovKit and matrix
function _kk_eigsolve(s::MatrixEDSolver{<:KrylovKitSolver}, howmany, which, kw_nt)
    # set up the starting vector
    T = eltype(s.basissetrep.sparse_matrix)
    if isnothing(s.v0)
        x0 = rand(T, dimension(s.basissetrep))
    else
        dvec = DVec(s.v0)
        x0 = [dvec[a] for a in s.basissetrep.basis]
    end
    # solve the problem
    vals, vecs, info = eigsolve(s.basissetrep.sparse_matrix, x0, howmany, which; kw_nt...)
    success = info.converged ≥ howmany

    return EDResult(
        s.algorithm,
        s.problem,
        vals,
        LazyDVecs(vecs, s.basissetrep.basis),
        vecs,
        s.basissetrep.basis,
        info,
        howmany,
        nothing,
        success,
    )
end

# solve with KrylovKit direct
function _kk_eigsolve(s::KrylovKitDirectEDSolver, howmany, which, kw_nt)
    basis = get(kw_nt, :basis, nothing)
    full_basis = get(kw_nt, :full_basis, false)
    kw_nt = delete(kw_nt, (:basis, :full_basis))

    linmap = _prepare_linear_map(s.problem.hamiltonian, s.v0; basis, full_basis)
    if isnothing(s.v0)
        x0 = rand(size(linmap, 1))
    else
        x0 = zeros(eltype(linmap), size(linmap, 1))
        for (k, v) in pairs(s.v0)
            x0[linmap.mapping[k]] = v
        end
    end
    vals, vecs, info = eigsolve(
        linmap, x0, howmany, which;
        issymmetric=issymmetric(linmap), ishermitian=ishermitian(linmap), kw_nt...
    )
    success = info.converged ≥ howmany
    basis = linmap.basis

    return EDResult(
        s.algorithm,
        s.problem,
        vals,
        LazyDVecs(vecs, basis),
        vecs,
        basis,
        info,
        howmany,
        nothing,
        success,
    )
end

end # module
