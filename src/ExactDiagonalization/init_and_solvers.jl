"""
    init(p::ExactDiagonalizationProblem, [algorithm]; kwargs...)

Initialize a solver for an [`ExactDiagonalizationProblem`](@ref) `p` with an optional
`algorithm`. Returns a solver instance that can be solved with
[`solve`](@ref solve(::ExactDiagonalizationProblem)).

For a description of the keyword arguments, see the documentation for
[`ExactDiagonalizationProblem`](@ref).
"""
function CommonSolve.init( # no algorithm specified as positional argument
    prob::ExactDiagonalizationProblem;
    kwargs...
)
    kwargs = (; prob.kwargs..., kwargs...) # remove duplicates
    algorithm = get(kwargs, :algorithm, LinearAlgebraSolver())
    delete(kwargs, :algorithm)
    new_prob = ExactDiagonalizationProblem(prob.hamiltonian, prob.initial_vector; kwargs...)
    return init(new_prob, algorithm)
end

# TODO since this is the same for all solvers, maybe move it to ExactDiagonalizationProblem?
function _set_up_starting_address(v0, hamiltonian)
    if isnothing(v0)
        addr_or_vec = starting_address(hamiltonian)
    elseif v0 isa AbstractFockAddress || v0 isa Vector{<:AbstractFockAddress}
        addr_or_vec = v0
    elseif v0 isa FrozenDVec
        addr_or_vec = keys(v0)
    else
        throw(ArgumentError("Invalid starting vector in `ExactDiagonalizationProblem`."))
    end

    return addr_or_vec
end
function _set_up_initial_vector(v0, basis)
    if isnothing(v0)
        return rand(length(basis))
    end

    if v0 isa Union{
        NTuple{<:Any,<:AbstractFockAddress},
        AbstractVector{<:AbstractFockAddress}
    }
        v0_dvec = Dict(addr => 1.0 for addr in v0)
    elseif v0 isa AbstractFockAddress
        v0_dvec = Dict(v0 => 1.0)
    elseif v0 isa DictVectors.FrozenDVec{<:AbstractFockAddress}
        v0_dvec = Dict(pairs(v0))
    else
        throw(ArgumentError("Invalid starting vector in `ExactDiagonalizationProblem`."))
    end

    return [get(v0_dvec, b, zero(valtype(v0_dvec))) for b in basis]
end

struct IterativeEDSolver{A,P,LM,T<:Number,F}
    algorithm::A
    problem::P
    linear_map::LM
    initial_vector::Vector{T}
    basis::Vector{F}
    solver_kwargs::NamedTuple
end
function Base.show(io::IO, s::IterativeEDSolver)
    io = IOContext(io, :compact => true)
    print(io, "IterativeEDSolver\n with algorithm $(s.algorithm) for hamiltonian = ")
    show(io, s.problem.hamiltonian)
    print(io, ",\n  kwargs = ")
    show(io, s.solver_kwargs)
    print(io, "\n)")
end

function CommonSolve.init(
    prob::ExactDiagonalizationProblem, algorithm::AbstractAlgorithm{true}; kwargs...
)
    !ishermitian(prob.hamiltonian) && algorithm isa LOBPCGSolver &&
        @warn "LOBPCGSolver() is not suitable for non-hermitian matrices."

    # Merge keyword arguments from problem, algorithm and ones passed to this function
    # and split them into sets that are passed to LinearMap and ones that
    # are left for the solver.

    kwargs = (; prob.kwargs..., algorithm.kwargs..., kwargs...)
    linmap_kwargs, solver_kwargs = extract_and_delete_keys(kwargs, :basis, :full_basis)

    # determine the starting address or vector and seed address to build the matrix from
    addr_or_vec = _set_up_starting_address(
        prob.initial_vector, prob.hamiltonian
    )

    # create the LinearMap
    linmap = LinearMap(prob.hamiltonian; starting_address=addr_or_vec, linmap_kwargs...)
    basis = linmap.basis

    initial_vector = _set_up_initial_vector(prob.initial_vector, basis)

    return IterativeEDSolver(algorithm, prob, linmap, initial_vector, basis, solver_kwargs)
end
function CommonSolve.init(
    prob::ExactDiagonalizationProblem, algorithm::AbstractAlgorithm{false}; kwargs...
)
    !ishermitian(prob.hamiltonian) && algorithm isa LOBPCGSolver &&
        @warn "LOBPCGSolver() is not suitable for non-hermitian matrices."

    # Merge keyword arguments from problem, algorithm and ones passed to this function
    # and split them into sets that are passed to BasisSetRepresentation and ones that
    # are left for the solver.
    kwargs = (; prob.kwargs..., algorithm.kwargs..., kwargs...)
    bsr_kwargs, solver_kwargs = extract_and_delete_keys(
        kwargs,
        :sizelim, :cutoff, :filter, :nnzs, :col_hint, :sort, :max_depth, :minimum_size
    )

    # determine the starting address or vector and seed address to build the matrix from
    addr_or_vec = _set_up_starting_address(
        prob.initial_vector, prob.hamiltonian
    )

    # create the BasisSetRepresentation
    bsr = BasisSetRepresentation(prob.hamiltonian, addr_or_vec; bsr_kwargs...)
    matrix = bsr.sparse_matrix
    basis = bsr.basis

    initial_vector = _set_up_initial_vector(prob.initial_vector, basis)

    return IterativeEDSolver(algorithm, prob, matrix, initial_vector, basis, solver_kwargs)
end

struct DenseEDSolver{P,A,BSR}
    problem::P
    algorithm::A
    basis_set_rep::BSR
    solver_kwargs::NamedTuple
end
function Base.show(io::IO, s::DenseEDSolver)
    io = IOContext(io, :compact => true)
    print(io, "DenseEDSolver\n for hamiltonian = ")
    show(io, s.problem.hamiltonian)
    print(io, ",\n  kwargs = ")
    show(io, s.solver_kwargs)
    print(io, "\n)")
end

function CommonSolve.init(
    prob::ExactDiagonalizationProblem, algorithm::LinearAlgebraSolver; kwargs...
)
    # Merge keyword arguments from problem, algorithm and ones passed to this function
    # and split them into sets that are passed to BasisSetRepresentation and ones that
    # are left for the solver.
    kwargs = (; sizelim=1e5, prob.kwargs..., algorithm.kwargs..., kwargs...)
    bsr_kwargs, solver_kwargs = extract_and_delete_keys(
        kwargs,
        :sizelim, :cutoff, :filter, :nnzs, :col_hint, :sort, :max_depth, :minimum_size
    )

    # determine the seed address to build the matrix from
    addr_or_vec = _set_up_starting_address(
        prob.initial_vector, prob.hamiltonian
    )

    # create the BasisSetRepresentation
    bsr = BasisSetRepresentation(prob.hamiltonian, addr_or_vec; bsr_kwargs...)

    return DenseEDSolver(prob, algorithm, bsr, solver_kwargs)
end
