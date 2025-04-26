"""
    ExactDiagonalizationProblem(hamiltonian::AbstractHamiltonian, [v0]; kwargs...)

Defines an exact diagonalization problem with an [`AbstractHamiltonian`](@ref) `hamiltonian`.
Optionally, a starting vector of type [`AbstractDVec`](@ref), or a single address or a
collection of addresses can be passed as `v0`.

`ExactDiagonalizationProblem`s can be solved with
[`solve`](@ref solve(::ExactDiagonalizationProblem)).

# Keyword arguments
- `algorithm=LinearAlgebraSolver()`: The algorithm to use for solving the problem. The
    algorithm can also be specified as the second positional argument in the `init`
    function.
- `linear_dimension` (optional): The estimated dimension of the problem. This is
    usually automatically determined from the Hamiltonian.
- Optional keyword arguments will be passed on to the `init` and `solve` functions.

# Algorithms
- [`LinearAlgebraSolver()`](@ref): An algorithm for solving the problem using the
    dense-matrix eigensolver from the `LinearAlgebra` standard library (eventually using
    LAPACK). Only suitable for small matrices.
- [`KrylovKitSolver(matrix_free=true)`](@ref): An algorithm for finding a few eigenvalues
    and vectors. With `matrix_free=true` the problem is solved without instatiating a
    matrix. This is suitable for large dimensions. With `matrix_free=false` the problem is
    solved after instantiating a sparse matrix. This is faster if sufficient memory is
    available. Requires `using KrylovKit`.
- [`ArpackSolver()`](@ref): An algorithm for solving the problem after instantiating a
    sparse matrix and using the Arpack Fortran library. Requires `using Arpack`.
- [`LOBPCGSolver()`](@ref): An algorithm for solving the problem after instantiating a
    sparse matrix using the LOBPCG method. Requires `using IterativeSolvers`.

# Keyword arguments for matrix-based algorithms (also accepted by [`init`](@ref init(::ExactDiagonalizationProblem)))
See [`BasisSetRepresentation`](@ref) for more information.
- `sizelim`: The maximum size of the basis set representation. The default is `10^6` for
    sparse matrices and `10^5` for dense matrices.
- `cutoff`: A cutoff value for the basis set representation.
- `filter`: A filter function for the basis set representation.
- `max_depth = Inf`: Limit the depth when building the matrix.
- `minimum_size = Inf`: Stop building the matrix after this size is reached.
- `nnzs = 0`: A hint for the number of non-zero elements in the basis set representation.
  Setting a non-zero value can speed up the computation.
- `col_hint = 0`: A hint for the number of columns in the basis set representation.
- `sort = false`: Whether to sort the basis set representation.

# Keyword arguments for iterative algorithms (also accepted by [`solve`](@ref solve(::ExactDiagonalizationProblem)))
- `verbose = false`: Whether to print additional information.
- `abstol = nothing`: The absolute tolerance for the solver. If `nothing`, the solver
    chooses a default value.
- `howmany = 1`: The minimum number of eigenvalues to compute.
- `which = :SR`: Whether to compute the largest or smallest eigenvalues.
- `maxiters = nothing`: The maximum number of iterations for the solver. If `nothing`, the
    solver chooses a default value.

# Solving an `ExactDiagonalizationProblem`
The [`solve`](@ref solve(::ExactDiagonalizationProblem)) function can be called directly on
an `ExactDiagonalizationProblem` to solve it. Alternatively, the
[`init`](@ref init(::ExactDiagonalizationProblem)) function can be used to initialize a
solver, which can then be solved with [`solve`](@ref solve(::ExactDiagonalizationProblem)).
The [`solve`](@ref solve(::ExactDiagonalizationProblem)) function returns a result
type with the eigenvalues, eigenvectors, and convergence information.

## Result type
The result type for the [`solve`](@ref solve(::ExactDiagonalizationProblem)) function is
determined by the algorithm used. It has the following fields:
- `values::Vector`: The eigenvalues.
- `vectors::Vector{<:AbstractDVec}`: The eigenvectors.
- `success::Bool`: A boolean flag indicating whether the solver was successful.
- `info`: Convergence information.
- `algorithm`: The algorithm used for the computation.
- `problem`: The `ExactDiagonalizationProblem` that was solved.
- Additional fields may be present depending on the algorithm used.

Iterating the result type will yield the eigenvalues, eigenvectors, and a boolean flag
`success` in that order.

# Examples
```jldoctest
julia> p = ExactDiagonalizationProblem(HubbardReal1D(BoseFS(1,1,1)))
ExactDiagonalizationProblem(
  HubbardReal1D(fs"|1 1 1⟩"; u=1.0, t=1.0),
  nothing;
  linear_dimension=10,
  NamedTuple()...
)

julia> result = solve(p) # convert to dense matrix and solve with LinearAlgebra.eigen
EDResult for algorithm LinearAlgebraSolver() with 10 eigenvalue(s),
  values = [-5.09593, -1.51882, -1.51882, 1.55611, 1.6093, 1.6093, 4.0, 4.53982, 4.90952, 4.90952]
  Convergence info: "Dense matrix eigensolver solution from `LinearAlgebra.eigen`", with howmany = 10 eigenvalues requested.
  success = true.

julia> using KrylovKit # an external package has to be installed and loaded

julia> s = init(p; algorithm = KrylovKitSolver(true)) # solve without building a matrix
IterativeEDSolver
 with algorithm KrylovKitSolver(matrix_free = true,) for hamiltonian = HubbardReal1D(fs"|1 1 1⟩"; u=1.0, t=1.0),
  kwargs = NamedTuple()
)

julia> values, vectors, success = solve(s);

julia> result.values[1] ≈ values[1]
true
```
See also [`solve(::ExactDiagonalizationProblem)`](@ref),
[`init(::ExactDiagonalizationProblem)`](@ref),
[`KrylovKitSolver`](@ref), [`ArpackSolver`](@ref), [`LinearAlgebraSolver`](@ref).
!!! note
    Using the `KrylovKitSolver()` algorithms requires the
    KrylovKit.jl package. The package can be loaded with `using KrylovKit`.
    Using the `ArpackSolver()` algorithm requires the Arpack.jl package. The package can be
    loaded with `using Arpack`.
    Using the `LOBPCGSolver()` algorithm requires the IterativeSolvers.jl package. The package
    can be loaded with `using IterativeSolvers`.
"""
struct ExactDiagonalizationProblem{H<:AbstractHamiltonian, V, D, AV}
    hamiltonian::H
    initial_vector::V
    linear_dimension::D
    addr_or_vec::AV # starting address or iterable of addresses
    kwargs::NamedTuple
end

function ExactDiagonalizationProblem(
    hamiltonian::H, initial_vector::V=nothing; linear_dimension=nothing, kwargs...
) where {H<:AbstractHamiltonian,V}
    # Set up the starting address or vector
    addr_or_vec = _set_up_starting_address(initial_vector, hamiltonian)
    if linear_dimension === nothing
        linear_dimension = dimension(
            hamiltonian,
            addr_or_vec isa AbstractFockAddress ? addr_or_vec : first(addr_or_vec)
        )
    end
    return ExactDiagonalizationProblem{H,V,typeof(linear_dimension), typeof(addr_or_vec)}(
        hamiltonian, initial_vector, linear_dimension, addr_or_vec, NamedTuple(kwargs)
    )
end

function ExactDiagonalizationProblem(
    hamiltonian::AbstractHamiltonian, v0::AbstractDVec; kwargs...
)
    return ExactDiagonalizationProblem(
        hamiltonian, FrozenDVec(collect(pairs(v0))); kwargs...
    )
end

function Base.show(io::IO, p::ExactDiagonalizationProblem)
    io = IOContext(io, :compact => true)
    print(io, "ExactDiagonalizationProblem(\n  ")
    show(io, p.hamiltonian)
    print(io, ",\n  ")
    show(io, p.initial_vector)
    print(io, ";\n  ")
    print(io, "linear_dimension=$(p.linear_dimension),\n  ")
    show(io, p.kwargs)
    print(io, "...\n)")
end
function Base.:(==)(p1::ExactDiagonalizationProblem, p2::ExactDiagonalizationProblem)
    return p1.hamiltonian == p2.hamiltonian &&
        p1.initial_vector == p2.initial_vector &&
        p1.kwargs == p2.kwargs
end

Rimu.Hamiltonians.dimension(p::ExactDiagonalizationProblem) = p.linear_dimension

function _set_up_starting_address(v0, ham)
    if isnothing(v0)
        addr_or_vec = starting_address(ham)
    elseif allows_address_type(ham, v0) ||
           v0 isa Union{NTuple,Vector} && allows_address_type(ham, eltype(v0))
        addr_or_vec = v0
    elseif v0 isa FrozenDVec
        addr_or_vec = keys(v0)
    else
        throw(ArgumentError("Invalid starting vector in `ExactDiagonalizationProblem`."))
    end

    return addr_or_vec # single address or iterable of addresses
end
