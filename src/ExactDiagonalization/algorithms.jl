abstract type AbstractAlgorithm{MatrixFree} end
ismatrixfree(::AbstractAlgorithm{MatrixFree}) where {MatrixFree} = MatrixFree

"""
    KrylovKitSolver(matrix_free::Bool; kwargs...)
    KrylovKitSolver(; matrix_free = false, kwargs...)

Algorithm for solving a large [`ExactDiagonalizationProblem`](@ref) to find a few
eigenvalues and vectors using the KrylovKit.jl package.
The Lanczos method is used for hermitian matrices, and the Arnoldi method is used for
non-hermitian matrices.

# Arguments
- `matrix_free = false`: Whether to use a matrix-free algorithm. If `false`, a sparse matrix
  will be instantiated. This is typically faster and recommended for small matrices,
  but requires more memory. If `true`, the matrix is not instantiated, which is useful for
  large matrices that would not fit into memory. The calculation will parallelise using
  threading if available by making use of [`LinearMap`](@ref).
- `kwargs`: Additional keyword arguments are passed on to the function
    [`KrylovKit.eigsolve()`](https://jutho.github.io/KrylovKit.jl/stable/man/eig/#KrylovKit.eigsolve).

See also [`ExactDiagonalizationProblem`](@ref),
[`solve`](@ref solve(::ExactDiagonalizationProblem)), [`ArpackSolver`](@ref),
[`LOBPCGSolver`](@ref).

!!! note
    Requires the KrylovKit.jl package to be loaded with `using KrylovKit`.
"""
struct KrylovKitSolver{MatrixFree} <: AbstractAlgorithm{MatrixFree}
    kwargs::NamedTuple
    # the inner constructor checks if KrylovKit is loaded
    function KrylovKitSolver{MF}(; kwargs...) where {MF}
        ext = Base.get_extension(@__MODULE__, :KrylovKitExt)
        if ext === nothing
            error("KrylovKitSolver requires that KrylovKit is loaded, i.e. `using KrylovKit`")
        else
            return new{MF}(NamedTuple(kwargs))
        end
    end
end
KrylovKitSolver(matrix_free::Bool; kwargs...) = KrylovKitSolver{matrix_free}(; kwargs...)
KrylovKitSolver(; matrix_free=false, kwargs...) = KrylovKitSolver(matrix_free; kwargs...)

function Base.show(io::IO, s::KrylovKitSolver)
    nt = (; matrix_free=ismatrixfree(s), s.kwargs...)
    io = IOContext(io, :compact => true)
    print(io, "KrylovKitSolver")
    show(io, nt)
end

"""
    ArpackSolver(matrix_free::Bool; kwargs...)
    ArpackSolver(; matrix_free = false, kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a sparse
matrix. It uses the Lanzcos method for hermitian problems, and the Arnoldi method for
non-hermitian problems, using the Arpack Fortran library.

# Arguments
- `matrix_free = false`: Whether to use a matrix-free algorithm. If `false`, a sparse matrix
  will be instantiated. This is typically faster and recommended for small matrices,
  but requires more memory. If `true`, the matrix is not instantiated, which is useful for
  large matrices that would not fit into memory. The calculation will parallelise using
  threading if available by making use of [`LinearMap`](@ref).
- Additional `kwargs` are passed on to the function
  [`Arpack.eigs()`](https://arpack.julialinearalgebra.org/stable/eigs/).

See also [`ExactDiagonalizationProblem`](@ref),
[`solve`](@ref solve(::ExactDiagonalizationProblem)), [`KrylovKitSolver`](@ref),
[`LOBPCGSolver`](@ref).

!!! note
    Requires the Arpack.jl package to be loaded with `using Arpack`.
"""
struct ArpackSolver{MatrixFree} <: AbstractAlgorithm{MatrixFree}
    kwargs::NamedTuple
    # the inner constructor checks if Arpack is loaded
    function ArpackSolver{MF}(; kwargs...) where {MF}
        ext = Base.get_extension(@__MODULE__, :ArpackExt)
        if ext === nothing
            error("ArpackSolver() requires that Arpack.jl is loaded, i.e. `using Arpack`")
        else
            return new{MF}(NamedTuple(kwargs))
        end
    end
end
ArpackSolver(matrix_free::Bool; kwargs...) = ArpackSolver{matrix_free}(; kwargs...)
ArpackSolver(; matrix_free=false, kwargs...) = ArpackSolver(matrix_free; kwargs...)

function Base.show(io::IO, s::ArpackSolver)
    nt = (; matrix_free=ismatrixfree(s), s.kwargs...)
    io = IOContext(io, :compact => true)
    print(io, "ArpackSolver")
    show(io, nt)
end

"""
    LOBPCGSolver(matrix_free::Bool; kwargs...)
    LOBPCGSolver(; matrix_free = false, kwargs...)

The Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).
Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) after instantiating a
sparse matrix.

LOBPCG is not suitable for non-hermitian eigenvalue problems.

# Arguments
- `matrix_free = false`: Whether to use a matrix-free algorithm. If `false`, a sparse matrix
  will be instantiated. This is typically faster and recommended for small matrices,
  but requires more memory. If `true`, the matrix is not instantiated, which is useful for
  large matrices that would not fit into memory. The calculation will parallelise using
  threading if available by making use of [`LinearMap`](@ref).
- Additional `kwargs` are passed on to the function
[`IterativeSolvers.lobpcg()`](https://iterativesolvers.julialinearalgebra.org/dev/eigenproblems/lobpcg/).

See also [`ExactDiagonalizationProblem`](@ref),
[`solve`](@ref solve(::ExactDiagonalizationProblem)), [`KrylovKitSolver`](@ref),
[`ArpackSolver`](@ref).

!!! note
    Requires the IterativeSolvers.jl package to be loaded with `using IterativeSolvers`.
"""
struct LOBPCGSolver{MatrixFree} <: AbstractAlgorithm{MatrixFree}
    kwargs::NamedTuple
    # the inner constructor checks if LinearSolvers is loaded
    function LOBPCGSolver{MF}(; kwargs...) where {MF}
        ext = Base.get_extension(@__MODULE__, :IterativeSolversExt)
        if ext === nothing
            error("LOBPCGSolver() requires that IterativeSolvers.jl is loaded, i.e. `using IterativeSolvers`")
        else
            return new{MF}(NamedTuple(kwargs))
        end
    end
end
LOBPCGSolver(matrix_free::Bool; kwargs...) = LOBPCGSolver{matrix_free}(; kwargs...)
LOBPCGSolver(; matrix_free=false, kwargs...) = LOBPCGSolver(matrix_free; kwargs...)

function Base.show(io::IO, s::LOBPCGSolver)
    nt = (; matrix_free=ismatrixfree(s), s.kwargs...)
    io = IOContext(io, :compact => true)
    print(io, "LOBPCGSolver")
    show(io, nt)
end

"""
    LinearAlgebraSolver(; kwargs...)

Algorithm for solving an [`ExactDiagonalizationProblem`](@ref) using the dense-matrix
eigensolver from the `LinearAlgebra` standard library. This is only suitable for small
matrices.

The `kwargs` are passed on to function [`LinearAlgebra.eigen`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen).

# Keyword arguments
- `permute = true`: Whether to permute the matrix before diagonalization.
- `scale = true`: Whether to scale the matrix before diagonalization.
- `sortby`: The sorting order for the eigenvalues.

See also [`ExactDiagonalizationProblem`](@ref),
[`solve`](@ref solve(::ExactDiagonalizationProblem)).
"""
struct LinearAlgebraSolver <: AbstractAlgorithm{false}
    kwargs::NamedTuple
end
LinearAlgebraSolver(; kwargs...) = LinearAlgebraSolver(NamedTuple(kwargs))

function Base.show(io::IO, s::LinearAlgebraSolver)
    io = IOContext(io, :compact => true)
    if isempty(s.kwargs)
        print(io, "LinearAlgebraSolver()")
    else
        print(io, "LinearAlgebraSolver")
        show(io, s.kwargs)
    end
end
