"""
    abstract type ModifiedHamiltonian{T} <: AbstractHamiltonian{T} end

Abstract type for defining wrappers over [`AbstractHamiltonian`](@ref)s that modify diagonal
and off-diagonal elements via the functions [`modify_diagonal`](@ref) and
[`modify_offdiagonal`](@ref).

The following need to be implemented
* [`parent_hamiltonian`](@ref)
* [`modify_diagonal`](@ref)
* [`modify_offdiagonal`](@ref)
* [`LOStructure(p)`](@ref) and `Base.adjoint` if known, defaults to
  [`AdjointUnknown()`](@ref).

The follwing are provided:
* [`starting_address(op)`](@ref)
* [`allows_address_type(op, type)`](@ref)
* [`operator_column(op, address)`](@ref)
* [`diagonal_element(column)`](@ref)
* [`num_offdiagonals(column)`](@ref) (this can be an upper bound)
* [`starting_address(column)`](@ref)
* [`offdiagonals(column)`](@ref)
* [`random_offdiagonal(column)`](@ref)
* [`dimension(op, address)`](@ref)

"""
abstract type ModifiedHamiltonian{T} <: AbstractHamiltonian{T} end

"""
    parent_hamiltonian(::ModifiedHamiltonian)

Return the Hamiltonian that is being modified.
"""
parent_hamiltonian

"""
    modify_diagonal(ham::ModifiedHamiltonian, source, value) -> val

Modifty the diagonal element where
`value = diagonal_element(operator_column(parent_hamiltonian(ham), source))`.
"""
modify_diagonal

"""
    modify_offdiagonal(ham::ModifiedHamiltonian, source, dest, element) -> (addr => val)

Modfy an offdiagonal element `dest => element` reachable from `source` in the
[`parent_hamiltonian`](@ref) of `ham`.

Should return a `Pair` of modified address `addr` and modified value `val`.
"""
modify_offdiagonal

function allows_address_type(h::ModifiedHamiltonian, ::Type{A}) where {A}
    return allows_address_type(parent_hamiltonian(h), A)
end

LOStructure(::Type{<:ModifiedHamiltonian}) = AdjointUnknown()
starting_address(h::ModifiedHamiltonian) = starting_address(parent_hamiltonian(h))
dimension(h::ModifiedHamiltonian, address) = dimension(parent_hamiltonian(h), address)

struct ModifiedHamiltonianColumn{
    A,T,H<:ModifiedHamiltonian{T},C
} <: AbstractOperatorColumn{A,T,H}
    address::A
    hamiltonian::H
    column::C
end
function operator_column(h::ModifiedHamiltonian, address)
    column = operator_column(parent_hamiltonian(h), address)
    return ModifiedHamiltonianColumn(address, h, column)
end
function Base.show(io::IO, col::ModifiedHamiltonianColumn)
    print(io, "operator_column(", col.hamiltonian, ", ")
    print(IOContext(io, :compact=>true), col.address, ")")
end

starting_address(col::ModifiedHamiltonianColumn) = starting_address(col.column)

function diagonal_element(col::ModifiedHamiltonianColumn)
    value = diagonal_element(col.column)
    return modify_diagonal(col.hamiltonian, starting_address(col.column), value)
end
function num_offdiagonals(col::ModifiedHamiltonianColumn)
    return num_offdiagonals(col.column)
end

function random_offdiagonal(col::ModifiedHamiltonianColumn)
    dest, prob, value = random_offdiagonal(col.column)
    source = starting_address(col.column)
    addr, value = modify_offdiagonal(col.hamiltonian, source, dest, value)
    return addr, prob, value
end

function offdiagonals(col::ModifiedHamiltonianColumn)
    ods = offdiagonals(col.column)
    if ods isa AbstractVector
        return ModifiedHamiltonianVectorOffdiagonals(col.address, col.hamiltonian, ods)
    else
        return ModifiedHamiltonianOffdiagonals(col.address, col.hamiltonian, ods)
    end
end

struct ModifiedHamiltonianVectorOffdiagonals{
    A,T,H<:ModifiedHamiltonian{T},O<:AbstractVector
} <: AbstractVector{Pair{A,T}}
    address::A
    hamiltonian::H
    offdiagonals::O
end
function Base.getindex(ods::ModifiedHamiltonianVectorOffdiagonals, i)
    return modify_offdiagonal(ods.hamiltonian, ods.address, ods.offdiagonals[i]...)
end
Base.size(ods::ModifiedHamiltonianVectorOffdiagonals) = size(ods.offdiagonals)

struct ModifiedHamiltonianOffdiagonals{A,T,H<:ModifiedHamiltonian{T},O}
    address::A
    hamiltonian::H
    offdiagonals::O
end
function Base.show(io::IO, ods::ModifiedHamiltonianOffdiagonals)
    print(io, "offdiagonals(operator_column(", ods.hamiltonian, ", ")
    print(IOContext(io, :compact=>true), ods.address, "))")
end

function Base.iterate(ods::ModifiedHamiltonianOffdiagonals, args...)
    it = iterate(ods.offdiagonals, args...)
    if isnothing(it)
        return nothing
    else
        result, state = it
        return modify_offdiagonal(ods.hamiltonian, ods.address, result...), state
    end
end
function Base.IteratorSize(ods::ModifiedHamiltonianOffdiagonals)
    return Base.IteratorSize(ods.offdiagonals)
end
function Base.length(ods::ModifiedHamiltonianOffdiagonals)
    return length(ods.offdiagonals)
end
Base.eltype(::ModifiedHamiltonianOffdiagonals{A,T}) where {A,T} = Pair{A,T}


"""
    TransformUndoer(transform::AbstractHamiltonian, op::AbstractObservable) <: AbstractHamiltonian

Create a new operator for the purpose of calculating overlaps of transformed
vectors, which are defined by some transformation `transform`. The new operator should
represent the effect of undoing the transformation before calculating overlaps, including
with an optional operator `op`.

Not exported; transformations should define all necessary methods and properties,
see [`AbstractHamiltonian`](@ref). An `ArgumentError` is thrown if used with an
unsupported transformation.

# Example

A similarity transform ``\\hat{G} = f \\hat{H} f^{-1}`` has eigenvector
``d = f \\cdot c`` where ``c`` is an eigenvector of ``\\hat{H}``. Then the
overlap ``c' \\cdot c = d' \\cdot f^{-2} \\cdot d`` can be computed by defining all
necessary methods for `TransformUndoer(G)` to represent the operator ``f^{-2}`` and
calculating `dot(d, TransformUndoer(G), d)`.

Observables in the transformed basis can be computed by defining `TransformUndoer(G, A)`
to represent ``f^{-1} A f^{-1}``.

# Supported transformations

* [`GutzwillerSampling`](@ref)
* [`GuidingVectorSampling`](@ref)
"""
struct TransformUndoer{
    T,K<:AbstractHamiltonian,O<:AbstractObservable
} <: ModifiedHamiltonian{T}
    transform::K
    op::O
end
function TransformUndoer(k::AbstractHamiltonian, op)
    return throw(ArgumentError("Unsupported transformation: $k"))
end
function TransformUndoer(k::AbstractHamiltonian)
    return TransformUndoer(k, IdentityOperator())
end
parent_hamiltonian(t::TransformUndoer) = t.op
starting_address(t::TransformUndoer) = starting_address(t.transform)

# common methods
function Base.:(==)(a::TransformUndoer, b::TransformUndoer)
    return a.transform == b.transform && a.op == b.op
end
