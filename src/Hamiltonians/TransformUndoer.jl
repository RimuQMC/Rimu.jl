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
parent_operator(t::TransformUndoer) = t.op
starting_address(t::TransformUndoer) = starting_address(t.transform)

# common methods
function Base.:(==)(a::TransformUndoer, b::TransformUndoer)
    return a.transform == b.transform && a.op == b.op
end
