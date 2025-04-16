"""
    OperatorAsMap <: LinearMap

Wrapper over [`AbstractOperator`](@ref) that allows it to be used as a [`LinearMap`](@ref) from [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl).

See [`LinearMap`](@ref) for usage.
"""
struct OperatorAsMap{T,H<:AbstractOperator{T},A} <: LinearMap{T}
    operator_adj::H
    basis::Vector{A}
    mapping::Dict{A,Int}
end

"""
    LinearMap(::AbstractOperator{T}, basis)
    LinearMap(::AbstractOperator{T}; basis)
    LinearMap(
        op::AbstractOperator{T};
        starting_address=starting_address(op),
        full_basis=false
    )

Wrapper for an [`AbstractOperator`](@ref) and a basis that allows multiplying regular Julia
vectors with the operator without storing the matrix representation of the operator in
memory.

If an `op::`[`AbstractOperator`](@ref) with no `basis` is passed, the basis is constructed
automatically from the [`starting_address`](@ref) (note that function is only defined for
[`AbstractHamiltonian`](@ref)s). In that case, when `full_basis=true` the entire basis is
constructed from an address as [`build_basis`](@ref)`(starting_address)`, otherwise it is
constructed as [`build_basis`](@ref)`(op, starting_address)`. You may want to set
`full_basis=false` when dealing with Hamiltonians that block, such as
[`HubbardMom1D`](@ref Main.HubbardMom1D), otherwise setting `full_basis=true` is more
efficient.

Implements the [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)
interface, and can be used in `Base.:*`, `mul!` and the three-argument `dot`. These
functions are multithreaded.

## Example

```julia
julia> H = HubbardReal1D(BoseFS(1, 1, 1, 1));

julia> bsr = BasisSetRepresentation(H);

julia> v = ones(length(bsr.basis));

julia> w1 = bsr.sparse_matrix * v;

julia> op = LinearMap(H, bsr.basis);

julia> w2 = op * v;

julia> w1 ≈ w2
true

julia> dot(w1, bsr.sparse_matrix, v) ≈ dot(w1, op, v)
true
```
"""
function LinearMaps.LinearMap(operator::AbstractOperator, basis)
    if !allows_address_type(operator, eltype(basis))
        throw(ArgumentError("basis is incompatible with operator"))
    end
    if LOStructure(operator) == AdjointUnknown()
        throw(ArgumentError("operator not supported. Please implement `adjoint`"))
    end

    mapping = Dict(zip(basis, eachindex(basis)))
    operator_adj = operator'
    H = typeof(operator_adj)
    T = eltype(operator_adj)
    A = eltype(basis)

    return OperatorAsMap{T,H,A}(operator_adj, basis, mapping)
end
function LinearMaps.LinearMap(
    operator::AbstractOperator;
    starting_address=starting_address(operator),
    basis=nothing,
    full_basis::Bool=false,
)
    if !isnothing(basis)
        full_basis && @warn "`basis` and `full_basis` given. Ignorning `full_basis`."
    elseif full_basis
        basis = build_basis(starting_address)
    else
        basis = build_basis(operator, starting_address)
    end
    return LinearMap(operator, basis)
end
function LinearMaps.LinearMap(
    operator::AbstractOperator,
    starting_address::AbstractFockAddress;
    full_basis::Bool=false,
    )
    return LinearMap(operator; starting_address, full_basis)
end

Base.size(op::OperatorAsMap) = (length(op.basis), length(op.basis))
Base.size(op::OperatorAsMap, i) = length(op.basis)
LinearAlgebra.ishermitian(op::OperatorAsMap) = ishermitian(op.operator_adj)
LinearAlgebra.issymmetric(op::OperatorAsMap) = issymmetric(op.operator_adj)

function Base.:(==)(op1::OperatorAsMap, op2::OperatorAsMap)
    return op1.operator_adj == op2.operator_adj && op1.basis == op2.basis
end

function Base.adjoint(op::OperatorAsMap{T,<:Any,A}) where {T,A}
    operator_adj = op.operator_adj'
    H = typeof(operator_adj)
    return OperatorAsMap{T,H,A}(operator_adj, op.basis, op.mapping)
end

# Compute `dot(H[i, :], v)` where `H` is the Hamiltonian stored in `op`. H' needs to be
# conj-ed everywhere since we actually want the transpose, not the adjoint.
function _row_dot_vector(op::OperatorAsMap, row_index, vector)
    row = op.basis[row_index]
    row_result = conj(diagonal_element(op.operator_adj, row)) * vector[row_index]
    for (col, val) in offdiagonals(op.operator_adj, row)
        if !iszero(val)
            j = op.mapping[col]
            row_result += conj(val) * vector[j]
        end
    end
    return row_result
end

function LinearMaps._unsafe_mul!(dst, op::OperatorAsMap, src::AbstractVector, α=1, β=0)
    Folds.foreach(eachindex(dst)) do i
        if iszero(β) # needs special case in case dst contains NaN or ±Inf
            dst[i] = _row_dot_vector(op, i, src) * α
        else
            dst[i] = _row_dot_vector(op, i, src) * α + dst[i] * β
        end
    end
    return dst
end

function LinearAlgebra.dot(dst, op::OperatorAsMap, src)
    Folds.sum(eachindex(dst)) do i
        conj(dst[i]) * _row_dot_vector(op, i, src)
    end
end

function SparseArrays.sparse(op::OperatorAsMap; kwargs...)
    return build_sparse_matrix_from_LO(op.operator_adj', op.basis; kwargs...)[1]
end
function Base.Matrix(op::OperatorAsMap; kwargs...)
    return Matrix(sparse(op; kwargs...))
end
function BasisSetRepresentation(op::OperatorAsMap; kwargs...)
    return BasisSetRepresentation(sparse(op; kwargs...), op.basis, op.operator_adj')
end
