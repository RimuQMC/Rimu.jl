"""
    OperatorAsMap(::AbstractOperator{T}, basis; eltype=T)
    OperatorAsMap(::AbstractHamiltonian{T}, [address]; full_basis=true, eltype=T)

Wrapper for an [`AbstractOperator`](@ref) and a basis that allows multiplying regular Julia
vectors with the operator.

The `eltype` argument can be used to change the eltype of the internal buffers, e.g. for
multiplying complex vectors with real operators.

If an [`AbstractHamiltonian`](@ref) with no `basis` is passed, the basis is constructed
automatically. In that case, when `full_basis=true` the entire basis is constructed from an
address as [`build_basis`](@ref)`(address)`, otherwise it is constructed as
[`build_basis`](@ref)`(hamiltonian, address)`. You may want to set `full_basis=false` when
dealing with Hamiltonians that block, such as [`HubbardMom1D`](@ref).

Supports calling, `Base.:*`, `mul!` and the three-argument `dot`.

## Example

```julia
julia> H = HubbardReal1D(BoseFS(1, 1, 1, 1));

julia> bsr = BasisSetRepresentation(H);

julia> v = ones(length(bsr.basis));

julia> w1 = bsr.sparse_matrix * v;

julia> op = ExactDiagonalization.OperatorAsMap(H, bsr.basis);

julia> w2 = op * v;

julia> w1 ≈ w2
true

julia> dot(w1, bsr.sparse_matrix, v) ≈ dot(w1, op, v)
true
```
"""
struct OperatorAsMap{T,H<:AbstractOperator{T},A} <: LinearMap{T}
    hamiltonian_adj::H
    basis::Vector{A}
    mapping::Dict{A,Int}
end
function OperatorAsMap(hamiltonian::AbstractOperator, basis::Vector=build_basis(hamiltonian))
    if !allows_address_type(hamiltonian, eltype(basis))
        throw(ArgumentError("basis is incompatible with operator"))
    end
    if LOStructure(hamiltonian) == AdjointUnknown()
        throw(ArgumentError("operator not supported. Please implement `adjoint`"))
    end

    mapping = Dict(zip(basis, eachindex(basis)))
    hamiltonian_adj = hamiltonian'
    H = typeof(hamiltonian_adj)
    T = eltype(hamiltonian_adj)
    A = eltype(basis)

    return OperatorAsMap{T,H,A}(hamiltonian, basis, mapping)
end

Base.size(op::OperatorAsMap) = (length(op.basis), length(op.basis))
Base.size(op::OperatorAsMap, i) = length(op.basis)
Base.eltype(::Type{OperatorAsMap{H}}) where {H} = eltype(H)
LinearAlgebra.ishermitian(op::OperatorAsMap) = ishermitian(op.hamiltonian_adj)
LinearAlgebra.issymmetric(op::OperatorAsMap) = issymmetric(op.hamiltonian_adj)

function Base.adjoint(op::OperatorAsMap{T,<:Any,A}) where {T,A}
    hamiltonian_adj = op.hamiltonian_adj'
    H = typeof(hamiltonian_adj)
    return OperatorAsMap{T,H,A}(hamiltonian_adj, op.basis, op.mapping)
end

"""
    row_dot_vector(op::OperatorAsMap, row_index, vector)

Compute `dot(H[i, :], v)` where `H` is the Hamiltonian stored in `op`.
"""
function row_dot_vector(op::OperatorAsMap, row_index, vector)
    row = op.basis[row_index]
    row_result = diagonal_element(op.hamiltonian_adj, row) * vector[row_index]
    for (col, val) in offdiagonals(op.hamiltonian_adj', row)
        j = op.mapping[col]
        row_result += val * vector[j]
    end
    return row_result
end

function LinearMaps._unsafe_mul!(dst, op::OperatorAsMap, src::AbstractVector, α=1, β=0)
    Folds.foreach(eachindex(dst)) do i
        dst[i] = row_dot_vector(op, i, src) * α + src[i] * β
    end
    return dst
end

function LinearAlgebra.dot(dst, op::OperatorAsMap, src)
    Folds.sum(eachindex(dst)) do i
        conj(dst[i]) * row_dot_vector(op, i, src)
    end
end
