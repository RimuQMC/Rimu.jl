"""
    Multiplier(::AbstractOperator{T}, basis; eltype=T)
    Multiplier(::AbstractHamiltonian{T}, [address]; full_basis=true, eltype=T)

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

julia> mul = ExactDiagonalization.Multiplier(H, bsr.basis);

julia> w2 = mul * v;

julia> w1 ≈ w2
true

julia> dot(w1, bsr.sparse_matrix, v) ≈ dot(w1, mul, v)
true
```
"""
struct Multiplier{T,H<:AbstractOperator,A,I}
    hamiltonian::H
    basis::Vector{A}
    mapping::Dict{A,I}
    size::Tuple{Int,Int}
    buffer::Matrix{T}
    indices::Vector{UnitRange{Int}}
end
function Multiplier(
    hamiltonian::H, basis::Vector{A}; eltype=eltype(H)
) where {A,H<:AbstractOperator}
    I = length(basis) > typemax(Int32) ? Int64 : Int32
    T = eltype
    mapping = Dict(b => I(i) for (b, i) in zip(basis, eachindex(basis)))
    threads = Threads.nthreads()
    buffer = zeros(T, (length(basis), threads))

    chunk_size = length(basis) ÷ threads
    prev = 0
    indices = UnitRange{Int}[]
    for t in 1:threads - 1
        push!(indices, prev+1:prev+chunk_size)
        prev += chunk_size
    end
    push!(indices, prev+1:length(basis))

    return Multiplier{T,H,A,I}(
        hamiltonian, basis, mapping, (length(basis), length(basis)), buffer, indices
    )
end
function Multiplier(
    hamiltonian::AbstractHamiltonian,
    address::AbstractFockAddress=starting_address(hamiltonian);
    full_basis=true, eltype=eltype(hamiltonian),
)
    if !full_basis || address isa OccupationNumberFS
        basis = build_basis(hamiltonian, address)
    else
        basis = build_basis(address)
    end
    return Multiplier(hamiltonian, basis)
end
function Base.show(io::IO, mul::Multiplier{T}) where {T}
    print(io, "Multiplier{$T}($(mul.hamiltonian))")
end

Base.size(mul::Multiplier) = mul.size
Base.size(mul::Multiplier, i) = mul.size[i]
Base.eltype(::Type{Multiplier{T}}) where {T} = T
Base.eltype(::Multiplier{T}) where {T} = T
LinearAlgebra.issymmetric(mul::Multiplier) = issymmetric(mul.hamiltonian)
LinearAlgebra.ishermitian(mul::Multiplier) = ishermitian(mul.hamiltonian)

function Base.adjoint(mul::Multiplier{T,<:Any,A,I}) where {T,A,I}
    hamiltonian = mul.hamiltonian'
    return Multiplier{T,typeof(hamiltonian),A,I}(
        hamiltonian, mul.basis, mul.mapping, mul.size,
    )
end

function LinearAlgebra.mul!(dst, mul::Multiplier{T}, src) where {T}
    @boundscheck begin
        length(src) == size(mul, 2) || throw(DimensionMismatch("operator has size $(size(mul)), vector has length $(length(src))"))
        length(dst) == size(mul, 1) || throw(DimensionMismatch("operator has size $(size(mul)), output vector has length $(length(dst))"))
        @assert size(mul.buffer, 1) == length(src)
    end
    H = mul.hamiltonian
    basis = mul.basis
    mapping = mul.mapping
    buffer = mul.buffer
    indices = mul.indices

    @inbounds Threads.@threads for t in 1:size(mul.buffer, 2)
        buffer[:, t] .= zero(T)
        for i in indices[t]
            addr1 = mul.basis[i]
            val1 = src[i]
            buffer[i, t] += diagonal_element(H, addr1) * val1
            for (addr2, elem) in offdiagonals(H, addr1)
                j = get(mapping, addr2, 0)
                !iszero(j) && (buffer[j, t] += elem * val1)
            end
        end
    end
    return sum!(dst, buffer)
end

function (mul::Multiplier)(src)
    dst = zeros(length(src))
    return LinearAlgebra.mul!(dst, mul, src)
end

Base.:*(mul, src) = mul(src)

function LinearAlgebra.dot(dst, mul::Multiplier, src)
    @boundscheck begin
        length(src) == size(mul, 2) || throw(DimensionMismatch("operator has size $(size(mul)), vector has length $(length(src))"))
        length(dst) == size(mul, 1) || throw(DimensionMismatch("operator has size $(size(mul)), output vector has length $(length(dst))"))
        @assert size(mul.buffer, 1) == length(src)
    end

    H = mul.hamiltonian
    basis = mul.basis
    mapping = mul.mapping
    buffer = mul.buffer
    indices = mul.indices

    @inbounds Threads.@threads for t in 1:size(mul.buffer, 2)
        buffer[1, t] = result = zero(eltype(buffer))
        for i in indices[t]
            addr1 = mul.basis[i]
            val1 = src[i]
            result += conj(dst[i]) * diagonal_element(H, addr1) * val1
            for (addr2, elem) in offdiagonals(H, addr1)
                j = get(mapping, addr2, 0)
                result += conj(get(dst, j, 0.0)) * elem * val1
            end
        end
        buffer[1, t] = result
    end
    return sum(buffer[1, t] for t in 1:size(mul.buffer, 2))
end
