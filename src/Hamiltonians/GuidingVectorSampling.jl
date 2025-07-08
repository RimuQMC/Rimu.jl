"""
    GuidingVectorSampling

Wrapper over any [`AbstractHamiltonian`](@ref) that implements guided vector a.k.a. guided
wave function sampling. In this importance sampling scheme the Hamiltonian is modified as
follows.

```math
\\tilde{H}_{ij} = v_i H_{ij} v_j^{-1}
```

and where `v` is the guiding vector. `v_i` and `v_j` are also thresholded to avoid dividing
by zero (see below).

# Constructors

* `GuidingVectorSampling(::AbstractHamiltonian, vector, eps)`
* `GuidingVectorSampling(::AbstractHamiltonian; vector, eps)`

`eps` is a thresholding parameter used to avoid dividing by zero; all values below `eps` are
set to `eps`. It is recommended that `eps` is in the same value range as the guiding
vector. The default value is set to `eps=norm(v, Inf) * 1e-2`

After construction, we can access the underlying hamiltonian with `G.hamiltonian`, the
`eps` parameter with `G.eps`, and the guiding vector with `G.vector`.

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS(1,1,1); u=6.0, t=1.0);

julia> v = DVec(starting_address(H) => 10);

julia> G = GuidingVectorSampling(H, v, 0.1);

julia> Matrix(H)
4×4 Matrix{Float64}:
 12.0      4.89898  4.89898  4.89898
  4.89898  9.0      0.0      0.0
  4.89898  0.0      9.0      0.0
  4.89898  0.0      0.0      0.0

julia> Matrix(G)
4×4 Matrix{Float64}:
 12.0        489.898  489.898  489.898
  0.0489898    9.0      0.0      0.0
  0.0489898    0.0      9.0      0.0
  0.0489898    0.0      0.0      0.0

julia> eigen(Matrix(H)).values
4-element Vector{Float64}:
 -2.3661456273236645
  4.9594958589580465
  8.999999999999996
 18.406649768365643

julia> eigen(Matrix(G)).values
4-element Vector{Float64}:
 -2.366145627323689
  4.9594958589580465
  8.999999999999998
 18.406649768365643
```

# Observables

To calculate observables, pass the transformed Hamiltonian `G` to
[`AllOverlaps`](@ref) with keyword argument `transform=G`.
"""
struct GuidingVectorSampling{A,T,H<:AbstractHamiltonian{T},D} <: ModifiedHamiltonian{T}
    # The A parameter sets whether this is an adjoint or not.
    hamiltonian::H
    vector::D
    eps::Float64
end

function GuidingVectorSampling(h, v::AbstractDVec, eps=1e-2 * norm(v, Inf))
    return GuidingVectorSampling{false,eltype(h),typeof(h),typeof(v)}(h, v, eps)
end
function GuidingVectorSampling(h; vector, eps=1e-2 * norm(vector, Inf))
    return GuidingVectorSampling(h, vector, eps)
end

function LOStructure(::Type{<:GuidingVectorSampling{<:Any,<:Any,H}}) where {H}
    if LOStructure(H) ≡ AdjointUnknown()
        return AdjointUnknown()
    else
        return AdjointKnown()
    end
end

function LinearAlgebra.adjoint(h::GuidingVectorSampling{A,T,<:Any,D}) where {A,T,D}
    h_adj = h.hamiltonian'
    return GuidingVectorSampling{!A,T,typeof(h_adj),D}(h_adj, h.vector, h.eps)
end

parent_hamiltonian(h::GuidingVectorSampling) = h.hamiltonian
modify_diagonal(::GuidingVectorSampling, _, value) = value

_apply_eps(x, eps) = ifelse(iszero(x), eps, ifelse(abs(x) < eps, sign(x) * eps, x))

function guiding_vector_modify(value, is_adjoint, eps, guide1, guide2)
    if iszero(guide1) && iszero(guide2)
        return value
    else
        guide1 = _apply_eps(guide1, eps)
        guide2 = _apply_eps(guide2, eps)
        if is_adjoint
            return value * (guide1 / guide2)
        else
            return value * (guide2 / guide1)
        end
    end
end

function modify_offdiagonal(h::GuidingVectorSampling{A}, in, out, value) where {A}
    guide1 = h.vector[in]
    guide2 = h.vector[out]

    return out => guiding_vector_modify(value, A, h.eps, guide1, guide2)
end

"""
    TransformUndoer(k::GuidingVectorSampling, op::AbstractOperator)
    TransformUndoer(k::GuidingVectorSampling)

For a guiding vector similarity transformation ``\\hat{G} = f \\hat{H} f^{-1}``
define the operator ``f^{-1} \\hat{A} f^{-1}``, and special case ``f^{-2}``, in order
to calculate observables. Here ``f`` is a diagonal operator whose entries are
the components of the guiding vector, i.e.``f_{ii} = v_i``.

See [`AllOverlaps`](@ref), [`GuidingVectorSampling`](@ref).
"""
function TransformUndoer(k::GuidingVectorSampling, op::Union{Nothing,AbstractOperator})
    if isnothing(op)
        T = eltype(k)
    else
        T = promote_type(eltype(k), eltype(op))
    end
    return TransformUndoer{T,typeof(k),typeof(op)}(k, op)
end

const GuidingVectorTransformUndoer{A} = TransformUndoer{<:Any,<:GuidingVectorSampling,A}

LOStructure(::Type{<:GuidingVectorTransformUndoer{A}}) where {A} = LOStructure(A)

function LinearAlgebra.adjoint(s::GuidingVectorTransformUndoer)
    a_adj = adjoint(s.op)
    return TransformUndoer(s.transform, a_adj)
end

function modify_diagonal(s::GuidingVectorTransformUndoer{<:AbstractOperator}, addr, val)
    guide = s.transform.vector[addr]

    return guiding_vector_modify(val, true, s.transform.eps, 1.0, 2 * guide)
end
function modify_offdiagonal(s::GuidingVectorTransformUndoer, in, out, val)
    guide1 = s.transform.vector[in]
    guide2 = s.transform.vector[out]

    return out => guiding_vector_modify(val, true, s.transform.eps, 1., guide1 + guide2)
end
