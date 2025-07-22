"""
    GutzwillerSampling(H::AbstractHamiltonian; g)

Wrapper over any [`AbstractHamiltonian`](@ref) that implements Gutzwiller sampling. In this
importance sampling scheme the Hamiltonian is modified as follows
```math
\\tilde{H}_{ij} = H_{ij} e^{-g(H_{ii} - H_{jj})} .
```
This way off-diagonal spawns to higher-energy configurations are discouraged and spawns to
lower-energy configurations encouraged for positive `g` while keeping the spectrum of the
Hamiltonian intact.

# Constructor

* `GutzwillerSampling(::AbstractHamiltonian, g)`
* `GutzwillerSampling(::AbstractHamiltonian; g)`

After construction, we can access the underlying Hamiltonian with `G.hamiltonian` and the
`g` parameter with `G.g`.

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS(1,1,1); u=6.0, t=1.0)
HubbardMom1D(fs"|1 1 1⟩"; u=6.0, t=1.0)

julia> G = GutzwillerSampling(H, g=0.3)
GutzwillerSampling(HubbardMom1D(fs"|1 1 1⟩"; u=6.0, t=1.0); g=0.3)

julia> Matrix(H; sort=true)
4×4 Matrix{Float64}:
 9.0      0.0       4.89898  0.0
 0.0      0.0       4.89898  0.0
 4.89898  4.89898  12.0      4.89898
 0.0      0.0       4.89898  9.0

julia> Matrix(G; sort=true)
4×4 Matrix{Float64}:
 9.0      0.0        12.0495  0.0
 0.0      0.0       179.294   0.0
 1.99178  0.133858   12.0     1.99178
 0.0      0.0        12.0495  9.0

julia> eigen(Matrix(H)).values
4-element Vector{Float64}:
 -2.3661456273236645
  4.9594958589580465
  8.999999999999996
 18.406649768365643

julia> eigen(Matrix(G)).values
4-element Vector{Float64}:
 -2.366145627323686
  4.959495858958046
  8.999999999999998
 18.40664976836564
```

# Observables

See [`AllOverlaps`](@ref) for calculation of observables with a transformed Hamiltonian.
"""
struct GutzwillerSampling{A,T,H<:AbstractHamiltonian{T}} <: ModifiedHamiltonian{T}
    # The A parameter sets whether this is an adjoint or not.
    hamiltonian::H
    g::Float64
end

function GutzwillerSampling(h, g)
    return GutzwillerSampling{false,eltype(h),typeof(h)}(h, Float64(g))
end
GutzwillerSampling(h; g) = GutzwillerSampling(h, g)

function Base.show(io::IO, h::GutzwillerSampling{A}) where {A}
    A && print(io, "adjoint(")
    print(io, "GutzwillerSampling(", h.hamiltonian, "; g=", h.g, ")")
    A && print(io, ")")
end

function LOStructure(::Type{<:GutzwillerSampling{<:Any,<:Any,H}}) where {H}
    if LOStructure(H) ≡ AdjointUnknown()
        return AdjointUnknown()
    else
        return AdjointKnown()
    end
end
function LinearAlgebra.adjoint(h::GutzwillerSampling{A}) where {A}
    h_adj = h.hamiltonian'
    return GutzwillerSampling{!A,eltype(h_adj),typeof(h_adj)}(h_adj, h.g)
end

function Base.:(==)(a::GutzwillerSampling{A}, b::GutzwillerSampling{B}) where {A,B}
   return A == B && a.g == b.g && a.hamiltonian == b.hamiltonian
end

function gutzwiller_modify(matrix_element, is_adjoint, g, diag1, diag2)
    if is_adjoint
        return matrix_element * exp(-g * (diag1 - diag2))
    else
        return matrix_element * exp(-g * (diag2 - diag1))
    end
end

parent_operator(h::GutzwillerSampling) = h.hamiltonian
modify_diagonal(h::GutzwillerSampling, _, value) = value

function modify_offdiagonal(h::GutzwillerSampling{A}, in, out, value) where {A}
    diag1 = diagonal_element(operator_column(h, in))
    diag2 = diagonal_element(operator_column(h, out))
    return out => gutzwiller_modify(value, A, h.g, diag1, diag2)
end

"""
    TransformUndoer(k::GutzwillerSampling, op::AbstractOperator)
    TransformUndoer(k::GutzwillerSampling)

For a Gutzwiller similarity transformation ``\\hat{G} = f \\hat{H} f^{-1}``
define the operator ``f^{-1} \\hat{A} f^{-1}``, and special case ``f^{-2}``, in order
to calculate observables. Here ``f`` is a diagonal operator whose entries are
``f_{ii} = e^{-g H_{ii}}``.

See [`AllOverlaps`](@ref), [`GutzwillerSampling`](@ref).
"""
function TransformUndoer(k::GutzwillerSampling, op::AbstractOperator)
    T = promote_type(eltype(k), eltype(op))
    return TransformUndoer{T,typeof(k),typeof(op)}(k, op)
end

undo_transform(g::GutzwillerSampling, op::AbstractOperator) = TransformUndoer(g, op)

const GutzwillerTransformUndoer{A} = TransformUndoer{<:Any,<:GutzwillerSampling,A}

LOStructure(::Type{<:GutzwillerTransformUndoer{A}}) where {A} = LOStructure(A)

function LinearAlgebra.adjoint(s::GutzwillerTransformUndoer)
    a_adj = adjoint(s.op)
    return TransformUndoer(s.transform, a_adj)
end
function modify_diagonal(s::GutzwillerTransformUndoer, addr, val)
    diag = diagonal_element(s.transform.hamiltonian, addr)

    return gutzwiller_modify(val, true, s.transform.g, 0.0, 2 * diag)
end
function modify_offdiagonal(s::GutzwillerTransformUndoer, in, out, val)
    diag1 = diagonal_element(s.transform.hamiltonian, in)
    diag2 = diagonal_element(s.transform.hamiltonian, out)
    return out => gutzwiller_modify(val, true, s.transform.g, 0.0, diag1 + diag2)
end
