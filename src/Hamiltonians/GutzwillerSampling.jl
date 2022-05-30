"""
    GutzwillerSampling(::AbstractHamiltonian; g)

Wrapper over any [`AbstractHamiltonian`](@ref) that implements Gutzwiller sampling. In this
importance sampling scheme the Hamiltonian is modified as follows
```math
\\tilde{H}_{ij} = H_{ij} e^{-g(H_{ii} - H_{jj})} .
```
This way off-diagonal spawns to higher-energy configurations are discouraged and
spawns to lower-energy configurations encouraged for positive `g`.

# Constructor

* `GutzwillerSampling(::AbstractHamiltonian, g)`
* `GutzwillerSampling(::AbstractHamiltonian; g)`

After construction, we can access the underlying Hamiltonian with `G.hamiltonian` and the
`g` parameter with `G.g`.

# General operators

Also defines operator `f A f` for arbitrary `A`, where `f^{-1} H f` is the transformed Hamiltonian, 
and special case `f^2`, in order to calculate overlaps. See [`AllOverlaps`](@ref).

* `SimTransOperator(k::GutzwillerSampling{H}, A)`
* `SimTransOperator(k::GutzwillerSampling{H})`

# Example

```jldoctest
julia> H = HubbardMom1D(BoseFS{3}((1,1,1)); u=6.0, t=1.0)
HubbardMom1D(BoseFS{3,3}((1, 1, 1)); u=6.0, t=1.0)

julia> I = GutzwillerSampling(H, g=0.3)
GutzwillerSampling(HubbardMom1D(BoseFS{3,3}((1, 1, 1)); u=6.0, t=1.0); g=0.3)

julia> get_offdiagonal(H, BoseFS((2, 1, 0)), 1)
(BoseFS{3,3}((1, 0, 2)), 2.0)

julia> get_offdiagonal(I, BoseFS((2, 1, 0)), 1)
(BoseFS{3,3}((1, 0, 2)), 0.8131393194811987)
```
"""
struct GutzwillerSampling{A,T,H<:AbstractHamiltonian{T},G} <: AbstractHamiltonian{T}
    # The A parameter sets whether this is an adjoint or not.
    # The G parameter is g parameter value.
    hamiltonian::H
end

function GutzwillerSampling(h, g)
    G = eltype(h)(g)
    return GutzwillerSampling{false,eltype(h),typeof(h),G}(h)
end
GutzwillerSampling(h; g) = GutzwillerSampling(h, g)

function Base.show(io::IO, h::GutzwillerSampling{A}) where {A}
    A && print(io, "adjoint(")
    print(io, "GutzwillerSampling(", h.hamiltonian, "; g=", h.g, ")")
    A && print(io, ")")
end

starting_address(h::GutzwillerSampling) = starting_address(h.hamiltonian)

LOStructure(::Type{<:GutzwillerSampling{<:Any,<:Any,H}}) where {H} = _lo_str(LOStructure(H))
_lo_str(::LOStructure) = AdjointKnown()
_lo_str(::AdjointUnknown) = AdjointUnknown()

function LinearAlgebra.adjoint(h::GutzwillerSampling{A,T,<:Any,G}) where {A,T,G}
    h_adj = h.hamiltonian'
    return GutzwillerSampling{!A,T,typeof(h_adj),G}(h_adj)
end

dimension(::Type{T}, h::GutzwillerSampling) where T = dimension(T, h.hamiltonian)

Base.getproperty(h::GutzwillerSampling, s::Symbol) = getproperty(h, Val(s))
Base.getproperty(h::GutzwillerSampling{<:Any,<:Any,<:Any,G}, ::Val{:g}) where G = G
Base.getproperty(h::GutzwillerSampling, ::Val{:hamiltonian}) = getfield(h, :hamiltonian)

# Forward some interface functions.
num_offdiagonals(h::GutzwillerSampling, add) = num_offdiagonals(h.hamiltonian, add)
diagonal_element(h::GutzwillerSampling, add) = diagonal_element(h.hamiltonian, add)

function gutzwiller_modify(matrix_element, is_adjoint, g, diag1, diag2)
    if is_adjoint
        return matrix_element * exp(-g * (diag1 - diag2))
    else
        return matrix_element * exp(-g * (diag2 - diag1))
    end
end

function get_offdiagonal(h::GutzwillerSampling{A}, add1, chosen) where A
    add2, matrix_element = get_offdiagonal(h.hamiltonian, add1, chosen)
    diag1 = diagonal_element(h, add1)
    diag2 = diagonal_element(h, add2)
    return add2, gutzwiller_modify(matrix_element, A, h.g, diag1, diag2)
end

struct GutzwillerOffdiagonals{
    F,T,A,G,H<:AbstractHamiltonian,N<:AbstractOffdiagonals{F,T}
}<:AbstractOffdiagonals{F,T}
    hamiltonian::H
    diag::T
    offdiagonals::N
end

function offdiagonals(ham::GutzwillerSampling{A,<:Any,<:Any,G}, a) where {A,G}
    hps = offdiagonals(ham.hamiltonian, a)
    diag = diagonal_element(ham, a)
    return GutzwillerOffdiagonals{typeof(a),eltype(ham),A,G,typeof(ham),typeof(hps)}(
        ham, diag, hps
    )
end

function Base.getindex(h::GutzwillerOffdiagonals{F,T,A,G}, i)::Tuple{F,T} where {F,T,A,G}
    add2, matrix_element = h.offdiagonals[i]
    diag2 = diagonal_element(h.hamiltonian, add2)
    return add2, gutzwiller_modify(matrix_element, A, G, h.diag, diag2)
end

Base.size(h::GutzwillerOffdiagonals) = size(h.offdiagonals)

# the similarity transformed operator `f A f` to calculate expectation value of an operator A
struct SimTransOperator{T,K<:AbstractHamiltonian{T},O<:Union{AbstractHamiltonian{T},Nothing}} <: AbstractHamiltonian{T}
    similarity::K
    op::O
end

function SimTransOperator(k, a)
    T = eltype(k)
    return SimTransOperator{T,typeof(k),typeof(a)}(k,a)
end

starting_address(s::SimTransOperator) = starting_address(s.similarity)
dimension(::Type{T}, s::SimTransOperator) where {T} = dimension(T, s.similarity)

# Gutzwiller sampling - general operator `f A f`
LOStructure(::Type{<:SimTransOperator{<:Any,<:GutzwillerSampling,<:A}}) where {A} = LOStructure(A)

function diagonal_element(s::SimTransOperator{<:Any,<:GutzwillerSampling,<:AbstractHamiltonian}, add)
    diagH = diagonal_element(s.similarity.hamiltonian, add)
    diagA = diagonal_element(s.op, add)
    return gutzwiller_modify(diagA, true, s.similarity.g, 2 * diagH, 0.)
end

function num_offdiagonals(s::SimTransOperator{<:Any,<:GutzwillerSampling,<:Any}, add)
    return num_offdiagonals(s.op, add)
end

function get_offdiagonal(s::SimTransOperator{<:Any,<:GutzwillerSampling,<:Any}, add, chosen)
    newadd, offd = get_offdiagonal(s.op, add, chosen)
    # Gutzwiller transformation is diagonal
    diagH1 = diagonal_element(s.similarity.hamiltonian, add)
    diagH2 = diagonal_element(s.similarity.hamiltonian, newadd)
    return newadd, gutzwiller_modify(offd, true, s.similarity.g, diagH1 + diagH2, 0.)
end

# Gutzwiller sampling - special case `f^2`
function SimTransOperator(k)
    T = eltype(k)
    return SimTransOperator{T,typeof(k),Nothing}(k,nothing)
end
LOStructure(::Type{<:SimTransOperator{<:Any,<:GutzwillerSampling,Nothing}}) = IsDiagonal()

function diagonal_element(s::SimTransOperator{<:Any,<:GutzwillerSampling,Nothing}, add)
    diagH = diagonal_element(s.similarity.hamiltonian, add)
    return gutzwiller_modify(1., true, s.similarity.g, 2 * diagH, 0.)
end

num_offdiagonals(s::SimTransOperator{<:Any,<:GutzwillerSampling,Nothing}, add) = 0