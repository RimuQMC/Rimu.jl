"""
    Stoquastic(ham <: AbstractHamiltonian) <: AbstractHamiltonian
A wrapper for an [`AbstractHamiltonian`](@ref) that replaces all off-diagonal matrix
elements `v` by `-abs(v)`, thus making the new Hamiltonian *stoquastic*.

A stoquastic Hamiltonian does not have a Monte Carlo sign problem. For a hermitian `ham`
the smallest eigenvalue of `Stoquastic(ham)` is ≤ the smallest eigenvalue of `ham`.
"""
struct Stoquastic{T,H} <: ModifiedHamiltonian{T}
    hamiltonian::H
end

Stoquastic(h) = Stoquastic{eltype(h),typeof(h)}(h)

LOStructure(::Type{<:Stoquastic{<:Any,H}}) where {H} = LOStructure(H)
Base.adjoint(h::Stoquastic) = Stoquastic(h.hamiltonian')

parent_operator(h::Stoquastic) = h.hamiltonian
modify_diagonal(h::Stoquastic, _, value) = value
modify_offdiagonal(h::Stoquastic{T}, _, addr, value) where {T} = addr => T(-abs(value))
