"""
    HamiltonianProduct(A::AbstractHamiltonian, B::AbstractHamiltonian; commuting=A==B)
    *(A::AbstractHamiltonian, B::AbstractHamiltonian)

The product of two [`AbstractHamiltonian`](@ref)s, acting from right to left. The two Hamiltonians
must act on the same address space. Set `commuting` to `true` if `A` and `B` commute.
"""
struct HamiltonianProduct{T, O1<:AbstractHamiltonian, O2<:AbstractHamiltonian, C} <: AbstractHamiltonian{T}
    op1::O1
    op2::O2
end
function HamiltonianProduct(
    op1::AbstractHamiltonian{T1}, op2::AbstractHamiltonian{T2};
    commuting = op1==op2 || isdiag(op1) && isdiag(op2)
) where {T1, T2}
    if !allows_address_type(op1, starting_address(op2))
        throw(ArgumentError("The Hamiltonians are not compatible."))
    end
    return HamiltonianProduct{promote_type(T1,T2), typeof(op1), typeof(op2), commuting}(op1, op2)
end
Base.:*(op1::AbstractHamiltonian, op2::AbstractHamiltonian) = HamiltonianProduct(op1, op2)

function allows_address_type(p::HamiltonianProduct, ::Type{A}) where {A}
    return allows_address_type(p.op2, A) && allows_address_type(p.op1, A)
end

starting_address(p::HamiltonianProduct) = starting_address(p.op2)

function LOStructure(::Type{<:HamiltonianProduct{<:Any,O1,O2,C}}) where {O1,O2,C}
    l1 = LOStructure(O1)
    l2 = LOStructure(O2)
    if l1 == IsDiagonal() && l2 == IsDiagonal()
        return IsDiagonal()
    elseif C
        if (l1 == IsHermitian() && l2 == IsHermitian()) ||
            (l1 == IsDiagonal() && eltype(O1) <: Real && l2 == IsHermitian()) ||
            (l2 == IsDiagonal() && eltype(O2) <: Real && l1 == IsHermitian())
            return IsHermitian()
        end
    elseif l1 != AdjointUnknown() && l2 != AdjointUnknown()
        return AdjointKnown()
    else
        return AdjointUnknown()
    end
end
function LinearAlgebra.adjoint(op::HamiltonianProduct{<:Any,<:Any,<:Any,C}) where {C}
    return HamiltonianProduct(op.op2',op.op1'; commuting=C)
end

function has_iterable_offdiagonals(::Type{<:HamiltonianProduct{<:Any,H1,H2}}) where {H1,H2}
    return has_iterable_offdiagonals(H1) && has_iterable_offdiagonals(H2)
end

function has_random_offdiagonal(::Type{<:HamiltonianProduct{<:Any,H1,H2}}) where {H1, H2}
    return has_random_offdiagonal(H1) && has_random_offdiagonal(H2)
end

struct ProductColumn{A,T,O<:HamiltonianProduct{T},C1,C2} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    col1::C1
    col2::C2
end
function operator_column(o::HamiltonianProduct, a)
    return ProductColumn(o, a, operator_column(o.op1, a), operator_column(o.op2, a))
end

parent_operator(c::ProductColumn) = c.operator
starting_address(c::ProductColumn) = c.address
num_offdiagonals(c::ProductColumn) = 2*(num_offdiagonals(c.col2)+1)^2

function diagonal_element(c::ProductColumn)
    #this is not the actual diagonal element, just a contribution to it
    return diagonal_element(c.col2)*diagonal_element(c.col1)
end

function random_offdiagonal(c::ProductColumn)
    p = num_offdiagonals(c.col2)
    if rand() < 1/(p+1)# diagonal element op2
        a_add, a_prob, a_elem = random_offdiagonal(c.col1)
        return a_add, a_prob/(p+1), a_elem*diagonal_element(c.col2)
    else
        b_add, b_prob, b_elem = random_offdiagonal(c.col2)
        b_prob *= p/(p+1)
        col1 = operator_column(c.operator.op1, b_add)
        q = num_offdiagonals(col1)
        if rand() < 1/(q+1)# diagonal element op1
            return b_add, b_prob/(q+1), b_elem*diagonal_element(col1)
        else
            a_add, a_prob, a_elem = random_offdiagonal(col1)
            a_prob *= q/(q+1)
            return a_add, a_prob*b_prob, a_elem*b_elem
        end
    end
end

struct ProductOffdiagonals{A,T,O<:HamiltonianProduct{T},OD1,OD2,S2}
    operator::O
    address::A
    diag2::T
    ods1::OD1
    ods2::OD2
    state2::S2
end
function offdiagonals(c::ProductColumn{<:Any,T}) where {T}
    ods2 = offdiagonals(c.col2)
    first2 = iterate(ods2)
    return ProductOffdiagonals(
        c.operator,
        c.address,
        T(diagonal_element(c.col2)),
        offdiagonals(c.col1),
        ods2,
        isnothing(first2) ? nothing : last(first2)
    )
end

Base.IteratorSize(::ProductOffdiagonals) = Base.SizeUnknown()
Base.eltype(::ProductOffdiagonals{A,T}) where {A,T} = Pair{A,T}

struct ProductIterState{S1,S2,O,T}
    state1::Union{Nothing,S1}
    state2::Union{Nothing,S2}
    ods1::Union{Nothing,O}
    val2::T
end

function Base.iterate(o::ProductOffdiagonals{<:Any,T,<:Any,OD1,<:Any,S2}) where {T,OD1,S2}
    #start with diagonal of op2, offdiagonals of op1
    first1 = iterate(o.ods1)
    if isnothing(first1)# no offdiagonals for op1, go to offdiagonals of op2
        first2 = iterate(o.ods2)
        if isnothing(first2)
            return nothing
        end
        (add2, val2), state2 = first2
        col1 = operator_column(o.operator.op1, add2)
        state = ProductIterState{Nothing,S2,OD1,T}(nothing, state2, offdiagonals(col1), val2)
        return add2 => diagonal_element(col1)*val2, state
    else
        (add1, val1), state1 = first1
        state = ProductIterState{typeof(state1),S2,OD1,T}(state1, nothing, nothing, o.diag2)
        return add1 => val1*o.diag2, state
    end
end

function Base.iterate(o::ProductOffdiagonals, state::ProductIterState{S1,S2,OD1,T}) where {S1,S2,OD1,T}
    (;state1, state2, ods1, val2) = state
    if isnothing(state2)# diagonal of op2, iterating op1
        next1 = iterate(o.ods1, state1)
        if isnothing(next1)
            first2 = iterate(o.ods2)
            if isnothing(first2)
                return nothing
            end
            (add2, val2), state2 = first2
            col1 = operator_column(o.operator.op1, add2)
            state = ProductIterState{S1,S2,OD1,T}(nothing, state2, offdiagonals(col1), val2)
            return add2 => diagonal_element(col1)*val2, state
        else
            (add1, val1), state1 = next1
            state = ProductIterState{S1,S2,OD1,T}(state1, nothing, nothing, o.diag2)
            return add1 => val1*o.diag2, state
        end
    elseif isnothing(state1)# just did diagonal element of op1
        first1 = iterate(ods1)
        if isnothing(first1)# no offdiagonals for op1, go back to op2
            next2 = iterate(o.ods2, state2)
            if isnothing(next2)
                return nothing
            end
            (add2, val2), state2 = next2
            col1 = operator_column(o.operator.op1, add2)
            state = ProductIterState{S1,S2,OD1,T}(nothing, state2, offdiagonals(col1), val2)
            return add2 => diagonal_element(col1)*val2, state
        else
            (add1, val1), state1 = first1
            return add1 => val1*val2, ProductIterState{typeof(state1),S2,OD1,T}(state1, state2, ods1, val2)
        end
    else# we have op1 offdiagonals and its state
        next1 = iterate(ods1, state1)
        if isnothing(next1)# reached the end of op1 column, go back to op2
            next2 = iterate(o.ods2, state2)
            if isnothing(next2)
                return nothing
            end
            (add2, val2), state2 = next2
            col1 = operator_column(o.operator.op1, add2)
            return add2 => diagonal_element(col1)*val2, ProductIterState{S1,S2,OD1,T}(nothing, state2, offdiagonals(col1), val2)
        else
            (add1, val1), state1 = next1
            return add1 => val1*val2, ProductIterState{S1,S2,OD1,T}(state1, state2, ods1, val2)
        end
    end
end

"""
    ScaledHamiltonian(H::AbstractHamiltonian, α) <: AbstractHamiltonian
    scale(H, α)
    α * H

The product of the Hamiltonian `H` with the scalar `α`.

See also [`HamiltonianSum`](@ref), [`HamiltonianProduct`](@ref), [`AbstractHamiltonian`](@ref).
"""
struct ScaledHamiltonian{T,H} <: ModifiedHamiltonian{T}
    hamiltonian::H
    α::T
end

function ScaledHamiltonian(h::AbstractHamiltonian{T1}, α::T2) where {T1,T2}
    T = promote_type(T1,T2)
    ScaledHamiltonian{T, typeof(h)}(h, T(α))
end

function ScaledHamiltonian(h::ScaledHamiltonian, β::Number)
    return ScaledHamiltonian(h.hamiltonian, h.α*β)
end

function Base.show(io::IO, h::ScaledHamiltonian{T}) where {T}
    if T <: Real
        print(io, h.α, " * ", h.hamiltonian)
    else
        print(io, "(", h.α, ") * ", h.hamiltonian)
    end
end

function LOStructure(::Type{<:ScaledHamiltonian{T,H}}) where {T,H}
    if LOStructure(H) == IsHermitian()
        if T <: Real
            return IsHermitian()
        else
            return AdjointKnown()
        end
    else
        return LOStructure(H)
    end
end

function LinearAlgebra.adjoint(h::ScaledHamiltonian)
    return ScaledHamiltonian(h.hamiltonian', conj(h.α))
end

parent_operator(h::ScaledHamiltonian) = h.hamiltonian
modify_diagonal(h::ScaledHamiltonian, _, value) = value*h.α
modify_offdiagonal(h::ScaledHamiltonian, _, addr, value) = addr => value*h.α

@doc (@doc ScaledHamiltonian)
function VectorInterface.scale(h::AbstractHamiltonian, α::T) where {T<:Number}
    if α == 1
        return h
    end
    return ScaledHamiltonian(h, α)
end

Base.:*(α::Number, h::AbstractHamiltonian) = scale(h, α)
