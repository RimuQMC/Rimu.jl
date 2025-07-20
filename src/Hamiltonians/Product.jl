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

struct ProductColumn{A,T,O<:HamiltonianProduct{T},C1,C2} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    col1::C1
    col2::C2
end
function operator_column(o::HamiltonianProduct, a) 
    return ProductColumn(o, a, operator_column(o.op1, a), operator_column(o.op2, a))
end

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

struct ProductOffdiagonals{A,T,O<:HamiltonianProduct{T},C,OD1,OD2}
    operator::O
    address::A
    col2::C
    ods1::OD1
    ods2::OD2
end
offdiagonals(c::ProductColumn) = ProductOffdiagonals(c.operator, c.address, c.col2, offdiagonals(c.col1), offdiagonals(c.col2))
Base.IteratorSize(::ProductOffdiagonals) = Base.SizeUnknown()
Base.eltype(::ProductOffdiagonals{A,T}) where {A,T} = Pair{A,T}

function Base.iterate(o::ProductOffdiagonals)
    #start with diagonal of op2, offdiagonals of op1
    first1 = iterate(o.ods1)
    if isnothing(first1)# no offdiagonals for op1, go to offdiagonals of op2
        first2 = iterate(o.ods2)
        if isnothing(first2)
            return nothing
        end
        (add2, val2), state2 = first2
        col1 = operator_column(o.operator.op1, add2)
        val1 = diagonal_element(col1)
        state = (state2, col1, val2)
        return add2 => val1*val2, state
    else
        (add1, val1), state1 = first1
        state = (o.ods1, state1)
        return add1 => val1*diagonal_element(o.col2), state
    end
end

function Base.iterate(o::ProductOffdiagonals, state)
    if length(state) == 2# diagonal of op2, iterating op1
        ods1, state1 = state
        next1 = iterate(ods1, state1)
        if isnothing(next1)
            first2 = iterate(o.ods2)
            if isnothing(first2)
                return nothing
            end
            (add2, val2), state2 = first2
            col1 = operator_column(o.operator.op1, add2)
            val1 = diagonal_element(col1)
            state = (state2, col1, val2)
            return add2 => val1*val2, state
        else
            (add1, val1), state1 = next1
            state = (ods1, state1)
            return add1 => val1*diagonal_element(o.col2), state
        end
    elseif length(state) == 3# just did diagonal element of op1, we have the column
        state2, col1, val2 = state
        ods1 = offdiagonals(col1)
        first1 = iterate(ods1)
        if isnothing(first1)# no offdiagonals for op1, go back to op2
            next2 = iterate(o.ods2, state2)
            if isnothing(next2)
                return nothing
            end
            (add2, val2), state2 = next2
            col1 = operator_column(o.operator.op1, add2)
            val1 = diagonal_element(col1)
            state = (state2, col1, val2)
            return add2 => val1*val2, state
        else
            (add1, val1), state1 = first1
            state = (state1, state2, ods1, val2)
            return add1 => val1*val2, state
        end
    else# we have op1 offdiagonals and its state
        state1, state2, ods1, val2 = state
        next1 = iterate(ods1, state1)
        if isnothing(next1)# reached the end of op1 column, go back to op2
            next2 = iterate(o.ods2, state2)
            if isnothing(next2)
                return nothing
            end
            (add2, val2), state2 = next2
            col1 = operator_column(o.operator.op1, add2)
            val1 = diagonal_element(col1)
            state = (state2, col1, val2)
            return add2 => val1*val2, state
        else
            (add1, val1), state1 = next1
            state = (state1, state2, ods1, val2)
            return add1 => val1*val2, state
        end
    end
end
