"""
    OperatorProduct(A::AbstractOperator, B::AbstractOperator)

The product of two operators.
"""
struct OperatorProduct{T, O1<:AbstractOperator, O2<:AbstractOperator} <: AbstractOperator{T}
    op1::O1
    op2::O2
end
function OperatorProduct(op1::AbstractOperator{T1}, op2::AbstractOperator{T2}) where {T1, T2}
    return OperatorProduct{promote_type(T1,T2), typeof(op1), typeof(op2)}(op1, op2)
end
Base.:*(op1::AbstractOperator, op2::AbstractOperator) = OperatorProduct(op1, op2)

allows_address_type(p::OperatorProduct, ::Type{A}) where {A} = allows_address_type(p.op2, A)

function LOStructure(::Type{<:OperatorProduct{<:Any,O1,O2}}) where {O1,O2}
    l1 = LOStructure(O1)
    l2 = LOStructure(O2)
    if l1 == IsHermitian() && l2 == IsHermitian()
        return IsHermitian()
    elseif l1 == IsDiagonal() && l2 == IsDiagonal()
        return IsDiagonal()
    elseif l1 != AdjointUnknown() && l2 != AdjointUnknown()
        return AdjointKnown()
    else
        return AdjointUnknown()
    end
end
function LinearAlgebra.adjoint(op::OperatorProduct)
    return OperatorProduct(op.op2',op.op1')
end

struct ProductColumn{A,T,O<:OperatorProduct{T},C} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    col2::C
end
operator_column(o::OperatorProduct, a) = ProductColumn(o, a, operator_column(o.op2, a))
starting_address(c::ProductColumn) = c.address
num_offdiagonals(c::ProductColumn) = 2*(num_offdiagonals(c.col2)+1)^2

function diagonal_element(c::ProductColumn)
    #this is not the actual diagonal element, just a contribution to it, but this is ok
    return diagonal_element(c.col2)*diagonal_element(operator_column(c.operator.op1, starting_address(c)))
end

function random_offdiagonal(c::ProductColumn)
    p = num_offdiagonals(c.col2)
    if rand() < 1/(p+1)
        b_add = starting_address(c)
        b_prob = 1/(p+1)
        b_elem = diagonal_element(c.col2)
        a_add, a_prob, a_elem = random_offdiagonal(operator_column(c.operator.op1, b_add))
        return a_add, a_prob*b_prob, a_elem*b_elem
    else
        b_add, b_prob, b_elem = random_offdiagonal(c.col2)
        b_prob *= p/(p+1)
        col1 = operator_column(c.operator.op1, b_add)
        q = num_offdiagonals(col1)
        if rand() < 1/(q+1)
            a_elem = diagonal_element(col1)
            a_prob = 1/(q+1)
            return b_add, a_prob*b_prob, a_elem*b_elem
        else
            a_add, a_prob, a_elem = random_offdiagonal(col1)
            a_prob *= q/(q+1)
            return a_add, a_prob*b_prob, a_elem*b_elem
        end
    end
end

struct ProductOffdiagonals{A,T,O<:OperatorProduct{T},OD}
    operator::O
    address::A
    ods2::OD
end
offdiagonals(c::ProductColumn) = ProductOffdiagonals(c.operator, c.address, offdiagonals(c.col2))
Base.IteratorSize(::ProductOffdiagonals) = Base.SizeUnknown()
Base.eltype(::ProductOffdiagonals{A,T}) where {A,T} = Pair{A,T}

function Base.iterate(o::ProductOffdiagonals)
    first2 = iterate(o.ods2)
    if isnothing(first2)
        return nothing
    end
    (add2, val2), state2 = first2
    col1 = operator_column(o.operator.op1, add2)
    val1 = diagonal_element(col1)
    state = (state2, col1, val2)
    return add2 => val1*val2, state
end

function Base.iterate(o::ProductOffdiagonals, state)
    if length(state) == 3# just did diagonal element of op1, we have the column
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
        end
        (add1, val1), state1 = first1
        state = (state1, state2, ods1, val2)
        return add1 => val1*val2, state
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
        end
        (add1, val1), state1 = next1
        state = (state1, state2, ods1, val2)
        return add1 => val1*val2, state
    end
end
