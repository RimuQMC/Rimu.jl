"""
    OperatorProduct(A::AbstractOperator, B::AbstractOperator)

The product of two operators.
"""
struct OperatorProduct{T, O1<:AbstractOperator{T}, O2<:AbstractOperator{T}} <: AbstractOperator{T}
    op1::O1
    op2::O2
end

Base.:*(op1::AbstractOperator, op2::AbstractOperator) = OperatorProduct(op1, op2)

struct ProductColumn{A,T,O<:OperatorProduct{T},C}
    operator::O
    address::A
    col2::C
end

function Base.show(io::IO, ::MIME"text/plain", c::ProductColumn)
    print(io, "ProductColumn(operator=$(c.operator), address=$(c.address))")
end

operator_column(o::OperatorProduct, a) = ProductColumn(o, a, operator_column(o.op2, a))

function random_element(c::ProductColumn)
    b_add, b_prob, b_elem = random_element(c.col2)
    a_add, a_prob, a_elem = random_element(operator_column(c.operator.op1, b_add))
    return a_add, a_prob*b_prob, a_elem*b_elem
end

Base.IteratorSize(::ProductColumn) = SizeUnknown()

function Base.iterate(c::ProductColumn)
    first2 = iterate(c.col2)
    if isnothing(first2)
        return nothing
    end
    (add2, val2), state2 = first2
    col1 = operator_column(c.operator.op1, add2)
    first1 = iterate(col1)
    while isnothing(first1)#iterate over op2 until we find a non-empty column of op1, or reach the end
        next2 = iterate(c.col2, state2)
        if isnothing(next2)
            return nothing
        end
        (add2, val2), state2 = next2
        col1 = operator_column(c.operator.op1, add2)
        first1 = iterate(col1)
    end
    (add1, val1), state1 = first1
    state = (state2, state1, col1, val2)
    return (add1 => val1*val2, state)
end

function Base.iterate(c::ProductColumn, state)
    state2, state1, col1, val2 = state
    next1 = iterate(col1, state1)
    while isnothing(next1)
        next2 = iterate(c.col2, state2)
        if isnothing(next2)
            return nothing
        end
        (add2, val2), state2 = next2
        col1 = operator_column(c.operator.op1, add2)
        next1 = iterate(col1)
    end
    (add1, val1), state1 = next1
    state = (state2, state1, col1, val2)
    return (add1 => val1*val2, state)
end
