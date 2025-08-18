"""
    add(A::AbstractHamiltonian, B::AbstractHamiltonian; a=1, b=1, weight=0.5) -> HamiltonianSum
    HamiltonianSum(A::AbstractHamiltonian, B::AbstractHamiltonian; weight=0.5)
    +(A::AbstractHamiltonian, B::AbstractHamiltonian)

The sum of two [`AbstractHamiltonian`](@ref)s with coefficients, `aA + bB`. The two
Hamiltonians must act on the same address space. The keyword argument `weight` is the
probability of random spawns from `A`, with `1 - weight` the probability of spawning from
`B`.
"""
struct HamiltonianSum{T, H1<:AbstractHamiltonian, H2<:AbstractHamiltonian} <: AbstractHamiltonian{T}
    h1::H1
    h2::H2
    weight::Float64
end
function HamiltonianSum(h1::AbstractHamiltonian{T1}, h2::AbstractHamiltonian{T2}; weight=0.5) where {T1, T2}
    if !(allows_address_type(h2, starting_address(h1))) || !(allows_address_type(h1, starting_address(h2)))
        throw(ArgumentError("The Hamiltonians are not compatible."))
    end
    T = promote_type(T1,T2)
    return HamiltonianSum{T, typeof(h1), typeof(h2)}(h1, h2, min(abs(weight), 1.0))
end
Base.:+(h1::AbstractHamiltonian, h2::AbstractHamiltonian) = HamiltonianSum(h1, h2)

function Base.show(io::IO, s::HamiltonianSum)
    print(io, "HamiltonianSum(", s.h1, ", ", s.h2, "; weight=", s.weight, ")")
end

function VectorInterface.add(h1::AbstractHamiltonian, h2::AbstractHamiltonian, a::Number, b::Number; weight=0.5)
    return HamiltonianSum(a*h1, b*h2; weight)
end

starting_address(s::HamiltonianSum) = starting_address(s.h1)

function allows_address_type(s::HamiltonianSum, ::Type{A}) where {A}
    return allows_address_type(s.h2, A) && allows_address_type(s.h1, A)
end

function LOStructure(::Type{<:HamiltonianSum{<:Any,H1,H2}}) where {H1,H2}
    l1 = LOStructure(H1)
    l2 = LOStructure(H2)
    if l1 == IsDiagonal() && l2 == IsDiagonal()
        return IsDiagonal()
    elseif l1 == IsHermitian() && l2 == IsHermitian()
        return IsHermitian()
    elseif l1 != AdjointUnknown() && l2 != AdjointUnknown()
        return AdjointKnown()
    else
        return AdjointUnknown()
    end
end

function LinearAlgebra.adjoint(s::HamiltonianSum)
    return HamiltonianSum(s.h1', s.h2'; weight=s.weight)
end

function has_iterable_offdiagonals(::Type{<:HamiltonianSum{<:Any,H1,H2}}) where {H1,H2}
    return has_iterable_offdiagonals(H1) && has_iterable_offdiagonals(H2)
end

function has_random_offdiagonal(::Type{<:HamiltonianSum{<:Any,H1,H2}}) where {H1, H2}
    return has_random_offdiagonal(H1) && has_random_offdiagonal(H2)
end

struct SumColumn{A,T,O<:HamiltonianSum{T},C1,C2} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    col1::C1
    col2::C2
    weight::Float64
end
function operator_column(s::HamiltonianSum, add)
    return SumColumn(s, add, operator_column(s.h1,add), operator_column(s.h2,add), s.weight)
end

parent_operator(c::SumColumn) = c.operator
starting_address(c::SumColumn) = c.address
num_offdiagonals(c::SumColumn) = num_offdiagonals(c.col1) + num_offdiagonals(c.col2)

function diagonal_element(c::SumColumn{<:Any,T}) where {T}
    return T(diagonal_element(c.col1) + diagonal_element(c.col2))
end

function random_offdiagonal(c::SumColumn{<:Any,T}) where {T}
    if rand() < c.weight
        add, prob, val = random_offdiagonal(c.col1)
        return add, prob*c.weight, T(val)
    else
        add, prob, val = random_offdiagonal(c.col2)
        return add, prob*(1-c.weight), T(val)
    end
end

struct SumOffdiagonals{A,T,O<:HamiltonianSum{T},OD1,OD2}
    operator::O
    address::A
    ods1::OD1
    ods2::OD2
end
function offdiagonals(c::SumColumn)
    return SumOffdiagonals(c.operator, c.address, offdiagonals(c.col1), offdiagonals(c.col2))
end

Base.IteratorSize(::SumOffdiagonals) = Base.SizeUnknown()
Base.eltype(::SumOffdiagonals{A,T}) where {A,T} = Pair{A,T}

function Base.iterate(o::SumOffdiagonals)
    first = iterate(o.ods1)
    if isnothing(first)
        first = iterate(o.ods2)
        if isnothing(first)
            return nothing
        end
        (add, val), state = first
        return add => val, (state, false)
    end
    (add, val), state = first
    return add => val, (state, true)
end

function Base.iterate(o::SumOffdiagonals, state)
    if state[2]
        next = iterate(o.ods1, state[1])
        if isnothing(next)
            first = iterate(o.ods2)
            if isnothing(first)
                return nothing
            end
            (add, val), state = first
            return add => val, (state, false)
        end
        (add, val), state = next
        return add => val, (state, true)
    else
        next = iterate(o.ods2, state[1])
        if isnothing(next)
            return nothing
        end
        (add, val), state = next
        return add => val, (state, false)
    end
end
