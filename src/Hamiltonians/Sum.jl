"""
    HamiltonianSum(A::AbstractHamiltonian, B::AbstractHamiltonian; weight=0.5) <: AbstractHamiltonian
    add(A::AbstractHamiltonian, B::AbstractHamiltonian, [a=1, b=1]; weight=0.5) -> HamiltonianSum
    +(A::AbstractHamiltonian, B::AbstractHamiltonian)

The sum of two [`AbstractHamiltonian`](@ref)s, ``A + B``. The two Hamiltonians must act
on the same address space. The keyword argument `weight` affects random spawning
with [`random_offdiagonal`](@ref) and determines the probability of random spawns
from `A`, with `1 - weight` the probability of spawning from `B`.

If coefficients `a` and `b` are given, the Hamiltonians are scaled with [`ScaledHamiltonian`](@ref),
to represent ``aA + bB``.

See also [`ShiftedHamiltonian`](@ref), [`ScaledHamiltonian`](@ref),
[`HamiltonianProduct`](@ref), [`AbstractHamiltonian`](@ref).
"""
struct HamiltonianSum{T, H1<:AbstractHamiltonian, H2<:AbstractHamiltonian} <: AbstractHamiltonian{T}
    h1::H1
    h2::H2
    weight::Float64
end
function HamiltonianSum(
    h1::AbstractHamiltonian{T1}, h2::AbstractHamiltonian{T2}; weight=0.5
) where {T1, T2}
    if !(allows_address_type(h2, starting_address(h1))) ||
        !(allows_address_type(h1, starting_address(h2)))
        throw(ArgumentError("The Hamiltonians are not compatible."))
    end
    T = promote_type(T1,T2)
    return HamiltonianSum{T, typeof(h1), typeof(h2)}(h1, h2, min(abs(weight), 1.0))
end
Base.:+(h1::AbstractHamiltonian, h2::AbstractHamiltonian) = HamiltonianSum(h1, h2)

function Base.show(io::IO, s::HamiltonianSum)
    print(io, "HamiltonianSum(", s.h1, ", ", s.h2, "; weight=", s.weight, ")")
end

@doc (@doc HamiltonianSum)
function VectorInterface.add(
    h1::AbstractHamiltonian, h2::AbstractHamiltonian, a::Number=1, b::Number=1; weight=0.5
)
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
    first1 = iterate(o.ods1)
    if isnothing(first1)
        first2 = iterate(o.ods2)
        if isnothing(first2)
            return nothing
        end
        (add, val), state = first2
        return add => val, (state, false)
    end
    (add, val), state = first1
    return add => val, (state, true)
end

function Base.iterate(o::SumOffdiagonals, state)
    if state[2] # iterating offdiagonals of first Hamiltonian
        next1 = iterate(o.ods1, state[1])
        if isnothing(next1)
            first2 = iterate(o.ods2)
            if isnothing(first2)
                return nothing
            end
            (add, val), state = first2
            return add => val, (state, false)
        end
        (add, val), state = next1
        return add => val, (state, true)
    else
        next2 = iterate(o.ods2, state[1])
        if isnothing(next2)
            return nothing
        end
        (add, val), state = next2
        return add => val, (state, false)
    end
end

"""
    ShiftedHamiltonian(h::AbstractHamiltonian, shift::Number) <: ModifiedHamiltonian
    add(h::AbstractHamiltonian, shift::UniformScaling) -> ShiftedHamiltonian
    h + shift * I

A Hamiltonian that has been shifted by a scalar value. In combination with
[`ScaledHamiltonian`](@ref), this can be used to represent a Hamiltonian of the form
``αH + βI``. Composite Hamiltonians constructed in this way are efficient for usage with
deterministic and stochastic operations.

## Example
```jldoctest
julia> hamiltonian = HubbardRealSpace(BoseFS(1,1));

julia> hamiltonian - 2I == ShiftedHamiltonian(hamiltonian, -2) == add(hamiltonian, -2I)
true

julia> Matrix(hamiltonian)
3×3 Matrix{Float64}:
  0.0      -2.82843  -2.82843
 -2.82843   1.0       0.0
 -2.82843   0.0       1.0

julia> Matrix(hamiltonian - 2I)
3×3 Matrix{Float64}:
 -2.0      -2.82843  -2.82843
 -2.82843  -1.0       0.0
 -2.82843   0.0      -1.0

julia> transition_operator  = I - im * 0.1 * hamiltonian
(1.0 + 0.0im)I + (-0.0 - 0.1im) * HubbardRealSpace(
  fs"|1 1⟩";
  geometry = CubicGrid((2,), (true,)),
  t = [1.0;;],
  u = [1.0;;],
)

julia> Matrix(transition_operator)
3×3 Matrix{ComplexF64}:
 1.0+0.0im       0.0+0.282843im  0.0+0.282843im
 0.0+0.282843im  1.0-0.1im       0.0+0.0im
 0.0+0.282843im  0.0+0.0im       1.0-0.1im
```

See also [`HamiltonianSum`](@ref), [`ScaledHamiltonian`](@ref),
[`ModifiedHamiltonian`](@ref), and [`AbstractHamiltonian`](@ref).
"""
struct ShiftedHamiltonian{T<:Number,H} <: ModifiedHamiltonian{T}
    hamiltonian::H
    shift::T
end

function ShiftedHamiltonian(h::AbstractHamiltonian{T1}, shift::T2) where {T1,T2<:Number}
    T = promote_type(T1,T2)
    return ShiftedHamiltonian{T, typeof(h)}(h, T(shift))
end

function ShiftedHamiltonian(h::ShiftedHamiltonian, shift::Number)
    return ShiftedHamiltonian(h.hamiltonian, h.shift + shift)
end

function Base.show(io::IO, s::ShiftedHamiltonian{T}) where {T}
    if T <: Real
        print(io, s.shift, "I + ", s.hamiltonian)
    else
        print(io, "(", s.shift, ")", "I + ", s.hamiltonian)
    end
end

function LinearAlgebra.adjoint(s::ShiftedHamiltonian)
    return ShiftedHamiltonian(s.hamiltonian', conj(s.shift))
end

function LOStructure(::Type{<:ShiftedHamiltonian{T,H}}) where {T,H}
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

parent_operator(s::ShiftedHamiltonian) = s.hamiltonian
modify_diagonal(s::ShiftedHamiltonian, _, value) = value + s.shift
modify_offdiagonal(s::ShiftedHamiltonian, _, addr, value) = addr => value

@doc (@doc ShiftedHamiltonian)
function VectorInterface.add(
    h::AbstractHamiltonian, shift::UniformScaling{T}, alpha::Number, beta::Number
) where {T<:Number}
    return ShiftedHamiltonian(alpha * h, beta * shift.λ)
end
function VectorInterface.add(
    shift::UniformScaling{T}, h::AbstractHamiltonian, alpha::Number, beta::Number
) where {T<:Number}
    return add(h, shift, beta, alpha)
end

@doc (@doc ShiftedHamiltonian)
function Base.:+(h::AbstractHamiltonian, shift::UniformScaling{T}) where {T<:Number}
    return ShiftedHamiltonian(h, shift.λ)
end
Base.:+(shift::UniformScaling{T}, h::AbstractHamiltonian) where {T<:Number} = h + shift
Base.:-(h::AbstractHamiltonian, shift::UniformScaling{T}) where {T<:Number} = h + (-shift)
function Base.:-(shift::UniformScaling{T}, h::AbstractHamiltonian) where {T<:Number}
    scale(h, -1) + shift
end
