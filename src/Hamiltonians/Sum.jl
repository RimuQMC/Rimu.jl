"""
    HamiltonianSum(A::AbstractHamiltonian, B::AbstractHamiltonian; weight=0.5) <: AbstractHamiltonian
    add(A::AbstractHamiltonian, B::AbstractHamiltonian, [a=1, b=1]; weight=0.5) -> HamiltonianSum
    +(A::AbstractHamiltonian, B::AbstractHamiltonian)

The sum of two [`AbstractHamiltonian`](@ref)s, ``A + B``. The two Hamiltonians must act
on the same address space. The keyword argument `weight` affects random spawning
with [`random_offdiagonal`](@ref) and determines the probability of random spawns
from `A`, with `1 - weight` the probability of spawning from `B`.

If coefficients `a` and `b` are given, the Hamiltonians are combined as
``aA + bB`` with the same address compatibility checks as `A + B`.

See also [`HamiltonianProduct`](@ref), [`AbstractHamiltonian`](@ref).
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
    h1::AbstractHamiltonian, h2::AbstractHamiltonian, a::Number=One(), b::Number=One(); weight=0.5
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
    ScaledOrShiftedHamiltonian(H::AbstractHamiltonian, alpha, beta) <: ModifiedHamiltonian
    +(H::AbstractHamiltonian, shift::UniformScaling)
    add(shift::UniformScaling, H::AbstractHamiltonian, [alpha, beta])
    *(alpha::Number, H::AbstractHamiltonian)
    scale(H, alpha)
    alpha * H + beta * I -> ScaledOrShiftedHamiltonian(H, alpha, beta)

A Hamiltonian that has been scaled by a scalar, `alpha * H`, or shifted by a scalar,
`H + beta * I`, or both, `alpha * H + beta * I`. Note that scaling and shifting a
Hamiltonian by a scalar using `add` or `+` requires a `UniformScaling` object, which is
created by multiplying a scalar with `LinearAlgebra.I`, representing a unit matrix.

Scaling is applied to all matrix elements of the Hamiltonian, while shifting only affects
the diagonal elements. As a consequence the eigenvalues of the Hamiltonian are scaled and
shifted, while the eigenvectors remain unchanged. Composite Hamiltonians constructed in this
way are efficient for usage with deterministic and stochastic operations. Nested and
consecutive scaling and shifting operations are automatically combined into a single
`ScaledOrShiftedHamiltonian` object.

## Example
```jldoctest
julia> H = HubbardReal1D(BoseFS(1,1));

julia> ssh = 3*H + 2*I
3*HubbardReal1D(fs"|1 1⟩"; u=1.0, t=1.0) + 2*I

julia> using Rimu.Hamiltonians: ScaledOrShiftedHamiltonian

julia> ssh == add(2*I, H, 3) == ScaledOrShiftedHamiltonian(H, 3, 2)
true

julia> Matrix(H)
3×3 Matrix{Float64}:
  0.0      -2.82843  -2.82843
 -2.82843   1.0       0.0
 -2.82843   0.0       1.0

julia> Matrix(3*H + 2*I)
3×3 Matrix{Float64}:
  2.0      -8.48528  -8.48528
 -8.48528   5.0       0.0
 -8.48528   0.0       5.0
```
!!! warning "Warning"
    The `ScaledOrShiftedHamiltonian` type is an implementation detail and may change in
    future versions of Rimu. Use the public interface functions [`add`](@ref), [`+`](@ref)
    and [`scale`](@ref) to construct scaled and shifted Hamiltonians.

See also [`HamiltonianSum`](@ref), [`HamiltonianProduct`](@ref),
[`ModifiedHamiltonian`](@ref), and [`AbstractHamiltonian`](@ref).
"""
struct ScaledOrShiftedHamiltonian{T, TA, TB, H} <: ModifiedHamiltonian{T}
    hamiltonian::H
    alpha::TA
    beta::TB
end

function ScaledOrShiftedHamiltonian(
    h::AbstractHamiltonian{TH}, alpha::TA, beta::TB
) where {TH,TA<:Number,TB<:Number}
    T = promote_type(TA, TB, TH)
    return ScaledOrShiftedHamiltonian{T, TA, TB, typeof(h)}(h, alpha, beta)
end
ScaledOrShiftedHamiltonian(h::AbstractHamiltonian, ::One, ::Zero) = h

function ScaledOrShiftedHamiltonian(
    h::ScaledOrShiftedHamiltonian, alpha::Number, beta::Number
)
    return ScaledOrShiftedHamiltonian(h.hamiltonian, alpha * h.alpha, alpha * h.beta + beta)
end

function Base.show(io::IO, s::ScaledOrShiftedHamiltonian{T, TA, TB}) where {T, TA, TB}
    if TA <: One
        print(io, s.hamiltonian)
    elseif TA <: Real
        print(io, s.alpha, "*", s.hamiltonian)
    else
        print(io, "(", s.alpha, ")*", s.hamiltonian)
    end
    if TB <: Zero
        return
    elseif TB <: Real
        print(io, " + ", s.beta, "*I")
    else
        print(io, " + (", s.beta, ")*I")
    end
end

function LinearAlgebra.adjoint(s::ScaledOrShiftedHamiltonian)
    return ScaledOrShiftedHamiltonian(s.hamiltonian', conj(s.alpha), conj(s.beta))
end

function LOStructure(::Type{<:ScaledOrShiftedHamiltonian{T,TA,TB,H}}) where {T,TA,TB,H}
    if LOStructure(H) == IsHermitian()
        if (TA <: Real || TA <: One) && (TB <: Real || TB <: Zero)
            return IsHermitian()
        else
            return AdjointKnown()
        end
    else
        return LOStructure(H)
    end
end

parent_operator(s::ScaledOrShiftedHamiltonian) = s.hamiltonian
function modify_diagonal(s::ScaledOrShiftedHamiltonian{T}, _, value) where {T}
    return T(s.alpha * value + s.beta)
end
function modify_offdiagonal(s::ScaledOrShiftedHamiltonian{T}, _, addr, value) where {T}
    return addr => T(s.alpha * value)
end

"""
    add(shift::UniformScaling, H::AbstractHamiltonian, [alpha, beta])
    add(H::AbstractHamiltonian, shift::UniformScaling, [beta, alpha])

Construct the linear combination `alpha * H + beta * I` where `I` is the identity
operator represented by a `UniformScaling`. This returns a modified operator representing
the Hamiltonian with all matrix elements scaled by `alpha` and all diagonal elements shifted
by `beta`. The coefficients `alpha` and `beta` default to `One()` if not specified.

The coefficient order matches the `add(y, x, α, β)` convention used throughout
`VectorInterface`.

See also [`+`](@ref), [`scale`](@ref), and [`HamiltonianSum`](@ref).
"""
function VectorInterface.add(
    shift::UniformScaling{T}, h::AbstractHamiltonian, alpha::Number, beta::Number
) where {T<:Number}
    return ScaledOrShiftedHamiltonian(h, alpha, shift.λ * beta)
end
function VectorInterface.add(
    h::AbstractHamiltonian, shift::UniformScaling{T}, beta::Number, alpha::Number
) where {T<:Number}
    return add(shift, h, alpha, beta)
end

"""
    +(H::AbstractHamiltonian, shift::UniformScaling)
    H + beta * I

Return a modified Hamiltonian where all diagonal elements are uniformly shifted by a scalar
`beta`. Use the identity operator `I` from `LinearAlgebra` to construct the shift.
The resulting Hamiltonian is equivalent to `add(H, beta * I)`.

See also [`add`](@ref), [`scale`](@ref).
"""
function Base.:+(h::AbstractHamiltonian, shift::UniformScaling{T}) where {T<:Number}
    return ScaledOrShiftedHamiltonian(h, One(), shift.λ)
end
Base.:+(shift::UniformScaling{T}, h::AbstractHamiltonian) where {T<:Number} = h + shift

Base.:-(h::AbstractHamiltonian, shift::UniformScaling{T}) where {T<:Number} = h + (-shift)
function Base.:-(shift::UniformScaling{T}, h::AbstractHamiltonian) where {T<:Number}
    return ScaledOrShiftedHamiltonian(h, -1, shift.λ)
end
Base.:-(h::AbstractHamiltonian) = ScaledOrShiftedHamiltonian(h, -1, Zero())

"""
    scale(H::AbstractHamiltonian, alpha)
    *(alpha::Number, H::AbstractHamiltonian)

Return the scalar multiple `alpha * H` as a modified Hamiltonian wrapper.
All matrix elements are scaled by the factor `alpha`.

See also [`add`](@ref), [`+`](@ref).
"""
function VectorInterface.scale(h::AbstractHamiltonian, alpha::T) where {T<:Number}
    return ScaledOrShiftedHamiltonian(h, alpha, Zero())
end

@doc (@doc scale)
Base.:*(alpha::Number, h::AbstractHamiltonian) = scale(h, alpha)
