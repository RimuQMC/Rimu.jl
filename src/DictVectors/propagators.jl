"""
    abstract type AbstractPropagator{T}

An operator equipped with an instance of [`PDWorkingMemory`](@ref). The main operation on
it is [`propagate!`](@ref), which performs a (stochastic) analog of operator application
and returns spawning statistics.

An `AbstractPropagator` must define either [`spawn_column!`](@ref) and
[`working_memory`](@ref), or [`propagate!`](@ref).

See [`FCIQMCPropagator`](@ref) and [`OperatorMulPropagator`](@ref).
"""
abstract type AbstractPropagator{T} end
Base.eltype(::Type{<:AbstractPropagator{T}}) where {T} = T

"""
# TODO
"""
spawn_column!

function propagate!(dst::PDVec, prop::AbstractPropagator, src)
    w = working_memory(prop)
    stats = perform_spawns!(w, src, prop)
    collect_local!(w)
    synchronize_remote!(w)
    move_and_compress!(dst, w)
    return stats
end

function LinearAlgebra.mul!(dst, prop::AbstractPropagator, src)
    propagate!(dst, prop, src)
    return dst
end

function Base.:*(prop::AbstractPropagator, v::PDVec)
    w = similar(v, promote_type(eltype(prop), valtype(v)))
    return mul!(w, prop, v)
end
(prop::AbstractPropagator)(v) = prop * v

"""
    FCIQMCPropagator(hamiltonian, shift, dτ, working_memory)

[`AbstractPropagator`](@ref) that performs

```math
w = v + dτ (S - H) v
```

where ``S`` is the `shift`, ``H`` the `hamiltonian`, and ``w`` and ``v`` the vectors.
"""
struct FCIQMCPropagator{H,S,T,W<:PDWorkingMemory{<:Any,T}} <: AbstractPropagator{T}
    hamiltonian::H
    shift::S
    dτ::Float64
    working_memory::W
end

working_memory(f::FCIQMCPropagator) = f.working_memory

function spawn_column!(column, f::FCIQMCPropagator, k, v)
    return fciqmc_col!(
        f.working_memory.style, column, f.hamiltonian, k, v, f.shift, f.dτ
    )
end

"""
    OperatorMulPropagator(operator, ::PDVec) <: AbstractPropagator
    OperatorMulPropagator(operator, ::PDWorkingMemory) <: AbstractPropagator

[`AbstractPropagator`](@ref) that performs matrix-vector multiplications.

If a vector is passed to the constructor, a working memory with a deterministic stochastic
style is created.
"""
struct OperatorMulPropagator{O,T,W<:PDWorkingMemory{<:Any,T}} <: AbstractPropagator{T}
    operator::O
    working_memory::W
end
function OperatorMulPropagator(operator, t::PDVec)
    if eltype(operator) === valtype(t)
        wm = PDWorkingMemory(t; style=IsDeterministic{eltype(operator)}())
    else
        T = promote_type(eltype(operator), valtype(t))
        wm = PDWorkingMemory(similar(t, T); style=IsDeterministic{T}())
    end
    return OperatorMulPropagator(operator, wm)
end

working_memory(o::OperatorMulPropagator) = o.working_memory

function spawn_column!(column, o::OperatorMulPropagator, k, v)
    T = eltype(o.operator)
    return fciqmc_col!(
        o.working_memory.style, column, o.operator, k, v, one(T), -one(T)
    )
end

###
### Operator linear algebra operations
###
"""
    mul!(y::PDVec, A::AbstractHamiltonian, x::PDVec, w::PDWorkingMemory)

Perform `y = A * x`. The working memory `w` is required to facilitate threaded/distributed
operations. `y` and `x` may be the same vector.
"""
function LinearAlgebra.mul!(dst::PDVec, op::AbstractHamiltonian, src::PDVec, w=PDWorkingMemory(dst))
    prop = OperatorMulPropagator(op, w)
    return mul!(dst, prop, src)
end

function Base.:*(op::AbstractHamiltonian, t::PDVec)
    prop = OperatorMulPropagator(op, t)
    dst = similar(t, eltype(prop))
    return mul!(dst, prop, t)
end

function LinearAlgebra.dot(t::PDVec, op::AbstractHamiltonian, u::PDVec, w)
    return dot(LOStructure(op), t, op, u, w)
end
function LinearAlgebra.dot(t::PDVec, op::AbstractHamiltonian, u::PDVec)
    return dot(LOStructure(op), t, op, u)
end

function LinearAlgebra.dot(
    ::IsDiagonal, t::PDVec, op::AbstractHamiltonian, u::PDVec, w=nothing
)
    return sum(pairs(u)) do (k, v)
        conj(t[k]) * diagonal_element(op, k) * v
    end
end
function LinearAlgebra.dot(
    ::LOStructure, left::PDVec, op::AbstractHamiltonian, right::PDVec, w=nothing
)
    # First two cases: only one vector is distrubuted. Avoid shuffling things around
    # by placing that one on the left to reduce the need for communication.
    if !is_distributed(left) && is_distributed(right)
        return dot(AdjointUnknown(), left, op, right)
    elseif is_distributed(left) && !is_distributed(right)
        return conj(dot(AdjointUnknown(), right, op', left))
    end
    # Other cases: both vectors distributed or not distributed. Put the shorter vector
    # on the right as is done for regular DVecs.
    if length(left) < length(right)
        return conj(dot(AdjointUnknown(), right, op', left))
    else
        return dot(AdjointUnknown(), left, op, right)
    end
end

function LinearAlgebra.dot(
    ::AdjointUnknown, t::PDVec, op::AbstractHamiltonian, source::PDVec, w
)
    if is_distributed(t)
        target = copy_to_local!(w, t)
    else
        target = t
    end
    return _dot(target, op, source)
end
function LinearAlgebra.dot(
    ::AdjointUnknown, t::PDVec, op::AbstractHamiltonian, source::PDVec
)
    if is_distributed(t)
        w = PDWorkingMemory(t)
        target = copy_to_local!(w, t)
    else
        target = t
    end
    return _dot(target, op, source)
end

function _dot(target, op, source)
    T = promote_type(valtype(target), valtype(source), eltype(op))
    return sum(pairs(source); init=zero(T)) do (k, v)
        diag = conj(target[k]) * diagonal_element(op, k) * v
        offdiag = sum(offdiagonals(op, k); init=zero(T)) do (k_off, v_off)
            conj(target[k_off]) * v_off * v
        end
        diag + offdiag
    end
end

function LinearAlgebra.dot(t::PDVec, ops::Tuple, source::PDVec, w)
    if is_distributed(t) && any(LOStructure(op) ≢ IsDiagonal() for op in ops)
        target = copy_to_local!(w, t)
    else
        target = t
    end
    return map(ops) do op
        _dot(target, op, source)
    end
end
