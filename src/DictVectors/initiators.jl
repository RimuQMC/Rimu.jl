"""
    InitiatorValue{V}(; safe::V, unsafe::V, initiator::V) where V
Composite "walker" with three fields. For use with [`InitiatorDVec`](@ref)s.
"""
struct InitiatorValue{V}
    safe::V
    unsafe::V
    initiator::V
end
function InitiatorValue{V}(;safe=zero(V), unsafe=zero(V), initiator=zero(V)) where {V}
    return InitiatorValue{V}(V(safe), V(unsafe), V(initiator))
end
function InitiatorValue{V}(i::InitiatorValue) where {V}
    return InitiatorValue{V}(V(i.safe), V(i.unsafe), V(i.initiator))
end

function Base.:+(v::InitiatorValue, w::InitiatorValue)
    return InitiatorValue(v.safe + w.safe, v.unsafe + w.unsafe, v.initiator + w.initiator)
end
function Base.:-(v::InitiatorValue, w::InitiatorValue)
    return InitiatorValue(v.safe - w.safe, v.unsafe - w.unsafe, v.initiator - w.initiator)
end
function Base.:-(v::InitiatorValue{V}) where {V}
    return InitiatorValue{V}(-v.safe, -v.unsafe, -v.initiator)
end
Base.zero(::Union{I,Type{I}}) where {V,I<:InitiatorValue{V}} = InitiatorValue{V}()

Base.convert(::Type{<:InitiatorValue{V}}, x::InitiatorValue{V}) where {V} = x
function Base.convert(::Type{<:InitiatorValue{U}}, x::InitiatorValue{V}) where {U,V}
    return InitiatorValue{U}(safe=x.safe, unsafe=x.unsafe, initiator=x.initiator)
end
function Base.convert(::Type{<:InitiatorValue{V}}, x) where {V}
    return InitiatorValue{V}(safe=convert(V, x))
end

"""
    AbstractInitiator

Abstract type for defining initiator rules. Concrete implementations:

* [`Initiator`](@ref)
* [`SimpleInitiator`](@ref)
* [`CoherentInitiator`](@ref)
"""
abstract type AbstractInitiator{V} end

"""
    Initiator

Initiator to be passed to [`InitiatorDVec`](@ref). Initiator rules:

* Initiators can spawn anywhere.
* Non-initiators can spawn to initiators.
"""
struct Initiator{V} <: AbstractInitiator{V}
    threshold::V
end

function value(i::Initiator, v::InitiatorValue)
    return v.safe + v.initiator + !iszero(v.initiator) * v.unsafe
end

"""
    SimpleInitiator

Simplified initiator to be passed to [`InitiatorDVec`](@ref). Initiator rules:

* Initiators can spawn anywhere.
* Non-initiators cannot spawn.
"""
struct SimpleInitiator{V} <: AbstractInitiator{V}
    threshold::V
end

function value(i::SimpleInitiator, v::InitiatorValue)
    return v.safe + v.initiator
end

"""
    CoherentInitiator

Initiator to be passed to [`InitiatorDVec`](@ref). Initiator rules:

* Initiators can spawn anywhere.
* Non-initiators can spawn to initiators.
* Multiple non-initiators can spawn to a single non-initiator if their contributions add up
  to a value greater than the initiator threshold.
"""
struct CoherentInitiator{V} <: AbstractInitiator{V}
    threshold::V
end

function value(i::CoherentInitiator, v::InitiatorValue)
    if !iszero(v.initiator) || abs(v.unsafe) > i.threshold
        return v.initiator + v.safe + v.unsafe
    else
        return v.initiator + v.safe
    end
end

"""
    InitiatorDVec

Functionally identical to [`DVec`](@ref), but contains [`InitiatorValue`]s
"""
struct InitiatorDVec{
    K,V,D<:Dict{K,InitiatorValue{V}},S<:StochasticStyle{V},I<:AbstractInitiator
} <: AbstractDVec{K,V}
    storage::D
    style::S
    initiator::I
end

###
### Constructors
###
function InitiatorDVec(
    dict::Dict{K,V}; style=default_style(V), initiator=Initiator(one(V)), capacity=0
) where {K,V}
    T = eltype(style)
    storage = Dict{K,InitiatorValue{T}}()
    sizehint!(storage, capacity)
    for (k, v) in pairs(dict)
        storage[k] = InitiatorValue{T}(safe=v)
    end
    return InitiatorDVec(storage, style, initiator)
end
function InitiatorDVec(args...; kwargs...)
    storage = Dict(args...)
    return InitiatorDVec(storage; kwargs...)
end
function InitiatorDVec(
    dv::AbstractDVec{K,V}; style=StochasticStyle(dv), initiator=Initiator(one(V)), capacity=0
) where {K,V}
    storage = Dict{K,InitiatorValue{V}}()
    sizehint!(storage, max(capacity, length(dv)))
    for (k, v) in pairs(dv)
        storage[k] = InitiatorValue{V}(safe=v)
    end
    return InitiatorDVec(storage, style, initiator)
end
function InitiatorDVec{K,V}(; kwargs...) where {K,V}
    return InitiatorDVec(Dict{K,V}(); kwargs...)
end

function Base.empty(dvec::InitiatorDVec)
    return InitiatorDVec(empty(dvec.storage), dvec.style, dvec.initiator)
end
function Base.empty(dvec::InitiatorDVec, args...)
    return InitiatorDVec(empty(dvec.storage, args...); initiator=dvec.initiator)
end
Base.similar(dvec::InitiatorDVec, args...; kwargs...) = empty(dvec, args...; kwargs...)

###
### Show
###
function Base.summary(io::IO, dvec::InitiatorDVec{K,V,S}) where {K,V,S}
    len = length(dvec)
    print(io, "InitiatorDVec{$K,$V,$S} with $len entries")
end

###
### Interface
###
StochasticStyle(dv::InitiatorDVec) = dv.style
storage(dv::InitiatorDVec) = dv.storage

function Base.getindex(dvec::InitiatorDVec{<:Any,V}, add) where {V}
    return value(dvec.initiator, get(dvec.storage, add, zero(InitiatorValue{V})))
end
function Base.setindex!(dvec::InitiatorDVec{<:Any,V}, v, k) where {V}
    if iszero(v)
        delete!(dvec.storage, k)
    else
        dvec.storage[k] = InitiatorValue{V}(safe=v)
    end
    return v
end

function Base.pairs(dvec::InitiatorDVec)
    return Iterators.map(pairs(dvec.storage)) do ((k, v))
        k => value(dvec.initiator, v)
    end
end
function Base.values(dvec::InitiatorDVec)
    return Iterators.map(values(dvec.storage)) do v
        value(dvec.initiator, v)
    end
end
function Base.get(dvec::InitiatorDVec, args...)
    return(value(dvec.initiator, get(dvec.storage, args...)))
end
function Base.get!(dvec::InitiatorDVec{<:Any,V}, key, default) where {V}
    return(value(dvec.initiator, get!(dvec.storage, key, InitiatorValue{V}(safe=default))))
end
function Base.get!(f::Function, dvec::InitiatorDVec{<:Any,V}, key) where {V}
    return(value(dvec.initiator, get!(dvec.storage, key, InitiatorValue{V}(safe=f()))))
end

@delegate InitiatorDVec.storage [haskey, getkey, pop!, isempty, length, keys]
@delegate_return_parent InitiatorDVec.storage [delete!, empty!, sizehint!]

function deposit!(w::InitiatorDVec{<:Any,V}, add, val, (p_add, p_val)) where {V}
    i = w.initiator
    if p_add == add # diagonal death
        if abs(p_val) > i.threshold
            new_val = InitiatorValue{V}(initiator=val)
        else
            new_val = InitiatorValue{V}(safe=val)
        end
    else # offdiagonal spawn
        if abs(p_val) > i.threshold
            new_val = InitiatorValue{V}(safe=val)
        else
            new_val = InitiatorValue{V}(unsafe=val)
        end
    end
    w.storage[add] = get(w.storage, add, zero(InitiatorValue{V})) + new_val
end
