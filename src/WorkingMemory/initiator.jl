struct InitiatorValue{V}
    safe::V
    unsafe::V
    initiator::V
end
function InitiatorValue{V}(;safe=zero(V), unsafe=zero(V), initiator=zero(V)) where {V}
    return InitiatorValue{V}(V(safe), V(unsafe), V(initiator))
end

function Base.:+(v::InitiatorValue, w::InitiatorValue)
    return InitiatorValue(v.safe + w.safe, v.unsafe + w.unsafe, v.initiator + w.initiator)
end
Base.zero(::Type{InitiatorValue{V}}) where {V} = InitiatorValue{V}()
Base.zero(::InitiatorValue{V}) where {V} = InitiatorValue{V}()

function value(v::InitiatorValue)
    return v.safe + v.unsafe * !iszero(v.initiator) + v.initiator
end

struct InitiatorMemory{
    K,V,M<:AbstractWorkingMemory{K,InitiatorValue{V}}
} <: AbstractWorkingMemory{K,V}
    mem::M
    threshold::V
end

Base.empty!(mem::InitiatorMemory) = empty!(mem.mem)
Base.pairs(mem::InitiatorMemory) = Iterators.map(p -> p[1] => value(p[2]), pairs(mem.mem))
DictVectors.capacity(mem::InitiatorMemory) = capacity(mem.mem)
DictVectors.StochasticStyle(mem::InitiatorMemory) = StochasticStyle(mem.mem)

function spawn!(mem::InitiatorMemory, k, v, (pk, pv))
    V = valtype(mem)
    is_initiator = abs(pv) > mem.threshold
    if k == pk # diagonal spawn
        if is_initiator
            val = InitiatorValue{V}(initiator=v)
        else
            val = InitiatorValue{V}(safe=v)
        end
    else # offdiagonal spawn
        if is_initiator
            val = InitiatorValue{V}(safe=v)
        else
            # spawns from non-initiators might be rejected later
            val = InitiatorValue{V}(unsafe=v)
        end
    end
    spawn!(mem.mem, k, val, (pk, pv))
end
