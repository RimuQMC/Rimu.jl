"""
    DVec(args...; kwargs...) <: AbstractDVec{K,V}

Dictionary-based vector-like data structure for storing `key <: K` and `value <: V` pairs
for use with [`ProjectorMonteCarloProblem`](@ref) and [`ExactDiagonalizationProblem`](@ref).
`DVec` supports the interface from VectorInterface.jl but stores only non-zero `value`s.

`DVec` is fast but does not support [`Initiator`](@ref)s (see [`InitiatorDVec`](@ref)) or
parallel and distributed vector operations (see [`PDVec`](@ref)).

See also: [`AbstractDVec`](@ref), [`InitiatorDVec`](@ref), [`PDVec`](@ref).

## Constructors

* `DVec(dict::AbstractDict[; style, capacity])`: create a `DVec` with `dict` for storage.
  Note that the data may or may not be copied.

* `DVec(args...[; style, capacity])`: `args...` can be `key => value` `Pair`s or an
  iterator over `Pair`s or `Tuple`s.

* `DVec{K,V}([; style, capacity])`: create an empty `DVec{K,V}`.

* `DVec(dv::AbstractDVec[; style, capacity])`: create a `DVec` with the same contents as
   `adv`. The `style` is inherited from `dv` by default.

## Keyword arguments
* `style::StochasticStyle` determines the mode of stochastic operations. See
    [`StochasticStyle`](@ref). The default `style` is selected based on the `DVec`'s
    `valtype` (see [`default_style`](@ref)). If a style is given and the `valtype` does not
    match the `style`'s `eltype`, the values are converted to an appropriate type.
* `capacity`: The capacity argument is optional and sets the initial size of the `DVec` via
    `Base.sizehint!`.

## Examples

```jldoctest
julia> dv = DVec(:a => 1)
DVec{Symbol,Int64} with 1 entry, style = IsStochasticInteger{Int64}()
  :a => 1

julia> dv = DVec(:a => 2, :b => 3; style=IsDeterministic())
DVec{Symbol,Float64} with 2 entries, style = IsDeterministic{Float64}()
  :a => 2.0
  :b => 3.0
```
"""
struct DVec{K,V,S<:StochasticStyle,D<:AbstractDict{K,V}} <: AbstractDVec{K,V}
    storage::D
    style::S
end

###
### Constructors
###
DVec(vararg::Vararg{Pair}; kwargs...) = DVec(vararg; kwargs...)
function DVec(iterator; kwargs...)
    return abstractdvec_from_iterator(DVec, iterator; kwargs...)
end

# from Dict; check style compatibility
function DVec(
    dict::Dict{K,V};
    style::StochasticStyle=default_style(V),
    capacity=0
) where {K,V}
    if !(V == scalartype(style) || eltype(V) == scalartype(style))
        throw(ArgumentError("Style $(style) is incompatible with value type $(V). "*
                            "Use a style with scalartype(style) == $(eltype(V)) instead."))
    end
    return DVec{K,V,typeof(style),typeof(dict)}(dict, style)
end
# Empty constructor.
function DVec{K,V}(; style::StochasticStyle=default_style(V), capacity=0) where {K,V}
    return DVec(Dict{K,V}(); style, capacity)
end
# From another DVec
function DVec(dv::AbstractDVec{K,V}; style=StochasticStyle(dv), capacity=0) where {K,V}
    dvec = DVec{K,V}(; style, capacity=max(capacity, length(dv)))
    return copyto!(dvec, dv)
end

function Base.empty(dvec::DVec{K,V}; style=dvec.style) where {K,V}   # was wrongly commented out
    return DVec{K,V}(; style)
end
function Base.empty(dvec::DVec{K,V}, ::Type{V}; style=dvec.style) where {K,V}
    return empty(dvec; style)
end
function Base.empty(dvec::DVec{K}, ::Type{V}; style=dvec.style(V)) where {K,V}
    return DVec{K,V}(; style)
end
function Base.empty(dvec::DVec, ::Type{K}, ::Type{V}; style=dvec.style(V)) where {K,V}
    return DVec{K,V}(; style)
end
###
### Show
###
function Base.summary(io::IO, dvec::DVec{K,V,S}) where {K,V,S}
    len = length(dvec)
    entries = length(dvec) == 1 ? "entry" : "entries"
    print(io, "DVec{$K,$V} with $len $entries, style = $(dvec.style)")
end

###
### Interface
###
StochasticStyle(dv::DVec) = dv.style
storage(dv::DVec) = dv.storage

function Base.getindex(dvec::DVec{<:Any,V}, add) where V
    return get(dvec.storage, add, zero(V))
end
function Base.setindex!(dvec::DVec, v, k)
    if iszero(v)
        delete!(dvec, k)
    else
        dvec.storage[k] = convert(valtype(dvec), v)
    end
    return v
end

Base.pairs(dvec::DVec) = dvec.storage

function VectorInterface.scale!(dvec::DVec, α::Number)
    if iszero(α)
        zerovector!(dvec)
    else
        scale!(dvec.storage.vals, α)
    end
    return dvec
end

import Base:
    get, get!, haskey, getkey, pop!, isempty, length, values, keys, delete!, empty!, sizehint!
@delegate DVec.storage [get, get!, haskey, getkey, pop!, isempty, length, values, keys]
@delegate_return_parent DVec.storage [delete!, empty!, sizehint!]

# simd sum for Dict
function Base.sum(f::F, dvec::DVec{<:Any,V,<:Any,<:Dict}) where {F,V}
    if isempty(dvec)
        return f(zero(V))
    else
        vals = dvec.storage.vals
        dict = dvec.storage
        result = f(vals[1] * Base.isslotfilled(dict, 1))
        @inbounds @simd for i in 2:length(vals)
            result += f(vals[i] * Base.isslotfilled(dict, i))
        end
        return result
    end
end
