"""
    AbstractDVec{K,V}

Abstract data type for vector-like data structures with sparse storage. While conceptually
`AbstractDVec`s represent elements of a vector space over a scalar type `V`, they are
indexed by an arbitrary type `K` (could be non-integers) similar to dictionaries. They
support the interface from [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)
and are designed to work well for quantum Monte Carlo with [`lomc!`](@ref Main.lomc!) and
for matrix-free linear algebra with [KrylovKit](https://github.com/Jutho/KrylovKit.jl).

Concrete implementations are available as [`DVec`](@ref Main.DictVectors.DVec) and
[`InitiatorDVec`](@ref Main.DictVectors.InitiatorDVec).

`AbstractDVec`s have a [`StochasticStyle`](@ref) which selects the spawning algorithm in
`FCIQMC`. Looking up an element that is not stored in the `AbstractDVec` should return a
zero, and setting a value to zero should remove it from the vector. To iterate over an
`AbstractDVec`, use `keys`, `pairs`, or `values`. When possible, use reduction functions
such as `sum` or `mapreduce`.

# Interface

The interface is similar to the `AbstractDict` interface, but with the changed behaviour as
noted above.  Implement what would be needed for the `AbstractDict` interface (`pairs`,
`keys`, `values`, `setindex!`, `getindex`, `delete!`, `length`, `empty`, `empty!`) and, in
addition:

* [`StochasticStyle`](@ref)
* [`storage`](@ref) returns an `AbstractDict` storing the raw data with possibly
  different `valtype` than `V`.
* [`deposit!`](@ref)

A default implementation for the
[VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl) interface is provided
through the above functions.

See also [`DictVectors`](@ref Main.DictVectors), [`Interfaces`](@ref).
"""
abstract type AbstractDVec{K,V} end

"""
    deposit!(w::AbstractDVec, add, val, parent::Pair)

Add `val` into `w` at address `add`, taking into account initiator rules if applicable.
`parent` contains the `address => value` pair from which the pair `add => val`
was created. [`InitiatorDVec`](@ref Main.DictVectors.InitiatorDVec) can intercept this and add its own functionality.
"""
function deposit!(w, add, val, _)
    w[add] += convert(valtype(w), val)
end

@deprecate zero! zerovector!

"""
    localpart(dv) -> AbstractDVec

Get the part of `dv` that is located on this MPI rank. Returns `dv` itself for
`AbstractDVec`s.
"""
localpart(dv) = dv # default for local data

"""
    storage(dvec) -> AbstractDict

Return the raw storage associated with `dvec` as an `AbstractDict`. Used in MPI
communication.
"""
storage(v::AbstractVector) = v

StochasticStyle(::AbstractArray{T}) where {T} = default_style(T)

"""
    freeze(dv)

Create a "frozen" version of `dv` which can no longer be modified or used in the
conventional manner, but supports faster dot products.

If `dv` is an [`MPIData`](@ref Main.Rimu.RMPI.MPIData), synchronize its contents among the ranks first.
"""
freeze(v::AbstractVector) = copy(v)

"""
    working_memory(dv::AbstractDVec)

Create a working memory instance compatible with `dv`. The working memory must be
compatible with [`sort_into_targets!`](@ref) and [`apply_operator!`](@ref).
"""
working_memory(dv) = similar(localpart(dv))

"""
    apply_operator!(working_memory, target, source, operator, boost=1, compress=Val(true)) ->
        stat_names, stats, working_memory, target

Perform a single matrix(/operator)-vector multiplication:

```math
v^{(n + 1)} = \\hat{T} v^{(n)} ,
```

where ``\\hat{T}`` is the `operator`, ``v^{(n+1)}`` is the `target` and ``v^{(n)}`` is the
`source`. The `working_memory` can be used as temporary storage.

The `boost` argument is passed to [apply_column!](@ref) and increases the number of spawns
performed. For the operator to be applied without compressing the vector after, set
`compress` to `Val(false)`.

Whether the operation is performed in a stochastic, semistochastic, or determistic way is
controlled by the trait `StochasticStyle(target)`. See [`StochasticStyle`](@ref).

Returns the step stats generated by the `StochasticStyle`, the working memory and the
`target` vector. `target` and `working_memory` may be mutated and/or swapped.
"""
function apply_operator!(
    working_memory, target, source, ham, boost=1, compress::Val{C}=Val(true)
) where {C}
    v = localpart(source)
    @assert working_memory ≢ v "`working_memory` and `source` must not have the same local storage"
    @assert localpart(target) ≢ v "`target` and `source` must not have the same local storage"
    zerovector!(working_memory)

    spawn_names, spawn_stats = step_stats(v)
    for (add, val) in pairs(v)
        spawn_stats += apply_column!(working_memory, ham, add, val, boost)
    end

    # Now, working_memory holds the new values - they need to be moved into the target.
    target, working_memory, spawn_stats = sort_into_targets!(
        target, working_memory, spawn_stats
    )
    if C
        comp_names, comp_stats = compress!(target)
        names = (spawn_names..., comp_names...)
        stats = (spawn_stats..., comp_stats...)
    else
        names = spawn_names
        stats = spawn_stats
    end

    return names, stats, working_memory, target
end

"""
    sort_into_targets!(target, source, stats) -> target, source, agg_stats

Aggregate coefficients from `source` to `target` and from `stats` to `agg_stats`
according to thread- or MPI-level parallelism.

Returns the new `target` and `source`, as the sorting process may involve swapping them.
"""
sort_into_targets!(dv::T, wm::T, stats) where {T} = wm, dv, stats
