const SubVector{T} = SubArray{T,1,Vector{T},Tuple{UnitRange{Int64}},true}
"""
    UniformSplit(array::Vector, min_chunk_size, max_chunks)

Split the array into at most `max_chunks` subarrays, with each at least `min_chunk_size` long.
If the number of elements in `array` is not divisible by the determined chunk size, the leftover elements are placed in the last chunk.

```jldoctest
julia> using Rimu.ExactDiagonalization: UniformSplit

julia> UniformSplit(collect(1:10), 3, 3)
3-element UniformSplit{Int64}:
 [1, 2, 3]
 [4, 5, 6]
 [7, 8, 9, 10]

julia> UniformSplit(collect(1:13), 5, 3)
2-element UniformSplit{Int64}:
 [1, 2, 3, 4, 5, 6]
 [7, 8, 9, 10, 11, 12, 13]
```
"""
struct UniformSplit{T} <: AbstractVector{SubVector{T}}
    array::Vector{T}
    n_chunks::Int
    chunk_size::Int

    function UniformSplit(array::Vector{T}, min_chunk_size, max_chunks) where {T}
        chunk_size = fld(length(array), max_chunks)
        if min_chunk_size > chunk_size
            n_chunks = max(fld(length(array), min_chunk_size), 1)
            chunk_size = fld(length(array), n_chunks)
        else
            n_chunks = max(fld(length(array), chunk_size), 1)
        end
        return new{T}(array, n_chunks, chunk_size)
    end
end

Base.size(split::UniformSplit) = (split.n_chunks,)

function Base.getindex(split::UniformSplit, i)
    index_start = (i - 1) * split.chunk_size + 1
    if i == split.n_chunks
        index_end = length(split.array)
    elseif 0 < i < split.n_chunks
        index_end = i * split.chunk_size
    else
        throw(BoundsError(split, i))
    end
    return view(split.array, index_start:index_end)
end

"""
    bb! = BasisBuilder{A::Type}(; col_hint=0)
    bb!(operator::AbstractOperator, pairs, seen)

Functor used with [`basis_breadth_first_search`](@ref) to build a basis of addresses of type
`A` from an operator. It contains a set of addresses (`bb!.frontier`) that is collected from
the `offdiagonals` of `operator` with all addresses contained the list of address-value
pairs `pairs` that are not element of `seen`.
"""
struct BasisBuilder{A}
    frontier::Set{A}

    function BasisBuilder{A}(; col_hint=0) where {A}
        return new{A}(sizehint!(Set{A}(), col_hint))
    end
end
init_accumulator(::Type{<:BasisBuilder}) = nothing
update_accumulator!(_, ::BasisBuilder, _) = nothing
function finalize_accumulator!(::Nothing, basis, sort; kwargs...)
    if sort
        return sort!(basis; kwargs...)
    else
        return basis
    end
end

function (bb::BasisBuilder)(operator, ks, seen)
    empty!(bb.frontier)
    for (k1, _) in ks
        for (k2, val) in offdiagonals(operator, k1)
            if val ≠ 0 && k2 ∉ seen
                push!(bb.frontier, k2)
            end
        end
    end
end

"""
    mb! = MatrixBuilder{A::Type}(; col_hint=0)
    mb!(operator::AbstractOperator, pairs, seen)

Functor used with [`basis_breadth_first_search`](@ref) to build a matrix from an
operator. It contains a set of addresses (`mb!.frontier`) that is collected from the
`offdiagonals` of `operator` with all addresses contained the list of address-value pairs
`pairs` that are not element of `seen`.

It also collects `mb!.is`, `mb!.js`, and `mb!.vs` which are used to build the sparse matrix
via `sparse!`.
"""
struct MatrixBuilder{A,T,I}
    js::Vector{I}
    is::Vector{A}
    vs::Vector{T}
    column_buffer::Dict{A,T}
    frontier::Set{A}

    function MatrixBuilder{A,T,I}(; col_hint=0) where {A,T,I}
        return new{A,T,I}(
            sizehint!(I[], col_hint),
            sizehint!(A[], col_hint),
            sizehint!(T[], col_hint),
            sizehint!(Dict{A,T}(), col_hint),
            sizehint!(Set{A}(), col_hint),
        )
    end
end

function init_accumulator(::Type{<:MatrixBuilder{<:Any,T,I}}) where {I,T}
    return MatrixBuilderAccumulator{T,I}()
end

function (builder::MatrixBuilder{<:Any,T})(operator, columns, seen) where {T}
    empty!(builder.is)
    empty!(builder.js)
    empty!(builder.vs)
    empty!(builder.frontier)
    for (col_key, col_index) in columns
        empty!(builder.column_buffer)
        diag = diagonal_element(operator, col_key)
        if !iszero(diag)
            builder.column_buffer[col_key] = diag
        end
        for (row_key, value) in offdiagonals(operator, col_key)
            if !iszero(value)
                new_value = get(builder.column_buffer, row_key, zero(T)) + value
                builder.column_buffer[row_key] = new_value
                row_key ∉ seen && push!(builder.frontier, row_key)
            end
        end

        old_len = length(builder.is)
        new_len = length(builder.is) + length(builder.column_buffer)
        resize!(builder.is, new_len)
        resize!(builder.js, new_len)
        resize!(builder.vs, new_len)
        @inbounds for (i, (row_key, value)) in enumerate(builder.column_buffer)
            builder.is[old_len + i] = row_key
            builder.js[old_len + i] = col_index
            builder.vs[old_len + i] = value
        end
    end
end

"""
    struct MatrixBuilderAccumulator

Used in conjunction with [`basis_breadth_first_search`](@ref) and
[`MatrixBuilder`](@ref). It is used to combine the sparse matrix as it is being built by
multiple threads.
"""
struct MatrixBuilderAccumulator{T,I}
    is::Vector{I}
    js::Vector{I}
    vs::Vector{T}

    MatrixBuilderAccumulator{T,I}() where {T,I} = new{T,I}(I[], I[], T[])
end

function update_accumulator!(acc::MatrixBuilderAccumulator, mb::MatrixBuilder, mapping)
    for i in eachindex(mb.is)
        if haskey(mapping, mb.is[i]) # skip keys that were filtered
            push!(acc.is, mapping[mb.is[i]])
            push!(acc.js, mb.js[i])
            push!(acc.vs, mb.vs[i])
        end
    end
end

function finalize_accumulator!(
    acc::MatrixBuilderAccumulator{T,I}, basis, sort; kwargs...
) where {T,I}
    n = length(basis)

    # see docstring of `sparse!` for an explanation for what this is
    klasttouch = Vector{I}(undef, n)
    csrrowptr = Vector{I}(undef, n + 1)
    csrcolval = Vector{I}(undef, length(acc.is))
    csrnzval = Vector{T}(undef, length(acc.is))

    matrix = SparseArrays.sparse!(
        acc.is, acc.js, acc.vs, n, n, +,
        klasttouch, csrrowptr, csrcolval, csrnzval,
        acc.is, acc.js, acc.vs,
    )
    if sort
        # `csrcolval` is no longer needed, so we can reuse it
        perm = resize!(csrcolval, length(basis))
        sortperm!(perm, basis; kwargs...)
        permute!(matrix, perm, perm), permute!(basis, perm)
    else
        return matrix, basis
    end
end

"""
    basis_breadth_first_search(::Type{Builder}, operator, starting_basis)

Internal function that performs breadth-first search (BFS) on an operator.

`Builder` is either [`MatrixBuilder`](@ref) or [`BasisBuilder`](@ref), which triggers
building a matrix and basis, or only the basis of addresses, respectively.
"""
function basis_breadth_first_search(
    ::Type{Builder}, operator, basis::Vector{A};
    min_batch_size=100,
    max_tasks=4 * Threads.nthreads(),

    max_depth=Inf,
    minimum_size=Inf,

    cutoff=nothing,
    filter=isnothing(cutoff) ? Returns(true) : a -> diagonal_element(operator, a) ≤ cutoff,

    sizelim=Inf,
    nnzs=0,
    col_hint=0,

    sort=false,
    by=identity,
    rev=false,
    lt=isless,
    order=Base.Forward,
) where {Builder,A}

    dim = dimension(operator, basis[1])
    if dim > sizelim
        throw(ArgumentError("dimension $dim larger than sizelim $sizelim"))
    end
    sizehint!(basis, nnzs)
    sort_kwargs = (; by, rev, lt, order)

    # addresses already visited and addresses visited in the last layer
    seen = Set(basis)
    last_seen = empty(seen)

    # addresses to visit and their index
    curr_frontier = collect(zip(basis, eachindex(basis)))
    next_frontier = empty(curr_frontier)

    # map from address to index
    mapping = Dict(zip(basis, eachindex(basis)))

    # builders store task-local storage and choose whether a matrix is to be built or not
    builders = [Builder(; col_hint) for _ in 1:max_tasks]
    result_accumulator = init_accumulator(Builder)

    depth = 0
    early_stop = false
    while !isempty(curr_frontier)
        depth += 1
        early_stop = length(basis) > minimum_size || depth > max_depth

        # We can stop here when not constructing the matrix. If we are, we still need
        # to do another round to evaluate the columns.
        if Builder <: BasisBuilder && early_stop
            break
        end

        # Split the workload into chunks and spawn a task for each. These now run
        # asynchronously while the main thread continues.
        split_frontier = UniformSplit(curr_frontier, min_batch_size, max_tasks)
        tasks = map(enumerate(split_frontier)) do (i, sub_frontier)
            Threads.@spawn builders[$i]($operator, $sub_frontier, seen)
        end

        # Procesess the tasks in order they were spawned. This is fine as we've used more
        # tasks than threads.
        for (task, builder) in zip(tasks, builders)
            wait(task)

            result = builder.frontier
            for k in result
                if k ∉ last_seen
                    push!(last_seen, k)
                    if !early_stop && (isnothing(filter) || filter(k))
                        push!(basis, k)
                        push!(next_frontier, (k, length(basis)))
                        mapping[k] = length(basis)
                    end
                end
            end

            # update matrix if needed
            update_accumulator!(result_accumulator, builder, mapping)
        end

        # This had to wait until now so we don't get into data races
        union!(seen, last_seen)
        curr_frontier, next_frontier = next_frontier, curr_frontier
        empty!(next_frontier)
        empty!(last_seen)
    end

    return finalize_accumulator!(result_accumulator, basis, sort; sort_kwargs...)
end

function _address_to_basis(operator, addr_or_basis)
    if addr_or_basis isa AbstractVector || addr_or_basis isa Tuple
        check_address_type(operator, eltype(addr_or_basis))
        return collect(addr_or_basis)
    else
        check_address_type(operator, typeof(addr_or_basis))
        return [addr_or_basis]
    end
end

"""
    build_basis(
        ham, address=starting_address(ham);
        cutoff, filter, sizelim, sort=false, kwargs...
    ) -> basis
    build_basis(ham, addresses::AbstractVector; kwargs...)

Get all basis element of a linear operator `ham` that are reachable (via
non-zero matrix elements) from the address `address`, returned as a vector.
Instead of a single address, a vector of `addresses` can be passed.
Does not return the matrix, for that purpose use [`BasisSetRepresentation`](@ref).

Providing an energy cutoff will skip addresses with diagonal elements greater
than `cutoff`. Alternatively, an arbitrary `filter` function can be used instead.
Addresses passed as arguments are not filtered.

Providing a `max_depth` will limit the size of the basis by only visiting addresses that are
connected to the `starting_address` through `max_depth` hops through the
Hamiltonian. Similarly, providing `minimum_size` will stop the bulding process after the
basis reaches a length of at least `minimum_size`.

A maximum basis size `sizelim` can be set which will throw an error if the expected
dimension of `ham` is larger than `sizelim`. This may be useful when memory may be a
concern. These options are disabled by default.

!!! warning
        The order the basis is returned in is arbitrary and non-deterministic. Use
        `sort=true` if the ordering matters.

"""
function build_basis(operator, addr=starting_address(operator); sizelim=Inf, kwargs...)
    basis = _address_to_basis(operator, addr)
    return basis_breadth_first_search(BasisBuilder{eltype(basis)}, operator, basis; sizelim, kwargs...)
end

"""
    build_sparse_matrix_from_LO(
        ham, address=starting_address(ham);
        cutoff, filter=nothing, nnzs, col_hint, sort=false, kwargs...
    ) -> sparse_matrix, basis
    build_sparse_matrix_from_LO(ham, addresses::AbstractVector; kwargs...)

Create a sparse matrix `sparse_matrix` of all reachable matrix elements of a linear operator
`ham` starting from `address`. Instead of a single address, a vector of `addresses` can be
passed.  The vector `basis` contains the addresses of basis configurations.

Providing the number `nnzs` of expected calculated matrix elements and `col_hint` for the
estimated number of nonzero off-diagonal matrix elements in each matrix column may improve
performance.

Providing an energy cutoff will skip the columns and rows with diagonal elements greater
than `cutoff`. Alternatively, an arbitrary `filter` function can be used instead. These are
not enabled by default. To generate the matrix truncated to the subspace spanned by the
`addresses`, use `filter = Returns(false)`.

Providing a `max_depth` will limit the size of the matrix by only visiting addresses that
are connected to the `starting_address` through `max_depth` hops through the
Hamiltonian. Similarly, providing `minimum_size` will stop the bulding process after the
basis reaches a length of at least `minimum_size`.

Setting `sort` to `true` will sort the `basis` and order the matrix rows and columns
accordingly. This is useful when the order of the columns matters, e.g. when comparing
matrices. Any additional keyword arguments are passed on to `Base.sortperm`.

!!! warning
        The order of the returned rows and columns is arbitrary and non-deterministic. Use
        `sort=true` if the ordering matters.

See [`BasisSetRepresentation`](@ref).
"""
function build_sparse_matrix_from_LO(
    operator, addr=starting_address(operator); sizelim=1e7, kwargs...
)
    basis = _address_to_basis(operator, addr)
    T = eltype(operator)
    return basis_breadth_first_search(
        MatrixBuilder{eltype(basis),T,Int32}, operator, basis;
        sizelim, kwargs...
    )
end
