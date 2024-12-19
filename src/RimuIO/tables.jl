"""
    struct DVecAsTable

Wrapper over the storage of a `DVec` that allows us to treat a `DVec` as a
table from Tables.jl. Constructed with `Tables.table(::DVec)`.
"""
struct DVecAsTable{K,V}
    dict::Dict{K,V}
end
function Base.iterate(tbl::DVecAsTable, st=0)
    itr = iterate(tbl.dict, st)
    if !isnothing(itr)
        pair, st = itr
        return (; key=pair[1], value=pair[2]), st
    else
        return nothing
    end
end

Base.length(tbl::DVecAsTable) = length(tbl.dict)

function Base.show(io::IO, tbl::DVecAsTable{K,V}) where {K,V}
    print(io, length(tbl), "-row DVecAsTable{$K,$V}")
end

Tables.table(dvec::DVec) = DVecAsTable(dvec.storage)
Tables.istable(::Type{<:DVecAsTable}) = true
Tables.rowaccess(::Type{<:DVecAsTable}) = true
Tables.schema(tbl::DVecAsTable{K,V}) where {K,V} = Tables.Schema((:key, :value), (K, V))
Tables.rows(tbl::DVecAsTable) = tbl

"""
    struct PDVecAsTable

Wrapper over the storage of a `PDVec` that allows us to treat a `PDVec` as a
table from Tables.jl. Constructed with `Tables.table(::PDVec)`.
"""
struct PDVecAsTable{K,V,N}
    segments::NTuple{N,Dict{K,V}}
end
function Base.iterate(tbl::PDVecAsTable, (st,i)=(0, 1))
    if i > length(tbl.segments)
        return nothing
    end

    itr = iterate(tbl.segments[i], st)
    if !isnothing(itr)
        pair, st = itr
        return (; key=pair[1], value=pair[2]), (st, i)
    else
        return iterate(tbl, (0, i+1))
    end
end

Base.length(tbl::PDVecAsTable) = sum(length, tbl.segments)

function Base.show(io::IO, tbl::PDVecAsTable{K,V,N}) where {K,V,N}
    print(io, length(tbl), "-row PDVecAsTable{$K,$V,$N}")
end

Tables.table(pdvec::PDVec) = PDVecAsTable(pdvec.segments)
Tables.istable(::Type{<:PDVecAsTable}) = true
Tables.rowaccess(::Type{<:PDVecAsTable}) = true
Tables.rows(tbl::PDVecAsTable) = tbl
Tables.schema(tbl::PDVecAsTable{K,V}) where {K,V} = Tables.Schema((:key, :value), (K, V))
Tables.partitions(tbl::PDVecAsTable) = map(DVecAsTable, tbl.segments)
