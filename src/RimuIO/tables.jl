"""
    struct DVecTable

Wrapper over the storage of a `DVec` that allows us to treat a `DVec` as a
table from Tables.jl. Constructed with `Tables.table(::DVec)`.
"""
struct DVecTable{K,V}
    dict::Dict{K,V}
end
function Base.iterate(tbl::DVecTable, st=0)
    itr = iterate(tbl.dict, st)
    if !isnothing(itr)
        pair, st = itr
        return (; key=pair[1], value=pair[2]), st
    else
        return nothing
    end
end

Base.length(tbl::DVecTable) = length(tbl.dict)

function Base.show(io::IO, tbl::DVecTable{K,V}) where {K,V}
    print(io, length(tbl), "-row DVecTable{$K,$V}")
end

Tables.table(dvec::DVec) = DVecTable(dvec.storage)
Tables.istable(::Type{<:DVecTable}) = true
Tables.rowaccess(::Type{<:DVecTable}) = true
Tables.schema(tbl::DVecTable{K,V}) where {K,V} = Tables.Schema((:key, :value), (K, V))
Tables.rows(tbl::DVecTable) = tbl

"""
    struct PDVecTable

Wrapper over the storage of a `PDVec` that allows us to treat a `PDVec` as a
table from Tables.jl. Constructed with `Tables.table(::PDVec)`.
"""
struct PDVecTable{K,V,N}
    segments::NTuple{N,Dict{K,V}}
end
function Base.iterate(tbl::PDVecTable, (st,i)=(0, 1))
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

Base.length(tbl::PDVecTable) = sum(length, tbl.segments)

function Base.show(io::IO, tbl::PDVecTable{K,V,N}) where {K,V,N}
    print(io, length(tbl), "-row PDVecTable{$K,$V,$N}")
end

Tables.table(pdvec::PDVec) = PDVecTable(pdvec.segments)
Tables.istable(::Type{<:PDVecTable}) = true
Tables.rowaccess(::Type{<:PDVecTable}) = true
Tables.rows(tbl::PDVecTable) = tbl
Tables.schema(tbl::PDVecTable{K,V}) where {K,V} = Tables.Schema((:key, :value), (K, V))
Tables.partitions(tbl::PDVecTable) = map(DVecTable, tbl.segments)
