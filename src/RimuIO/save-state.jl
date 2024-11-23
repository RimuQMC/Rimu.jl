using Rimu, Arrow, Tables, MPI, KrylovKit

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

function save_state(args...; kwargs...)
    comm = MPI.COMM_WORLD
    if MPI.Comm_size(comm) > 1
        save_state_mpi(args...; kwargs...)
    else
        save_state_serial(args...; kwargs...)
    end
end

function save_state_serial(filename, vector; io=devnull, kwargs...)
    metadata = [string(k) => string(v) for (k, v) in kwargs]
    print(io, "saving vector...")
    time = @elapsed Arrow.write(filename, Tables.table(vector); compress=:zstd, metadata)
    println(io, "done in $(round(time, sigdigits=3)) s")
end

using MPI

function save_state_mpi(filename, vector; io=stderr, kwargs...)
    comm = MPI.COMM_WORLD

    # First rank creates the file and saves metadata.
    total_time = @elapsed begin
        if MPI.Comm_rank(comm) == 0
            println(io, "saving vector...")
            metadata = [string(k) => string(v) for (k, v) in kwargs]
            time = @elapsed begin
                Arrow.write(
                    filename, Tables.table(vector);
                    compress=:zstd, metadata, file=false
                )
            end
            println(io, "    rank 0: $(round(time, sigdigits=3)) s")
        end
        # Other ranks save their chunks in order.
        for rank in 1:(MPI.Comm_size(comm) - 1)
            MPI.Barrier(comm)
            if MPI.Comm_rank(comm) == rank
                time = @elapsed Arrow.append(filename, Tables.table(vector))
                println(io, "    rank $rank: $(round(time, sigdigits=3)) s")
            end
        end
    end
    if io ≠ devnull
        MPI.Barrier(comm)
    end
    if MPI.Comm_rank(comm) == 0
        println(io, "done in $(round(total_time, sigdigits=3)) s")
    end
end
