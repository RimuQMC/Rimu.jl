"""
Module to provide file input and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save_df(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load_df(filename)`](@ref) Load Arrow file into dataframe.
* [`RimuIO.save_state(filename, vector; metadata...)`](@ref) Save a vector and
  optinal metadata in Arrow format.
* [`RimuIO.load_state(filename)`](@ref) Load a file created by `save_state`.
"""
module RimuIO

using DataFrames: DataFrames, DataFrame, metadata!
using StaticArrays: StaticArrays, SVector

using Rimu: mpi_size, mpi_rank, mpi_barrier
using Rimu.BitStringAddresses: BitStringAddresses, BitString, BoseFS,
    CompositeFS, FermiFS, SortedParticleList,
    num_modes, num_particles
using Rimu.DictVectors: PDVec, DVec, target_segment
using Rimu.Interfaces: Interfaces, localpart, storage
using Rimu.StochasticStyles: default_style, IsDynamicSemistochastic

import Rimu, Tables, Arrow, Arrow.ArrowTypes

export save_df, load_df, save_state, load_state

include("tables.jl")
include("arrowtypes.jl")

"""
    save_df(filename, df::DataFrame; kwargs...)

Save dataframe in Arrow format.

Keyword arguments are passed on to
[`Arrow.write`](https://arrow.apache.org/julia/dev/reference/#Arrow.write). Compression is
enabled by default for large `DataFrame`s (over 10,000 rows).

Table-level metadata of the `DataFrame` is saved as Arrow metadata (with `String` value)
unless overwritten with the keyword argument `metadata`.

See also [`RimuIO.load_df`](@ref).
"""
function save_df(
    filename, df::DataFrame;
    compress = size(df)[1]>10_000 ? :zstd : nothing,
    metadata = nothing,
    kwargs...
)
    if metadata === nothing
        metadata = [key => string(val) for (key, val) in DataFrames.metadata(df)]
    end
    push!(metadata, "RIMU_PACKAGE_VERSION" => string(Rimu.PACKAGE_VERSION))
    Arrow.write(filename, df; compress, metadata, kwargs...)
end

"""
    load_df(filename; propagate_metadata = true, add_filename = true) -> DataFrame

Load Arrow file into `DataFrame`. Optionally propagate metadata to `DataFrame` and
add the file name as metadata.

See also [`RimuIO.save_df`](@ref).
"""
function load_df(filename; propagate_metadata = true, add_filename = true)
    table = Arrow.Table(filename)
    df = DataFrame(table)
    if propagate_metadata
        meta_data = Arrow.getmetadata(table)
        isnothing(meta_data) || for (key, val) in meta_data
            metadata!(df, key, val)
        end
    end
    add_filename && metadata!(df, "filename", filename) # add filename as metadata
    return df
end

"""
    save_state(filename, vector; io, kwargs...)

Save [`PDVec`](@ref) or [`DVec`](@ref) `vector` to an arrow file `filename`.

`io` determines the output stream to write progress to. Defaults to `stderr` when MPI is
enabled and `devnull` otherwise.

All other `kwargs` are saved as strings to the arrow file and will be parsed back when the
state is loaded.

See also [`load_state`](@ref).
"""
function save_state(filename, vector; kwargs...)
    new_kwargs = (; RIMU_PACKAGE_VERSION=Rimu.PACKAGE_VERSION, kwargs...)
    if mpi_size() > 1
        _save_state_mpi(filename, vector; new_kwargs...)
    else
        _save_state_serial(filename, vector; new_kwargs...)
    end
end

function _save_state_serial(filename, vector; io=devnull, kwargs...)
    metadata = [string(k) => string(v) for (k, v) in kwargs]
    print(io, "saving vector...")
    time = @elapsed Arrow.write(filename, Tables.table(vector); compress=:zstd, metadata)
    println(io, " done in $(round(time, sigdigits=3)) s")
end

function _save_state_mpi(filename, vector; io=stderr, kwargs...)
    # First rank creates the file and saves metadata.
    total_time = @elapsed begin
        if mpi_rank() == 0
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
        # Other ranks append their data to the file in order.
        for rank in 1:(mpi_size() - 1)
            mpi_barrier()
            if mpi_rank() == rank
                time = @elapsed Arrow.append(filename, Tables.table(vector))
                println(io, "    rank $rank: $(round(time, sigdigits=3)) s")
            end
        end
    end
    mpi_barrier()
    if mpi_rank() == 0
        println(io, "done in $(round(total_time, sigdigits=3)) s")
    end
end

"""
    load_state(filename; kwargs...) -> PDVec, NamedTuple
    load_state(PDVec, filename; kwargs...) -> PDVec, NamedTuple
    load_state(DVec, filename; kwargs...) -> DVec, NamedTuple

Load the state saved in the Arrow file `filename`. `kwargs` are passed to the constructor of
[`PDVec`](@ref)/[`DVec`](@ref). Any metadata stored in the file is be parsed as a number (if
possible) and returned alongside the vector in a `NamedTuple`.

See also [`save_state`](@ref).
"""
function load_state(::Type{D}, filename; style=nothing, kwargs...) where {D}
    tbl = Arrow.Table(filename)
    if Tables.schema(tbl).names ≠ (:key, :value)
        throw(ArgumentError("`$filename` is not a valid Rimu state file"))
    end
    K = eltype(tbl.key)
    V = eltype(tbl.value)
    if isnothing(style)
        if V <: AbstractFloat
            style = IsDynamicSemistochastic()
        else
            style = default_style(V)
        end
    end
    vector = D{K,V}(; style, kwargs...)
    copyto!(vector, tbl.key, tbl.value)

    arrow_meta = Arrow.metadata(tbl)[]
    if !isnothing(arrow_meta)
        metadata_pairs = map(collect(arrow_meta)) do (k, v)
            k == "RIMU_PACKAGE_VERSION" && return Symbol(k) => VersionNumber(v)
            v_int = tryparse(Int, v)
            !isnothing(v_int) && return Symbol(k) => v_int
            v_float = tryparse(Float64, v)
            !isnothing(v_float) && return Symbol(k) => v_float
            v_cmp = tryparse(ComplexF64, v)
            !isnothing(v_cmp) && return Symbol(k) => v_cmp
            v_bool = tryparse(Bool, v)
            !isnothing(v_bool) && return Symbol(k) => v_bool
            Symbol(k) => v
        end
        metadata = NamedTuple(metadata_pairs)
    else
        metadata = (;)
    end

    return vector, metadata
end

function load_state(filename; kwargs...)
    if Threads.nthreads() == 1 && mpi_size() == 1
        return load_state(DVec, filename; kwargs...)
    else
        return load_state(PDVec, filename; kwargs...)
    end
end

end # module RimuIO
