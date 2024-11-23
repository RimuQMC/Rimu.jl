"""
Module to provide file input and output functionality for `Rimu`.
Provides convenience functions:

* [`RimuIO.save_df(filename, df::DataFrame)`](@ref) Save dataframe in Arrow format.
* [`RimuIO.load_df(filename)`](@ref) Load Arrow file into dataframe.
"""
module RimuIO

using Arrow: Arrow, ArrowTypes
using BSON: BSON, bson
using DataFrames: DataFrames, DataFrame, metadata!
using StaticArrays: StaticArrays, SVector

using Rimu.BitStringAddresses: BitStringAddresses, BitString, BoseFS,
    CompositeFS, FermiFS, SortedParticleList,
    num_modes, num_particles
using Rimu.DictVectors: PDVec, DVec, target_segment
using Rimu.Interfaces: Interfaces, localpart, storage
using Rimu.StochasticStyles: default_style, IsDynamicSemistochastic


export save_df, load_df

include("arrowtypes.jl")

"""
    RimuIO.save_df(filename, df::DataFrame; kwargs...)
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
    Arrow.write(filename, df; compress, metadata, kwargs...)
end

"""
    RimuIO.load_df(filename; propagate_metadata = true, add_filename = true) -> DataFrame
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

function save_state(filename, vector; kwargs...)
    metadata = [string(k) => string(v) for (k, v) in kwargs]
    Arrow.write(
        filename, (; keys=collect(keys(vector)), values=collect(values(vector)));
        compress=:zstd, metadata,
    )
end

function load_state(filename; style=nothing, kwargs...)
    tbl = Arrow.Table(filename)
    K = eltype(tbl.key)
    V = eltype(tbl.value)
    if isnothing(style)
        if V <: AbstractFloat
            style = IsDynamicSemistochastic()
        else
            style = default_style(V)
        end
    end
    vector = PDVec{K,V}(; style, kwargs...)
    fill_vector!(vector, tbl.key, tbl.value)

    arrow_meta = Arrow.metadata(tbl)[]
    if !isnothing(arrow_meta)
        metadata = NamedTuple(Symbol(k) => eval(Meta.parse(v)) for (k, v) in arrow_meta)
    else
        metadata = (;)
    end

    return vector, metadata
end

function fill_vector!(vector::PDVec, keys, vals)
    Threads.@threads for seg_id in eachindex(vector.segments)
        seg = vector.segments[seg_id]
        sizehint!(seg, length(keys) ÷ length(vector.segments))
        for (k, v) in zip(keys, vals)
            if target_segment(vector, k) == (seg_id, true)
                seg[k] = v
            end
        end
    end
end
function fill_vector!(vector::DVec, keys, vals)
    for (k, v) in zip(keys, vals)
        vector[k] = v
    end
end

end # module RimuIO
