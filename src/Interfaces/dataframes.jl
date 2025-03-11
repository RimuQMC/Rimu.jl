function num_replicas(df::DataFrame)
    if haskey(metadata(df), "num_replicas")
        return parse(Int, metadata(df, "num_replicas"))
    else
        num = length(filter(startswith("norm"), names(df)))
        num > 0 || throw(ArgumentError("No replicas found in DataFrame"))
        return num
    end
end

function num_spectral_states(df::DataFrame)
    if haskey(metadata(df), "num_spectral_states")
        return parse(Int, metadata(df, "num_spectral_states"))
    else
        if length(filter(startswith("norm"), names(df))) == 0
            throw(ArgumentError("No spectral states found in DataFrame"))
        end
        return 1
    end
end

function num_overlaps(df::DataFrame)
    if haskey(metadata(df), "num_overlaps")
        return parse(Int, metadata(df, "num_overlaps"))
    else
        if length(filter(startswith("norm"), names(df))) == 0
            throw(ArgumentError("No replicas found in DataFrame"))
        end
        return length(filter(startswith(r"c[0-9]+_dot"), names(df)))
    end
end
