# GlobalStepAction is defined in projector_monte_carlo_problem.jl to avoid circular
# dependencies

"""
    OperatorOverlaps(operator; name=:operator_overlaps) <: GlobalStepAction
Compute and report the overlaps ⟨ψ_i|O|ψ_j⟩ between all pairs of replica states
for a given operator `O`. The results are returned in a `NamedTuple` with a single field
with key `name` (default `:operator_overlaps`) and value array of overlaps.
"""
@kwdef struct OperatorOverlaps{OpType} <: GlobalStepAction
    operator::OpType
    name::Symbol = :operator_overlaps
end
function (ooa::OperatorOverlaps)(state::ReplicaState)
    n_specs = num_spectral_states(state)
    n_reps = num_replicas(state)
    vectors = state_vectors(state) # 2D array: (replica, spectral state)
    @assert n_specs == 1 "OperatorOverlaps currently only supports single spectral state simulations."
    expvals = map(StrictPairIter(n_reps)) do (i, j)
        bra = vectors[i,1]
        ket = vectors[j,1]
        dot(bra, ooa.operator, ket)
    end
    return NamedTuple((ooa.name => expvals,))
end

# StrictPairIter yields tuples (a,b) with 1 <= a < b <= n for n items
"""
    StrictPairIter(n::Int)
Iterator that yields all unique pairs (i, j) with 1 ≤ i < j ≤ n.
"""
struct StrictPairIter
    n::Int
end

Base.eltype(::Type{StrictPairIter}) = Tuple{Int,Int}
Base.IteratorSize(::Type{StrictPairIter}) = Base.HasLength()
Base.length(iter::StrictPairIter) = div(iter.n * (iter.n - 1), 2)

# start iteration
Base.iterate(iter::StrictPairIter) = Base.iterate(iter, (1, 2))

# iterate with state (i, j)
function Base.iterate(iter::StrictPairIter, state::Tuple{Int,Int})
    i, j = state
    n = iter.n
    while i < n
        if j <= n
            val = (i, j)
            # choose next state: advance j, or move to next i
            if j + 1 <= n
                return val, (i, j + 1)
            else
                return val, (i + 1, i + 2)
            end
        else
            i += 1
            j = i + 1
        end
    end
    return nothing
end
