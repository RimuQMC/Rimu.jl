# GlobalStepAction is defined in projector_monte_carlo_problem.jl to avoid circular
# dependencies

"""
    OperatorOverlaps(operator; name=:operator_overlaps) <: GlobalStepAction
Compute and report the overlaps ⟨ψ_i|O|ψ_j⟩ between all pairs of replica states
for a given operator `O` and for each spectral state as a matrix. The results are reported
with the name given by the `name` keyword argument.

See also [`GlobalStepAction`](@ref) and [`CoefficientVectorOverlaps`](@ref).
"""
@kwdef struct OperatorOverlaps{OpType} <: GlobalStepAction
    operator::OpType
    name::Symbol = :operator_overlaps
end
OperatorOverlaps(op; name::Symbol=:operator_overlaps) = OperatorOverlaps(op,name)

function (ooa::OperatorOverlaps)(state::ReplicaState)
    n_specs = num_spectral_states(state)
    n_reps = num_replicas(state)
    vectors = state_vectors(state) # 2D array: (replica, spectral state)
    overlaps = [dot(vectors[i, s], ooa.operator, vectors[j, s]) for
                (i, j) in StrictPairIter(n_reps),
                s in 1:n_specs
    ]
    return NamedTuple((ooa.name => overlaps,))
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

"""
    CoefficientVectorOverlaps(; name=:coefficient_vector_overlaps) <: GlobalStepAction

Compute and report the overlaps ⟨ψ_i|ψ_j⟩ between all pairs of replica states for each
spectral state as a matrix. The results are reported with the name given by the `name`
keyword argument.

See also [`GlobalStepAction`](@ref) and [`OperatorOverlaps`](@ref).
"""
@kwdef struct CoefficientVectorOverlaps <: GlobalStepAction
    name::Symbol = :coefficient_vector_overlaps
end
function (cvo::CoefficientVectorOverlaps)(state::ReplicaState)
    n_specs = num_spectral_states(state)
    n_reps = num_replicas(state)
    vectors = state_vectors(state) # 2D array: (replica, spectral state)
    overlaps = [dot(vectors[i,s], vectors[j,s]) for
        (i,j) in StrictPairIter(n_reps),
        s in 1:n_specs
    ]
    return NamedTuple((cvo.name => overlaps,))
end
