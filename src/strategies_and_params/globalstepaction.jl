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

"""
     TestOneParticleDensity(v; normalize=true) <: AbstractOperator{typeof(v)}
A one particle operator constructed from a provided test vector `v`. An expectation value
with this operator yields a lower bound on the largest eigenvalue (and an upper bound on the
smallest eigenvalue) of the one-particle density matrix.
If `normalize` is true (default), the vector is normalized before use.

```math
    ρ̂ = ∑_{ij} v_i^* v_j â^†_{i} â_{j}
```
"""
struct TestOneParticleDensity{T,V<:SVector{<:Any,T},M} <: AbstractObservable{T}
    test_vector::V
end
function TestOneParticleDensity(v; normalize=true)
    if normalize
        v = v / norm(v)
    end
    M = length(v)
    T = float(eltype(v))
    tv = SVector{M,T}(v)
    return TestOneParticleDensity{T,typeof(tv),M}(tv)
end

function Interfaces.allows_address_type(
    ::TestOneParticleDensity{T,A,M}, ::Type{B}
) where {T,A,M,B}
    B <: SingleComponentFockAddress && num_modes(B) == M
end

struct TestOneParticleDensityColumn{A,T,O,OMM} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    omm::OMM
end
function Interfaces.operator_column(o::TestOneParticleDensity, add::A) where {A}
    allows_address_type(o, A) || throw(ArgumentError("Address type not allowed for this operator"))
    omm = occupied_mode_map(add)
    return TestOneParticleDensityColumn{A,eltype(o),typeof(o),typeof(omm)}(o, add, omm)
end
Interfaces.parent_operator(c::TestOneParticleDensityColumn) = c.operator
Interfaces.starting_address(c::TestOneParticleDensityColumn) = c.address
function Interfaces.diagonal_element(c::TestOneParticleDensityColumn{<:Any,T}) where {T}
    val = zero(T)
    @inbounds for idx in c.omm
        val += abs2(c.operator.test_vector[idx.mode])*idx.occnum
    end
    return val
end
function Interfaces.num_offdiagonals(c::TestOneParticleDensityColumn)
    return length(c.omm) * (num_modes(c.address) - 1)
end
function Interfaces.offdiagonals(c::TestOneParticleDensityColumn)
    TestOneParticleDensityOffdiagonals(c)
end
struct TestOneParticleDensityOffdiagonals{C}
    column::C
end
Base.IteratorSize(::TestOneParticleDensityOffdiagonals) = Base.SizeUnknown()
# Base.length(od::TestOneParticleDensityOffdiagonals) = num_offdiagonals(od.column)
function Base.iterate(od::TestOneParticleDensityOffdiagonals, state=(1, 1))
    c = od.column
    omm = c.omm
    n_modes = num_modes(c.address)
    i, j = state # i: mode number for creation, j: index in omm for annihilation
    #  ∑_{ij} v_i^* v_j â^†_{i} â_{j} |address⟩
    while j <= length(omm)
        src = omm[j]
        while i <= n_modes
            if i != src.mode # omit same mode as they contribute to diagonal
                # create new address with excitation
                dst = find_mode(c.address, i)
                address, value = excitation(c.address, (dst,), (src,))
                if !iszero(value)
                    value *= conj(c.operator.test_vector[i]) * c.operator.test_vector[src.mode]
                    # choose next state
                    if i + 1 <= n_modes
                        return (address, value), (i + 1, j)
                    else
                        return (address, value), (1, j+1)
                    end
                end
            end
            i += 1
        end
        j += 1
        i = 1
    end
    return nothing
end
