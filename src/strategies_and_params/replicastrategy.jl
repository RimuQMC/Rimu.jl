"""
    ReplicaStrategy{N}

Supertype for strategies that can be passed to [`ProjectorMonteCarloProblem`](@ref) and
control how many replicas are used, and what information is computed and returned. The
number of replicas is `N`.

## Concrete implementations

* [`NoStats`](@ref): run (possibly one) replica(s), but don't report any additional info.
* [`AllOverlaps`](@ref): report overlaps between all pairs of replica vectors.

## Interface

A subtype of `ReplicaStrategy{N}` must implement the following
function:

* [`Rimu.replica_stats`](@ref) - return a
  tuple of `String`s or `Symbols` of names for replica statistics and a tuple of the values.
  These will be reported to the `DataFrame` returned by [`ProjectorMonteCarloProblem`](@ref).
"""
abstract type ReplicaStrategy{N} end

"""
    num_replicas(state_or_strategy)

Return the number of replicas used in the simulation.
"""
num_replicas(::ReplicaStrategy{N}) where {N} = N

"""
    replica_stats(RS::ReplicaStrategy{N}, spectral_states::NTuple{N,SingleState}) -> (names, values)

Return the names and values of statistics related to `N` replica states consistent with the
[`ReplicaStrategy`](@ref) `RS`. `names`
should be a tuple of `Symbol`s or `String`s and `values` should be a tuple of the same
length. This function will be called every [`reporting_interval`](@ref) steps from
[`ProjectorMonteCarloProblem`](@ref), or once per time step if `reporting_interval` is not
defined.

Part of the [`ReplicaStrategy`](@ref) interface. See also [`SingleState`](@ref).
"""
replica_stats

"""
    NoStats(N=1) <: ReplicaStrategy{N}

The default [`ReplicaStrategy`](@ref). `N` replicas are run, but no statistics are
collected.

See also [`ProjectorMonteCarloProblem`](@ref).
"""
struct NoStats{N} <: ReplicaStrategy{N} end
NoStats(N=1) = NoStats{N}()

replica_stats(::NoStats, _) = (), ()
check_transform(::NoStats, _) = nothing

# TODO: add custom names
"""
    AllOverlaps(n_replicas=2; operator=nothing, transform=nothing, vecnorm=true, mixed_spectral_overlaps=false)
        <: ReplicaStrategy{n_replicas}

Run `n_replicas` replicas and report overlaps between all pairs of replica vectors. If
`operator` is not `nothing`, the overlap `dot(r1, operator, r2)` is reported as well. If
`operator` is a tuple of operators, the overlaps are computed for all operators.

Column names in the report are of the form `r{i}s{k}_dot_r{j}s{k}` for vector-vector
overlaps, and `r{i}s{k}_Op{m}_r{j}s{k}` for operator overlaps, where `i` and `j` label the
replicas, `k` labels the spectral state, and `m` labels the operators.

The `r{i}s{k}_dot_r{j}s{k}` overlap can be omitted with the flag `vecnorm=false`.

By default, overlaps of different spectral states are omitted. To include overlaps of
different spectral states `r{i}s{k}_dot_r{j}s{l}` and `r{i}s{k}_Op{m}_r{j}s{l}`, use the
flag `mixed_spectral_overlaps=true`. 

See [`ProjectorMonteCarloProblem`](@ref), [`ReplicaStrategy`](@ref) and
[`AbstractOperator`](@ref Interfaces.AbstractOperator) (for an interface for implementing
operators).

# Transformed Hamiltonians

If a transformed Hamiltonian `G` has been passed to [`ProjectorMonteCarloProblem`](@ref)
then overlaps can be calculated by passing the same transformed Hamiltonian to `AllOverlaps`
by setting `transform=G`. A warning is given if these two Hamiltonians do not match.

Implemented transformations are:

 * [`GutzwillerSampling`](@ref)
 * [`GuidingVectorSampling`](@ref)

In the case of a transformed Hamiltonian the overlaps are defined as follows. For a
similarity transformation `G` of the Hamiltonian (see e.g. [`GutzwillerSampling`](@ref).)
```math
    \\hat{G} = f \\hat{H} f^{-1}.
```
The expectation value of an operator ``\\hat{A}`` is
```math
    \\langle \\hat{A} \\rangle = \\langle \\psi | \\hat{A} | \\psi \\rangle
        = \\frac{\\langle \\phi | f^{-1} \\hat{A} f^{-1} | \\phi \\rangle}{\\langle \\phi | f^{-2} | \\phi \\rangle}
```
where
```math
    | \\phi \\rangle = f | \\psi \\rangle
```
is the (right) eigenvector of ``\\hat{G}`` and ``| \\psi \\rangle`` is an eigenvector of
``\\hat{H}``.

For an ``m``-tuple of input operators ``(\\hat{A}_1, ..., \\hat{A}_m)``, overlaps of
``\\langle \\phi | f^{-1} \\hat{A} f^{-1} | \\phi \\rangle`` are reported as
`r{i}s{k}_Op{m}_r{j}s{k}`. The correct vector-vector overlap ``\\langle \\phi | f^{-2} | \\phi
\\rangle`` is reported *last* as `r{i}s{k}_Op{m+1}_r{j}s{k}`. This is in addition to the *bare*
vector-vector overlap ``\\langle \\phi | \\phi \\rangle`` that is reported as 
`r{i}s{k}_dot_r{j}s{k}`.
"""
struct AllOverlaps{N,M,O,B,S} <: ReplicaStrategy{N}
    operators::O
end

const TupleOrVector = Union{Tuple, Vector}

function AllOverlaps(
    n_replicas=2;
    operator=nothing,
    transform=nothing,
    vecnorm=true,
    mixed_spectral_overlaps=false
)
    n_replicas isa Integer || throw(ArgumentError("n_replicas must be an integer"))
    if isnothing(operator)
        operators = ()
    elseif operator isa TupleOrVector
        if !(eltype(operator) <: AbstractOperator)
            throw(ArgumentError("operator must be an AbstractOperator or a Tuple or "*
                "Vector of AbstractHamiltonians"))
        end
        operators = operator
    else
        operators = (operator,)
    end
    if isnothing(transform)
        ops = operators
    else
        fsq = Rimu.Hamiltonians.TransformUndoer(transform)
        ops = (map(op -> Rimu.Hamiltonians.TransformUndoer(transform, op), operators)..., fsq)
    end
    if !vecnorm && length(ops) == 0
        return NoStats(n_replicas)
    end
    return AllOverlaps{n_replicas,length(ops),typeof(ops),vecnorm,mixed_spectral_overlaps}(ops)
end

function replica_stats(rs::AllOverlaps{N,<:Any,<:Any,B,S}, spectral_states::NTuple{N}) where {N,B,S}
    n_spectral = num_spectral_states(spectral_states[1])
    vecs = SMatrix{N,n_spectral}(
        spectral_states[i][j].v for i in 1:N, j in 1:n_spectral
    )
    wms = SMatrix{N,n_spectral}(
        spectral_states[i][j].wm for i in 1:N, j in 1:n_spectral
    )
    return all_overlaps(rs.operators, vecs, wms, Val(B), Val(S))
end

"""
    all_overlaps(operators, vectors, working_memories, vecnorm=true, mixed_spectral_overlaps=false)

Get all overlaps between vectors and operators.  The flag `vecnorm` can disable the
vector-vector overlap `r{i}s{k}_dot_r{j}s{k}`.
"""
function all_overlaps(
    operators::TupleOrVector, vecs::SMatrix{N,M,<:AbstractDVec}, wms, ::Val{B}, ::Val{S}
) where {N,M,B,S}
    T = promote_type((valtype(v) for v in vecs)..., eltype.(operators)...)
    names = String[]
    values = T[]
    for i in 1:N, k in 1:M
        if all(isdiag, operators)
            v = vecs[i,k]
        else
            v = DictVectors.copy_to_local!(wms[i,k], vecs[i,k])
        end

        if S
            for j in 1:N, l in k+1:M
                if B
                    push!(names, "r$(i)s$(k)_dot_r$(j)s$(l)")
                    push!(values, dot(vecs[i,k], vecs[j,l]))
                end
                for (m, op) in enumerate(operators)
                    push!(names, "r$(i)s$(k)_Op$(m)_r$(j)s$(l)")
                    # Using dot_from_right here because dot will try to copy_to_local! if
                    # called directly.
                    push!(values, dot_from_right(v, op, vecs[j,l]))
                end
            end
        end
        for j in i+1:N
            if B
                push!(names, "r$(i)s$(k)_dot_r$(j)s$(k)")
                push!(values, dot(vecs[i,k], vecs[j,k]))
            end
            for (m, op) in enumerate(operators)
                push!(names, "r$(i)s$(k)_Op$(m)_r$(j)s$(k)")
                push!(values, dot_from_right(v, op, vecs[j,k]))
            end
        end
    end

    num_reports = M * (N * (N - 1) ÷ 2) * (B + length(operators)) + S * N^2 * (M * (M - 1) ÷ 2) * (B + length(operators))
    return SVector{num_reports,String}(names).data, SVector{num_reports,T}(values).data
end

"""
    check_transform(r::AllOverlaps, ham)

Check that the transformation provided to `r::AllOverlaps` matches the given Hamiltonian `ham`.
Used as a sanity check before starting main [`ProjectorMonteCarloProblem`](@ref) loop.
"""
function check_transform(r::AllOverlaps, ham::AbstractHamiltonian)
    ops = r.operators
    if !isempty(ops)
        op_transform = all(op -> typeof(op)<:Rimu.Hamiltonians.TransformUndoer, ops)
        ham_transform = hasproperty(ham, :hamiltonian)    # need a better test for this
        if op_transform && ham_transform && !all(op -> ham == op.transform, ops)
            # both are transformed but different
            @warn "Overlaps transformation not consistent with Hamiltonian transformation."
        elseif op_transform ⊻ ham_transform
            # only one is transformed
            @warn "Expected overlaps and Hamiltonian to be transformed; got only one."
        end
    end
    return nothing
end
