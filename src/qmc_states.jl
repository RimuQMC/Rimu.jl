"""
    SingleState(hamiltonian, algorithm, v, wm, pnorm, params, id)

Struct that holds a single state vector and all information needed for an independent run
of the algorithm. Can be advanced a step forward with [`advance!`](@ref).

See also [`SpectralState`](@ref), [`SpectralStrategy`](@ref),
[`ReplicaState`](@ref), [`ReplicaStrategy`](@ref), [`replica_stats`](@ref),
[`PMCSimulation`](@ref).
"""
mutable struct SingleState{H,A,V,W,SP}
    # Future TODO: rename these fields, add interface for accessing them.
    hamiltonian::H
    algorithm::A
    v::V       # vector
    pv::V      # previous vector.
    wm::W      # working memory. Maybe working memories could be shared among replicas?
    shift_parameters::SP # norm, shift, time_step; is mutable
    id::String # id is appended to column names
end

# This constructor is currently only used by lomc! and should not be used for new code.
function SingleState(h, v, wm, shift_strategy, time_step_strategy, shift, dτ::Float64, id="")
    if isnothing(wm)
        wm = similar(v)
    end
    pv = zerovector(v)
    sp = initialise_shift_parameters(shift_strategy, shift, walkernumber(v), dτ)
    alg = FCIQMC(; shift_strategy, time_step_strategy)
    return SingleState(h, alg, v, pv, wm, sp, id)
end

function Base.show(io::IO, r::SingleState)
    print(
        io,
        "SingleState(v: ", length(r.v), "-element ", nameof(typeof(r.v)),
        ", wm: ", length(r.wm), "-element ", nameof(typeof(r.wm)), ")"
    )
end

"""
    state_vectors(state)
Returns a collection of configuration vectors from the `state`.

See also [`single_states`](@ref), [`SingleState`](@ref), [`ReplicaState`](@ref),
[`SpectralState`](@ref), [`PMCSimulation`](@ref).
"""
function state_vectors(state::SingleState)
    return SVector(state.v,)
end

"""
    single_states(state)

Returns a collection of `SingleState`s from the `state`.

See also [`state_vectors`](@ref), [`SingleState`](@ref), [`ReplicaState`](@ref),
[`SpectralState`](@ref), [`PMCSimulation`](@ref).
"""
function single_states(state::SingleState)
    return SVector(state,)
end

"""
    SpectralState
Holds one or several [`SingleState`](@ref)s representing the ground state and excited
states of a single replica.

See also [`SpectralStrategy`](@ref), [`ReplicaState`](@ref), [`SingleState`](@ref),
[`PMCSimulation`](@ref).
"""
struct SpectralState{
    N,
    NS<:NTuple{N,SingleState},
    SS<:SpectralStrategy{N}
}
    single_states::NS # Tuple of SingleState
    spectral_strategy::SS # SpectralStrategy
    id::String # id is appended to column names
end
function SpectralState(t::Tuple, ss::SpectralStrategy, id="")
    return SpectralState(t, ss, id)
end
num_spectral_states(::SpectralState{N}) where {N} = N

function Base.show(io::IO, s::SpectralState)
    print(io, "SpectralState")
    print(io, " with ", num_spectral_states(s), " spectral states")
    print(io, "\n    spectral_strategy: ", s.spectral_strategy)
    for (i, r) in enumerate(s.single_states)
        print(io, "\n      $i: ", r)
    end
end

function state_vectors(state::SpectralState)
    return mapreduce(state_vectors, vcat, state.single_states)
end

function single_states(state::SpectralState)
    return mapreduce(single_states, vcat, state.single_states)
end

"""
    _n_walkers(v, shift_strategy)
Returns an estimate of the expected number of walkers as an integer.
"""
function _n_walkers(v, shift_strategy)
    n = if hasfield(typeof(shift_strategy), :targetwalkers)
        shift_strategy.targetwalkers
    else # e.g. for LogUpdate()
        walkernumber(v)
    end
    return ceil(Int, max(real(n), imag(n)))
end

"""
    ReplicaState

Holds information about multiple replicas of [`SpectralState`](@ref)s.

See also [`ReplicaStrategy`](@ref), [`SpectralState`](@ref)s, [`SingleState`](@ref),
[`PMCSimulation`](@ref).
"""
struct ReplicaState{
    N, # number of replicas
    S, # number of spectral states
    H,
    R<:NTuple{N,<:SpectralState{S}},
    RS<:ReportingStrategy,
    RRS<:ReplicaStrategy,
    PS<:NTuple{<:Any,PostStepStrategy},
}
    hamiltonian::H
    spectral_states::R
    maxlength::Ref{Int}
    step::Ref{Int}
    simulation_plan::SimulationPlan
    reporting_strategy::RS
    post_step_strategy::PS
    replica_strategy::RRS
end

# This constructor is currently only used by lomc! and should not be used for new code.
# It may be removed in the future.
function ReplicaState(
    hamiltonian, v;
    starting_step = 0,
    last_step = 100,
    simulation_plan = SimulationPlan(; starting_step, last_step),
    time_step = 0.01,
    address = starting_address(hamiltonian),
    shift = float(valtype(v))(diagonal_element(hamiltonian, address)),
    wm = nothing,
    style = nothing,
    targetwalkers = 1000,
    shift_strategy::ShiftStrategy = DoubleLogUpdate(; targetwalkers),
    reporting_strategy::ReportingStrategy = ReportDFAndInfo(),
    time_step_strategy::TimeStepStrategy = ConstantTimeStep(),
    threading = nothing,
    replica_strategy::ReplicaStrategy = NoStats(),
    spectral_strategy::SpectralStrategy = GramSchmidt(),
    post_step_strategy = (),
    maxlength=2 * _n_walkers(v, shift_strategy) + 100, # padding for small walker numbers
)
    Hamiltonians.check_address_type(hamiltonian, keytype(v))
    # Set up reporting_strategy and params
    reporting_strategy = refine_reporting_strategy(reporting_strategy)

    if threading ≠ nothing
        @warn "Starting vector is provided. Ignoring `threading=$threading`."
    end
    if style ≠ nothing
        @warn "Starting vector is provided. Ignoring `style=$style`."
    end
    wm = isnothing(wm) ? working_memory(v) : wm

    # Set up post_step_strategy
    if !(post_step_strategy isa Tuple)
        post_step_strategy = (post_step_strategy,)
    end

    # set up single_states
    n_spectral_states = num_spectral_states(spectral_strategy)
    n_spectral_states == 1 || throw(ArgumentError("Only one spectral state is supported."))

    # Set up spectral_states
    nreplicas = num_replicas(replica_strategy)
    if nreplicas > 1
        spectral_states = ntuple(nreplicas) do i
            SpectralState(
                (SingleState(
                    hamiltonian,
                    deepcopy(v),
                    deepcopy(wm),
                    deepcopy(shift_strategy),
                    deepcopy(time_step_strategy),
                    shift,
                    time_step,
                    "_$i"),
                ),
                spectral_strategy
            )
        end
    else
        spectral_states = (SpectralState(
            (SingleState(hamiltonian, v, wm, shift_strategy, time_step_strategy, shift, time_step),),
            spectral_strategy
        ),)
    end

    return ReplicaState(
        hamiltonian, spectral_states, Ref(Int(maxlength)),
        Ref(simulation_plan.starting_step), # step
        simulation_plan,
        # Ref(Int(laststep)),
        reporting_strategy, post_step_strategy, replica_strategy
    )
end

num_replicas(::ReplicaState{N}) where {N} = N
num_spectral_states(::ReplicaState{<:Any, S}) where {S} = S

function Base.show(io::IO, st::ReplicaState)
    print(io, "ReplicaState")
    print(io, " with ", num_replicas(st), " replicas  and ")
    print(io, num_spectral_states(st), " spectral states")
    print(io, "\n  H:    ", st.hamiltonian)
    print(io, "\n  step: ", st.step[], " / ", st.simulation_plan.last_step)
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.spectral_states)
        print(io, "\n    $i: ", r)
    end
end

"""
    state_vectors(state::ReplicaState)
Returns a collection of configuration vectors from the `state`.
The vectors can be accessed by indexing the resulting collection, where the row index
corresponds to the spectral state index and the column index corresponds to the row index.

See also [`single_states`](@ref), [`SingleState`](@ref), [`ReplicaState`](@ref),
[`SpectralState`](@ref), [`PMCSimulation`](@ref).
"""
@inline function state_vectors(state::ReplicaState{N,S}) where {N,S}
    # Annoyingly this function is allocating if N > 1
    return SMatrix{S,N}(
        state.spectral_states[fld1(i,S)].single_states[mod1(i,S)].v for i in 1:N*S
    )
    # return SMatrix{S,N}(mapreduce(state_vectors, hcat, state.spectral_states))
end

"""
    single_states(state::ReplicaState)
Returns a collection of `SingleState`s from the `state`.
The `SingleState`s can be accessed by indexing the resulting collection, where the row index
corresponds to the spectral state index and the column index corresponds to the row index.

See also [`state_vectors`](@ref), [`SingleState`](@ref), [`ReplicaState`](@ref),
[`SpectralState`](@ref), [`PMCSimulation`](@ref).
"""
function single_states(state::ReplicaState{N,S}) where {N,S}
    return SMatrix{S,N}(mapreduce(single_states, hcat, state.spectral_states))
end

function report_default_metadata!(report::Report, state::ReplicaState)
    report_metadata!(report, "Rimu.PACKAGE_VERSION", Rimu.PACKAGE_VERSION)
    # add metadata from state
    replica = state.spectral_states[1].single_states[1]
    algorithm = replica.algorithm
    shift_parameters = replica.shift_parameters
    report_metadata!(report, "laststep", state.simulation_plan.last_step)
    report_metadata!(report, "num_replicas", num_replicas(state))
    report_metadata!(report, "num_spectral_states", num_spectral_states(state))
    report_metadata!(report, "hamiltonian", state.hamiltonian)
    report_metadata!(report, "reporting_strategy", state.reporting_strategy)
    report_metadata!(report, "shift_strategy", algorithm.shift_strategy)
    report_metadata!(report, "time_step_strategy", algorithm.time_step_strategy)
    report_metadata!(report, "time_step", shift_parameters.time_step)
    report_metadata!(report, "step", state.step[])
    report_metadata!(report, "shift", shift_parameters.shift)
    report_metadata!(report, "maxlength", state.maxlength[])
    report_metadata!(report, "post_step_strategy", state.post_step_strategy)
    report_metadata!(report, "v_summary", summary(replica.v))
    report_metadata!(report, "v_type", typeof(replica.v))
    return report
end

"""
    default_starting_vector(hamiltonian::AbstractHamiltonian; kwargs...)
    default_starting_vector(
        address=starting_address(hamiltonian);
        style=IsStochasticInteger(),
        initiator=NonInitiator(),
        threading=nothing
    )
Return a default starting vector for [`lomc!`](@ref). The default choice for the starting
vector is
```julia
v = PDVec(address => 10; style, initiator)
```
if threading is available, or otherwise
```julia
v = DVec(address => 10; style)
```
if `initiator == NonInitiator()`, and
```julia
v = InitiatorDVec(address => 10; style, initiator)
```
if not. See [`PDVec`](@ref), [`DVec`](@ref), [`InitiatorDVec`](@ref),
[`StochasticStyle`](@ref), and [`InitiatorRule`].
"""
function default_starting_vector(
    hamiltonian::AbstractHamiltonian;
    address=starting_address(hamiltonian), kwargs...
)
    return default_starting_vector(address; kwargs...)
end
function default_starting_vector(address;
    style=IsStochasticInteger(),
    threading=nothing,
    initiator=NonInitiator(),
)
    if isnothing(threading)
        threading = Threads.nthreads() > 1
    end
    if threading
        v = PDVec(address => 10; style, initiator)
    elseif initiator isa NonInitiator
        v = DVec(address => 10; style)
    else
        v = InitiatorDVec(address => 10; style, initiator)
    end
    return v
end
