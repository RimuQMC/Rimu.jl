"""
    SingleState(v, wm, pnorm, params, id)

Struct that holds all information needed for an independent run of the algorithm.
Can be advanced a step forward with [`advance!`](@ref).

See also [`ReplicaState`](@ref), [`ReplicaStrategy`](@ref), [`replica_stats`](@ref),
[`lomc!`](@ref).
"""
mutable struct SingleState{H,V,W,SS<:ShiftStrategy,TS<:TimeStepStrategy,SP}
    # Future TODO: rename these fields, add interface for accessing them.
    hamiltonian::H
    v::V       # vector
    pv::V      # previous vector.
    wm::W      # working memory. Maybe working memories could be shared among replicas?
    s_strat::SS # shift strategy
    τ_strat::TS # time step strategy
    shift_parameters::SP # norm, shift, time_step; is mutable
    id::String # id is appended to column names
end

function SingleState(h, v, wm, s_strat, τ_strat, shift, dτ::Float64, id="")
    if isnothing(wm)
        wm = similar(v)
    end
    pv = zerovector(v)
    sp = initialise_shift_parameters(s_strat, shift, walkernumber(v), dτ)
    return SingleState(h, v, pv, wm, s_strat, τ_strat, sp, id)
    # return SingleState{
    #     typeof(h),typeof(v),typeof(wm),typeof(s_strat),typeof(τ_strat),typeof(sp)
    # }(h, v, pv, wm, s_strat, τ_strat, sp, id)
end

function Base.show(io::IO, r::SingleState)
    print(
        io,
        "SingleState(v: ", length(r.v), "-element ", nameof(typeof(r.v)),
        ", wm: ", length(r.wm), "-element ", nameof(typeof(r.wm)), ")"
    )
end

"""
    _n_walkers(v, s_strat)
Returns an estimate of the expected number of walkers as an integer.
"""
function _n_walkers(v, s_strat)
    n = if hasfield(typeof(s_strat), :targetwalkers)
        s_strat.targetwalkers
    else # e.g. for LogUpdate()
        walkernumber(v)
    end
    return ceil(Int, max(real(n), imag(n)))
end

"""
    ReplicaState

Holds information about multiple replicas.
"""
struct ReplicaState{
    H,
    N,
    R<:NTuple{N,<:SingleState},
    RS<:ReportingStrategy,
    RRS<:ReplicaStrategy,
    PS<:NTuple{<:Any,PostStepStrategy},
}
    hamiltonian::H
    replica_states::R
    maxlength::Ref{Int}
    step::Ref{Int}
    # laststep::Ref{Int}
    simulation_plan::SimulationPlan
    reporting_strategy::RS
    post_step_strategy::PS
    replica_strategy::RRS
end

function ReplicaState(
    hamiltonian, v;
    step=nothing,
    laststep=nothing,
    simulation_plan=nothing,
    dτ=nothing,
    shift=nothing,
    wm=nothing,
    style=nothing,
    targetwalkers=1000,
    address=starting_address(hamiltonian),
    params::FciqmcRunStrategy=RunTillLastStep(
        laststep=100,
        shift=float(valtype(v))(diagonal_element(hamiltonian, address))
    ),
    s_strat::ShiftStrategy=DoubleLogUpdate(; targetwalkers),
    reporting_strategy::ReportingStrategy=ReportDFAndInfo(),
    τ_strat::TimeStepStrategy=ConstantTimeStep(),
    threading=nothing,
    replica_strategy::ReplicaStrategy=NoStats(),
    post_step_strategy=(),
    maxlength=2 * _n_walkers(v, s_strat) + 100, # padding for small walker numbers
)
    Hamiltonians.check_address_type(hamiltonian, keytype(v))
    # Set up reporting_strategy and params
    reporting_strategy = refine_reporting_strategy(reporting_strategy)

    # eventually we want to deprecate the use of params
    if !isnothing(params)
        if !isnothing(step)
            params.step = step
        end
        if !isnothing(laststep)
            params.laststep = laststep
        end
        if !isnothing(dτ)
            params.dτ = dτ
        end
        if !isnothing(shift)
            params.shift = shift
        end
        step = params.step
        dτ = params.dτ
        shift = params.shift
        laststep = params.laststep
    else
        if isnothing(step)
            step = 0
        end
        if isnothing(laststep)
            laststep = 100
        end
        if isnothing(dτ)
            dτ = 0.01
        end
        if isnothing(shift)
            shift = float(valtype(v))(diagonal_element(hamiltonian, address))
        end
    end

    if isnothing(simulation_plan)
        simulation_plan = SimulationPlan(;
            starting_step=step,
            last_step=laststep
        )
    end

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

    # Set up replica_states
    nreplicas = num_replicas(replica_strategy)
    if nreplicas > 1
        replica_states = ntuple(nreplicas) do i
            SingleState(
                hamiltonian,
                deepcopy(v),
                deepcopy(wm),
                deepcopy(s_strat),
                deepcopy(τ_strat),
                shift,
                dτ,
                "_$i")
        end
    else
        replica_states = (SingleState(hamiltonian, v, wm, s_strat, τ_strat, shift, dτ),)
    end

    return ReplicaState(
        hamiltonian, replica_states, Ref(Int(maxlength)),
        Ref(simulation_plan.starting_step), # step
        simulation_plan,
        # Ref(Int(laststep)),
        reporting_strategy, post_step_strategy, replica_strategy
    )
end

function Base.show(io::IO, st::ReplicaState)
    print(io, "ReplicaState")
    if length(st.replica_states) > 1
        print(io, " with ", length(st.replica_states), " replicas")
    end
    print(io, "\n  H:    ", st.hamiltonian)
    print(io, "\n  step: ", st.step[], " / ", st.simulation_plan.last_step)
    print(io, "\n  replicas: ")
    for (i, r) in enumerate(st.replica_states)
        print(io, "\n    $i: ", r)
    end
end

function report_default_metadata!(report::Report, state::ReplicaState)
    report_metadata!(report, "Rimu.PACKAGE_VERSION", Rimu.PACKAGE_VERSION)
    # add metadata from state
    replica = state.replica_states[1]
    shift_parameters = replica.shift_parameters
    report_metadata!(report, "laststep", state.simulation_plan.last_step)
    report_metadata!(report, "num_replicas", length(state.replica_states))
    report_metadata!(report, "hamiltonian", state.hamiltonian)
    report_metadata!(report, "reporting_strategy", state.reporting_strategy)
    report_metadata!(report, "s_strat", replica.s_strat)
    report_metadata!(report, "τ_strat", replica.τ_strat)
    report_metadata!(report, "dτ", shift_parameters.time_step)
    report_metadata!(report, "step", state.step[])
    report_metadata!(report, "shift", shift_parameters.shift)
    report_metadata!(report, "maxlength", state.maxlength[])
    report_metadata!(report, "post_step_strategy", state.post_step_strategy)
    report_metadata!(report, "v_summary", summary(state.replica_states[1].v))
    report_metadata!(report, "v_type", typeof(state.replica_states[1].v))
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
