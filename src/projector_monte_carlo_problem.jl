"""
    PMCAlgorithm
Abstract type for projector Monte Carlo algorithms.

See [`ProjectorMonteCarloProblem`](@ref), [`FCIQMC`](@ref).
"""
abstract type PMCAlgorithm end

"""
    SimulationPlan(; starting_step = 1, last_step = 100, wall_time = Inf)
Defines the duration of the simulation. The simulation ends when the `last_step` is reached
or the `wall_time` is exceeded.

See [`ProjectorMonteCarloProblem`](@ref), [`PMCSimulation`](@ref).
"""
Base.@kwdef struct SimulationPlan
    starting_step::Int = 0
    last_step::Int = 100
    wall_time::Float64 = Inf
end
function Base.show(io::IO, plan::SimulationPlan)
    print(
        io, "SimulationPlan(starting_step=", plan.starting_step,
        ", last_step=", plan.last_step, ", wall_time=", plan.wall_time, ")"
    )
end

"""
    ProjectorMonteCarloProblem(hamiltonian::AbstractHamiltonian; kwargs...)
Defines a problem to be solved by projector quantum Monte Carlo (QMC) methods, such as the
the [`FCIQMC`](@ref) algorithm.

# Common keyword arguments and defaults:
- `time_step = 0.01`: Initial time step size.
- `last_step = 100`: Controls the number of steps.
- `target_walkers = 1_000`: Target for the 1-norm of the coefficient vector.
- `start_at = starting_address(hamiltonian)`: Define the initial state vector(s).
    An ``r × s`` matrix of state vectors can be passed where ``r`` is the
    number of replicas and ``s`` the number of spectral states. See also
    [`default_starting_vector`](@ref).
- `style = IsDynamicSemistochastic()`: The [`StochasticStyle`](@ref) of the simulation.
- `initiator = false`: Whether to use initiators. Can be `true`, `false`, or a valid
    [`InitiatorRule`](@ref).
- `threading`: Default is to use multithreading and/or
  [MPI](https://juliaparallel.org/MPI.jl/latest/) if available. Set to
  `true` to force [`PDVec`](@ref) for the starting vector, `false` for serial computation;
  may be overridden by `start_at`.
- `reporting_strategy = ReportDFAndInfo()`: How and when to report results, see
  [`ReportingStrategy`](@ref).
- `post_step_strategy = ()`: Extract observables (e.g.
  [`ProjectedEnergy`](@ref)), see [`PostStepStrategy`](@ref).
- `n_replicas = 1`: Number of synchronised independent simulations.
- `replica_strategy = NoStats(n_replicas)`: Which results to report from replica
  simulations, see [`ReplicaStrategy`](@ref).
- `n_spectral = 1`: Number of targeted spectral states. Set `n_spectral > 1` to find excited
  states.
- `spectral_strategy = GramSchmidt(n_spectral)`: The [`SpectralStrategy`](@ref) used for
  orthogonalizing spectral states.

# Example

```jldoctest
julia> hamiltonian = HubbardReal1D(BoseFS(1,2,3));

julia> problem = ProjectorMonteCarloProblem(hamiltonian; target_walkers = 500, last_step = 100);

julia> simulation = solve(problem);

julia> simulation.success[]
true

julia> size(DataFrame(simulation))
(100, 9)
```

# Further keyword arguments:
- `starting_step = 1`: Starting step of the simulation.
- `wall_time = Inf`: Maximum time allowed for the simulation.
- `simulation_plan = SimulationPlan(; starting_step, last_step, wall_time)`: Defines the
    duration of the simulation. Takes precedence over `last_step` and `wall_time`.
- `ζ = 0.08`: Damping parameter for the shift update.
- `ξ = ζ^2/4`: Forcing parameter for the shift update.
- `shift_strategy = DoubleLogUpdate(; target_walkers, ζ, ξ)`: How to update the `shift`,
    see [`ShiftStrategy`](@ref).
- `time_step_strategy = ConstantTimeStep()`: Adjust time step or not, see
    `TimeStepStrategy`.
- `algorithm = FCIQMC(; shift_strategy, time_step_strategy)`: The algorithm to use.
    Currenlty only [`FCIQMC`](@ref) is implemented.
- `shift`: Initial shift value or collection of shift values. Determined by default from the
    Hamiltonian and the starting vectors.
- `initial_shift_parameters`: Initial shift parameters or collection of initial shift
    parameters. Overrides `shift` if provided.
- `max_length = 2 * target_walkers + 100`: Maximum length of the vectors.
- `display_name = "PMCSimulation"`: Name displayed in progress bar (via `ProgressLogging`).
- `metadata`: User-supplied metadata to be added to the report. Must be an iterable of
  pairs or a `NamedTuple`, e.g. `metadata = ("key1" => "value1", "key2" => "value2")`.
  All metadata is converted to strings.
- `random_seed = true`: Provide and store a seed for the random number generator. If set to
    `true`, a new random seed is generated from `RandomDevice()`. If set to number, this
    number is used as the seed. This seed is used by `solve` (and `init`) to re-seed the
    default random number generator (consistently on each MPI rank) such that
    `solve`ing the same `ProjectorMonteCarloProblem` twice will yield identical results. If
    set to `false`, no seed is used and consecutive random numbers are used.
- `minimum_size = 2*num_spectral_states(spectral_strategy)`: The minimum size of the basis
    used to construct starting vectors for simulations of spectral states, if `start_at`
    is not provided.

See also [`init`](@ref), [`solve`](@ref).
"""
struct ProjectorMonteCarloProblem{N,S} # is not type stable but does not matter
    # N is the number of replicas, S is the number of spectral states
    algorithm::PMCAlgorithm
    hamiltonian::AbstractHamiltonian
    start_at  # starting_vectors
    style::StochasticStyle
    initiator::InitiatorRule
    threading::Bool
    simulation_plan::SimulationPlan
    replica_strategy::ReplicaStrategy{N}
    initial_shift_parameters
    reporting_strategy::ReportingStrategy
    post_step_strategy::Tuple
    spectral_strategy::SpectralStrategy{S}
    max_length::Int
    metadata::LittleDict{String,String} # user-supplied metadata + display_name
    random_seed::Union{Nothing,UInt64}
    minimum_size::Int
end

function Base.show(io::IO, p::ProjectorMonteCarloProblem)
    nr = num_replicas(p)
    ns = num_spectral_states(p)
    println(io, "ProjectorMonteCarloProblem with $nr replica(s) and $ns spectral state(s):")
    isnothing(p.algorithm) || println(io, "  algorithm = ", p.algorithm)
    println(io, "  hamiltonian = ", p.hamiltonian)
    println(io, "  style = ", p.style)
    println(io, "  initiator = ", p.initiator)
    println(io, "  threading = ", p.threading)
    println(io, "  simulation_plan = ", p.simulation_plan)
    println(io, "  replica_strategy = ", p.replica_strategy)
    print(io, "  reporting_strategy = ", p.reporting_strategy)
    println(io, "  post_step_strategy = ", p.post_step_strategy)
    println(io, "  spectral_strategy = ", p.spectral_strategy)
    println(io, "  max_length = ", p.max_length)
    println(io, "  metadata = ", p.metadata)
    print(io, "  random_seed = ", p.random_seed)
end


function ProjectorMonteCarloProblem(
    hamiltonian::AbstractHamiltonian;
    n_replicas = 1,
    start_at = starting_address(hamiltonian),
    shift = nothing,
    style = IsDynamicSemistochastic(),
    initiator = false,
    threading = nothing,
    time_step = 0.01,
    starting_step = 0,
    last_step = 100,
    wall_time = Inf,
    walltime = nothing, # deprecated
    simulation_plan = nothing,
    replica_strategy = NoStats(n_replicas),
    targetwalkers = nothing, # deprecated
    target_walkers = 1_000,
    ζ = 0.08,
    ξ = ζ^2/4,
    shift_strategy = nothing,
    time_step_strategy=ConstantTimeStep(),
    algorithm=nothing,
    initial_shift_parameters=nothing,
    reporting_strategy = ReportDFAndInfo(),
    post_step_strategy = (),
    n_spectral = 1,
    spectral_strategy = GramSchmidt(n_spectral),
    minimum_size = 2*num_spectral_states(spectral_strategy),
    max_length = nothing,
    maxlength = nothing, # deprecated
    metadata = nothing,
    display_name = "PMCSimulation",
    random_seed = true
)
    if !isnothing(walltime)
        @warn "The keyword argument `walltime` is deprecated. Use `wall_time` instead."
        wall_time = walltime
    end
    if isnothing(simulation_plan)
        simulation_plan = SimulationPlan(starting_step, last_step, wall_time)
    end

    if !isnothing(targetwalkers)
        @warn "The keyword argument `targetwalkers` is deprecated. Use `target_walkers` instead."
        target_walkers = targetwalkers
    end

    if isnothing(shift_strategy)
        shift_strategy = DoubleLogUpdate(; target_walkers, ζ, ξ)
    end

    if isnothing(algorithm)
        algorithm = FCIQMC(; shift_strategy, time_step_strategy)
    end

    n_replicas = num_replicas(replica_strategy) # replica_strategy may override n_replicas

    n_spectral = num_spectral_states(spectral_strategy) # spectral_strategy may override n_spectral

    if replica_strategy isa AllOverlaps && n_spectral > 1
        throw(ArgumentError("AllOverlaps is not implemented for more than one spectral state."))
    end

    if random_seed == true
        random_seed = rand(RandomDevice(),UInt64)
    elseif random_seed == false
        random_seed = nothing
    elseif !isnothing(random_seed)
        random_seed = UInt64(random_seed)
    end

    if initiator isa Bool
        initiator = initiator ? Initiator() : NonInitiator()
    end

    if isnothing(threading)
        s_strat = algorithm.shift_strategy
        if !hasfield(typeof(s_strat), :target_walkers) || abs(s_strat.target_walkers) > 1_000
            threading = Threads.nthreads() > 1
        else
            threading = false
        end
    end

    # a proper setup of initial_shift_parameters is done in PMCSimulation
    # here we just store the initial shift and time_step if initial_shift_parameters is not
    # provided
    if isnothing(initial_shift_parameters)
        initial_shift_parameters = (; shift, time_step)
    end

    shift_strategy = algorithm.shift_strategy

    if isnothing(maxlength) # deprecated
        if isdefined(shift_strategy, :target_walkers)
            maxlength = round(Int, 2 * abs(shift_strategy.target_walkers) + 100)
        else
            maxlength = round(Int, 2 * abs(target_walkers) + 100)
        end
        # padding for small walkernumbers
    else
        @warn "The keyword argument `maxlength` is deprecated. Use `max_length` instead."
    end
    max_length = isnothing(max_length) ? maxlength : max_length

    # convert metadata to LittleDict
    report = Report()
    report_metadata!(report, "display_name", display_name)
    isnothing(metadata) || report_metadata!(report, metadata) # add user metadata
    metadata = report.meta::LittleDict{String, String}

    # set up post_step_strategy as a tuple
    if post_step_strategy isa PostStepStrategy
        post_step_strategy = (post_step_strategy,)
    end

    if !(eltype(hamiltonian)<: Real)
        throw(ArgumentError("Only real-valued Hamiltonians are currently supported "*
            "for ProjectorMonteCarloProblem. Please get in touch with the Rimu.jl " *
            "developers if you need a complex-valued Hamiltonian!"))
    end

    return ProjectorMonteCarloProblem{n_replicas,num_spectral_states(spectral_strategy)}(
        algorithm,
        hamiltonian,
        start_at, # starting_vectors,
        style,
        initiator,
        threading,
        simulation_plan,
        replica_strategy,
        initial_shift_parameters,
        reporting_strategy,
        post_step_strategy,
        spectral_strategy,
        max_length,
        metadata,
        random_seed,
        minimum_size
    )
end

num_replicas(::ProjectorMonteCarloProblem{N}) where N = N
num_spectral_states(::ProjectorMonteCarloProblem{<:Any,S}) where {S} = S
