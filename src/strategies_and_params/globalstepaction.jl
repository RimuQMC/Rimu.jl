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

"""
    ParticleDensityGradientOverlap(op; name = (:gradient_vector_overlaps, 
        :coefficient_vector_overlaps), test_vector_function,
        parameter=[SVector{1,Float64}(ones(Float64, 1))], 
        normalise::Bool=true) <: GlobalStepAction

Compute and report the particle density gradient overlaps ⟨ψ_i|∂O/∂α|ψ_j⟩  and 
coefficient vector overlaps ⟨ψ_i|ψ_j⟩ between all pairs of replica states for 
a given operator `O` and its parameters `α`. `parameter' is of type 
`Vector{SVector}` where each index refers to perticular spectral state.
`test_vector_function` is a nothing or a function of `(α, M)` depending on whether
the optimization is applied to entire Vector or it with the fixed functional form.
The results are returned in a `NamedTuple` with a single field with key `name` 
(default `(:gradient_vector_overlaps, :coefficient_vector_overlaps`) and 
value array of overlaps.

"""
mutable struct ParticleDensityGradientOverlap{OpType, normalize, P} <: GlobalStepAction
    operators::OpType
    name::Tuple{Symbol, Symbol}
    test_vector_function::Union{Function, Nothing}
    parameter::P
end

function ParticleDensityGradientOverlap(op; name = (:gradient_vector_overlaps, 
        :coefficient_vector_overlaps), test_vector_function=nothing, 
        parameter::Vector=[SVector{1,Float64}(ones(Float64, 1))],
        normalize::Bool=true)
    return ParticleDensityGradientOverlap{typeof(op), normalize, typeof(parameter)}(op, name, 
        test_vector_function, parameter)
end

function (ooa::ParticleDensityGradientOverlap{<:Any, normalize})(
        state::ReplicaState) where {normalize}

    n_specs = num_spectral_states(state)
    n_reps = num_replicas(state)
    vectors = state_vectors(state) # 2D array: (replica, spectral state)
    M = num_modes(keytype(vectors[1,1]))
    full_vector = isnothing(ooa.test_vector_function) ? true : false
    gradient = zeros(eltype(ooa.parameter), binomial(n_reps, 2) , n_specs)
    coeff = zeros(eltype(ooa.parameter[1]), binomial(n_reps,2), n_specs)

    for s in 1:n_specs
        gev = isnothing(ooa.test_vector_function) ? (ooa.parameter[s],) : 
            ooa.test_vector_function(ooa.parameter[s], M)

        coeff[:, s] = [dot(vectors[i, s], vectors[j, s]) for (i, j) in StrictPairIter(n_reps)]
        
        if iszero(sum(coeff[:,s]))
            gradient[:,s] = [zero(ooa.parameter[s]) for _ in 1:binomial(n_reps,2)]
        else
            zeta = sum([dot(vectors[i, s], ooa.operators[1](gev[1];normalize), 
                vectors[j, s]) for (i, j) in StrictPairIter(n_reps)])/sum(coeff[:,s])

            gradient[:,s] += [dot(vectors[i, s], ooa.operators[2](gev;normalize, zeta, 
                full_vector), vectors[j, s]) for (i, j) in StrictPairIter(n_reps)]
        end
    end
    return (ooa.name[1] => gradient, ooa.name[2] => coeff,)
end

"""
    OverlapwithOptimization(gradientoverlap; name = :parameter, 
        method = RAdam(0.1), step=100, threshold = 1e-3)) <: GlobalStepAction

Compute and report the particle density gradient overlaps ⟨ψ_i|∂O/∂α|ψ_j⟩ and 
optimize the parameters of `gradientoverlap`(<:GlobalStepAction) after every 
`step` number of collected gradient data in the FCIQMC simulation between all 
pairs of replica states for a given operator `O`. The results are returned in a 
`NamedTuple` with a field provided from gradientoverlap and single field with
key `name` (default `:parameters`) and value array of overlaps. 
    The optimization is carried out using the optimization `method` 
(default to RAdam(0.1)) which is downloaded from `Optimisers.jl`. There are 
other methods that can be usedsuch as Adam and Momentum.
    FCIQMC simulation is setup to be turminate when the sum of the absolute 
value of gradient become smaller then `threshold` (default to 1e-3). 

# Examples

```jldoctest
julia> address = FermiFS(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
FermiFS{5,10}(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

julia> h = HubbardRealSpace(address; w=-1.0)
HubbardRealSpace(
  fs"|↑↑↑↑↑⋅⋅⋅⋅⋅⟩",
  geometry = CubicGrid((10,), (true,)),
  t = [1.0;;],
  u = [1.0;;],
  w = [-1.0;;],
)

julia> parameter = [SVector{45,Float64}([1/45 for _ in 1:45])];

julia> gop = ParticleDensityGradientOverlap((TestTwoParticleDensity,
                   TwoParticleDensityGradient); name=(:gradient_test_overlaps,
                   :coefficient_vector_overlaps), 
                   test_vector_function = nothing, parameter);

julia> oops = OverlapwithOptimization(gop; name = :parameter, step = 5, 
                   threshold = 1e-2, method = Adam(0.1))
OverlapwithOptimization{0.01, 
    @NamedTuple{x::Optimisers.Leaf{Adam{Float64, Tuple{Float64, Float64}, Float64},
     Tuple{Vector{Float64}, Vector{Float64}, Tuple{Float64, Float64}}}}}(
        ParticleDensityGradientOverlap{Tuple{UnionAll, UnionAll}, true, 
        Vector{SVector{45, Float64}}}((TestTwoParticleDensity, 
        TwoParticleDensityGradient), (:gradient_test_overlaps, 
        :coefficient_vector_overlaps), nothing, 
        SVector{45, Float64}[[0.022222222222222223, ...0.022222222222222223]]), 
        :parameter, (x = Leaf(Adam(eta=0.1, beta=(0.9, 0.999), epsilon=1.0e-8), 
        ([0.0,...,0.0], [0.0, ... 0.0], (0.9, 0.999))),), 5)

julia> p = ProjectorMonteCarloProblem(h; n_replicas=3, global_step_actions=(oops,))
ProjectorMonteCarloProblem with 3 replica(s) and 1 spectral state(s):
    algorithm = FCIQMC(DoubleLogUpdate{Int64}(1000, 0.08, 0.0016), ConstantTimeStep())
    hamiltonian = HubbardRealSpace(
    fs"|↑↑↑↑↑⋅⋅⋅⋅⋅⟩",
    geometry = CubicGrid((10,), (true,)),
    t = [1.0;;],
    u = [1.0;;],
    w = [-1.0;;],
    )
    style = IsDynamicSemistochastic{Float64,ThresholdCompression,DynamicSemistochastic}()
    initiator = NonInitiator()
    threading = false
    simulation_plan = SimulationPlan(starting_step=0, last_step=100, wall_time=Inf)
    replica_strategy = NoStats{3}()
    reporting_strategy = ReportDFAndInfo
    reporting_interval: Int64 1
    info_interval: Int64 100
    io: Base.TTY
    writeinfo: Bool false
    post_step_strategy = ()
    spectral_strategy = GramSchmidt(1; orthogonalization_interval=1)
    global_step_actions = (OverlapwithOptimization{0.01, ...}(
        ParticleDensityGradientOverlap...),)
    metadata = OrderedCollections.LittleDict("display_name" => "PMCSimulation")
    random_seed = 2730854655602855441

julia> solve(p)
Progress: 100%|█████████████████████████████████████████████████| Time: 0:00:22
PMCSimulation with 3 replica(s) and 1 spectral state(s).
  Algorithm:   FCIQMC(DoubleLogUpdate{Int64}(1000, 0.08, 0.0016), ConstantTimeStep())
  Hamiltonian: HubbardRealSpace(
  fs"|↑↑↑↑↑⋅⋅⋅⋅⋅⟩",
  geometry = CubicGrid((10,), (true,)),
  t = [1.0;;],
  u = [1.0;;],
  w = [-1.0;;],
)
  Step:        100 / 100
  modified = true, aborted = false, success = true
```
"""
mutable struct OverlapwithOptimization{threshold,T} <: GlobalStepAction
    gradientoverlap::GlobalStepAction
    name::Symbol
    Setup::T
    Step::Int
end

function OverlapwithOptimization(gradientoverlap; name = :parameter, 
        method = RAdam(0.1), step=100, threshold = 1e-3)
    _setup = setup(method, (x = destructure(gradientoverlap.parameter)[1],))
    return OverlapwithOptimization{threshold, typeof(_setup)}(gradientoverlap, name, 
        _setup, step)
end

function (ooa::OverlapwithOptimization)(state::ReplicaState) 
    return NamedTuple(((ooa.gradientoverlap)(state)...,
                ooa.name=>ooa.gradientoverlap.parameter,))
end

function (ooa::OverlapwithOptimization{threshold})(df::DataFrame) where threshold
    # destructure the parameters of each spectrual state to a single Vector.
    para = (x = destructure(ooa.gradientoverlap.parameter)[1],)
    v, re = destructure(grad_rrs(ooa, df))
    g = (x = -v,)
    _setup, para = update(ooa.Setup, para, g);
    
    # restructure (re) the parameters of each spectrual state to a seperate SVector. 
    ooa.gradientoverlap.parameter = re(para.x)
    ooa.Setup = _setup
    return sum(abs.(g.x)) < threshold
end

function grad_rrs(ooa::OverlapwithOptimization, df)
    A = sum(df[!,ooa.gradientoverlap.name[1]])
    A_d = sum(df[!,ooa.gradientoverlap.name[2]])
    return [sum(A[:,s])./sum(A_d[:,s]) for s in 1:length(A[1,:])]
end
