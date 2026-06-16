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
    return NamedTuple((ooa.name => overlaps,)), false
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
    return NamedTuple((cvo.name => overlaps,)), false
end

"""
    ParticleDensityGradientOverlap(op; testfunction,
        parameter=[SVector{1,Float64}(ones(Float64, 1))], 
        normalise::Bool=true) <: GlobalStepAction

Compute and report the particle density gradient overlaps ⟨ψ_i|∂O/∂α|ψ_j⟩, 
coefficient vector overlaps ⟨ψ_i|ψ_j⟩ between all pairs of replica states 
for a given operator `O` and its optimization parameters `α`. 
`optimizationparameter` is of type `Vector{SVector}` where each index refers 
to perticular spectral state.`testfunction` is a nothing or a function with 
parameters (̄α, # of sites) depending on whether the optimization is 
applied to entire Vector or it with the fixed functional form.
"""
mutable struct ParticleDensityGradientOverlap{OpType, normalize, P} <: GlobalStepAction
    operators::OpType
    testfunction::Union{Function, Nothing}
    optimizationparameter::P
end

function ParticleDensityGradientOverlap(op; testfunction=nothing, 
        optimizationparameter::Vector=[SVector{1,Float64}(ones(Float64, 1))],
        normalize::Bool=true)
    return ParticleDensityGradientOverlap{typeof(op), normalize, typeof(optimizationparameter)}(
        op, testfunction, optimizationparameter)
end

function (ooa::ParticleDensityGradientOverlap{<:Any, normalize})(
        state::ReplicaState) where {normalize}

    n_specs = num_spectral_states(state)
    n_reps = num_replicas(state)
    vectors = state_vectors(state) # 2D array: (replica, spectral state)
    M = num_modes(keytype(vectors[1,1]))
    gradient = zeros(eltype(ooa.optimizationparameter), binomial(n_reps, 2) , n_specs)
    coeff = zeros(eltype(ooa.optimizationparameter[1]), binomial(n_reps,2), n_specs)

    for s in 1:n_specs

        coeff[:, s] .= [dot(vectors[i, s], vectors[j, s]) for (i, j) 
                                    in StrictPairIter(n_reps)]
        
        if !iszero(sum(coeff[:,s]))
            test_vector, jacobian = isnothing(ooa.testfunction) ? 
                (ooa.optimizationparameter[s],nothing) : 
                ooa.testfunction(ooa.optimizationparameter[s], M)

            op = ooa.operators[1](test_vector; normalize)
            zeta = sum([dot_from_right(vectors[i, s], op, vectors[j, s]) / 
                sum(coeff[:,s]) for (i, j) in StrictPairIter(n_reps)])
                 
            if !iszero(zeta)
                G = ooa.operators[2](test_vector, jacobian; normalize, zeta)
                
                gradient[:,s] .= [dot_from_right(vectors[i, s], G, 
                    vectors[j, s]) for (i, j) in StrictPairIter(n_reps)]
            end
        end
    end
    return gradient, coeff, ooa.optimizationparameter
end

"""
    OptimizationAction(gradientaction; 
        method = RAdam(0.1), optimizationstep=100, threshold = 1e-3)) <: GlobalStepAction

Compute and report gradient of an observable `O` (⟨ψ_i|∂O/∂α|ψ_j⟩) at each reporting 
step and optimize the `optimizationparameters` of `gradientaction`(<:GlobalStepAction) 
after every `optimizationstep * reporting step` in the FCIQMC simulation between all 
pairs of replica states and spectral states. The gradient and `optimizationparameter`
(argument of `gradientaction`) are returned in a `NamedTuple` with a single field 
with the name `gradient` and `optimizationparameter` respectively.
    The optimization is carried out using the optimization `method` 
(default to RAdam(0.1)) which is downloaded from `Optimisers.jl`. There are 
other methods that can be used such as Adam and Momentum.
    FCIQMC simulation is optimization_state to be turminate when the sum of the absolute 
value of gradient become smaller then `threshold` (default to 1e-3). 
# Examples

```jldoctest
julia> using StaticArrays: SVector

julia> address = FermiFS(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
FermiFS{5,10}(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)

julia> h = HubbardRealSpace(address; w=-1.0)
HubbardRealSpace(
  fs"|↑↑↑↑↑⋅⋅⋅⋅⋅⟩";
  geometry = CubicGrid((10,), (true,)),
  t = [1.0;;],
  u = [1.0;;],
  w = [-1.0;;],
)

julia> parameter = [SVector{45,Float64}([1/45 for _ in 1:45])];

julia> gop = ParticleDensityGradientOverlap((TestTwoParticleDensity,
                   TestTwoParticleDensityGradient);
                   testfunction = nothing, parameter);

julia> oops = OptimizationAction(gop; optimizationstep = 5, threshold = 1e-2);

julia> p = ProjectorMonteCarloProblem(h; n_replicas=3, global_step_actions=(oops,));

julia> solve(p);
```
"""
mutable struct OptimizationAction{threshold,O,T} <: GlobalStepAction
    gradientaction::GlobalStepAction
    optimizationstate::O
    optimizationstep::Int
    gradientnumerator::T
    gradientdenominator::Vector{Float64}
end

function OptimizationAction(
    gradientaction; method = RAdam(0.1), optimizationstep=100, threshold = 1e-3
)

    optimizationstate = setup(method, (x = destructure(gradientaction.optimizationparameter)[1],))
    gradientnumerator = zero(gradientaction.optimizationparameter)
    gradientdenominator = zeros(Float64, length(gradientaction.optimizationparameter))
    return OptimizationAction{threshold, typeof(optimizationstate), typeof(gradientnumerator)}(
        gradientaction, optimizationstate, optimizationstep, gradientnumerator, gradientdenominator
    )
end

function (ooa::OptimizationAction{threshold})(state::ReplicaState) where threshold

    if state.step[] % (ooa.optimizationstep * state.reporting_strategy.reporting_interval) == 0
        gradient, coeff, parameter = (ooa.gradientaction)(state) 
        ooa.gradientnumerator += [sum(gradient[:,s]) for s in 1:length(gradient[1,:])]
        ooa.gradientdenominator += [sum(coeff[:,s]) for s in 1:length(coeff[1,:])]


        para = (x = destructure(ooa.gradientaction.optimizationparameter)[1],)
        grad = ooa.gradientnumerator ./ ooa.gradientdenominator
        ooa.gradientnumerator .= zero(ooa.gradientnumerator)
        ooa.gradientdenominator .= zero(ooa.gradientdenominator)
        v, re = destructure(grad)
        if isnan(sum(abs.(v))) # to ignore NaN as parameter
            return NamedTuple((:gradient => grad, 
                :optimizationparameter => ooa.gradientaction.parameter)), false
        end
        optimizationstate, para = update(ooa.optimizationstate, para, (x = -v,));
        
        # restructure (re) the parameters of each spectral state to a separate SVector. 
        ooa.gradientaction.optimizationparameter = re(para.x)
        ooa.optimizationstate = optimizationstate
        
        return  NamedTuple((:gradient => grad, 
            :optimizationparameter => ooa.gradientaction.optimizationparameter)),
            sum(abs.(v)) < threshold
    else
        gradient, coeff, parameter = (ooa.gradientaction)(state) 
        ooa.gradientnumerator .+= [sum(gradient[:,s]) for s in 1:length(gradient[1,:])]
        ooa.gradientdenominator += [sum(coeff[:,s]) for s in 1:length(coeff[1,:])]
        grad = ooa.gradientnumerator ./ ooa.gradientdenominator
        return NamedTuple((:gradient => grad, 
            :optimizationparameter => ooa.gradientaction.optimizationparameter)), false
    end
end
