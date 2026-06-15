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
        :coefficient_vector_overlaps, :parameters), testfunction,
        parameter=[SVector{1,Float64}(ones(Float64, 1))], 
        normalise::Bool=true) <: GlobalStepAction

Compute and report the particle density gradient overlaps ⟨ψ_i|∂O/∂α|ψ_j⟩  
and coefficient vector overlaps ⟨ψ_i|ψ_j⟩ between all pairs of replica 
states for a given operator `O` and its parameters `α`. `parameter` is of 
type `Vector{SVector}` where each index refers to perticular spectral state.
`testfunction` is a nothing or a function with parameters (̄α, # of sites) 
depending on whether the optimization is applied to entire Vector or it 
with the fixed functional form.
The results and `parameter` are returned in a `NamedTuple` with a single 
field with key `name` (default `(:gradient_vector_overlaps, 
:coefficient_vector_overlaps, :parameters)`) and value array of overlaps.

"""
mutable struct ParticleDensityGradientOverlap{OpType, normalize, P} <: GlobalStepAction
    operators::OpType
    name::Tuple{Symbol, Symbol, Symbol}
    testfunction::Union{Function, Nothing}
    parameter::P
end

function ParticleDensityGradientOverlap(op; name = (:gradient_vector_overlaps, 
        :coefficient_vector_overlaps, :parameters), testfunction=nothing, 
        parameter::Vector=[SVector{1,Float64}(ones(Float64, 1))],
        normalize::Bool=true)
    return ParticleDensityGradientOverlap{typeof(op), normalize, typeof(parameter)}(
        op, name, testfunction, parameter)
end

function (ooa::ParticleDensityGradientOverlap{<:Any, normalize})(
        state::ReplicaState) where {normalize}

    n_specs = num_spectral_states(state)
    n_reps = num_replicas(state)
    vectors = state_vectors(state) # 2D array: (replica, spectral state)
    M = num_modes(keytype(vectors[1,1]))
    gradient = zeros(eltype(ooa.parameter), binomial(n_reps, 2) , n_specs)
    coeff = zeros(eltype(ooa.parameter[1]), binomial(n_reps,2), n_specs)

    for s in 1:n_specs

        coeff[:, s] .= [dot(vectors[i, s], vectors[j, s]) for (i, j) 
                                    in StrictPairIter(n_reps)]
        
        if !iszero(sum(coeff[:,s]))
            test_vector, jacobian = isnothing(ooa.testfunction) ? 
                (ooa.parameter[s],nothing) : 
                ooa.testfunction(ooa.parameter[s], M)

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
    return NamedTuple((ooa.name[1] => gradient, ooa.name[2] => coeff, 
            ooa.name[3] => ooa.parameter))
end

"""
    OverlapwithOptimization(gradientoverlap; 
        method = RAdam(0.1), step=100, threshold = 1e-3)) <: GlobalStepAction

Compute and report the particle density gradient overlaps ⟨ψ_i|∂O/∂α|ψ_j⟩ and
optimize the parameters of `gradientoverlap` (<:GlobalStepAction) after every
`step` collected gradient data points in the FCIQMC simulation between all
pairs of replica states for a given operator `O`. The results returned are
those provided by `gradientoverlap`.
    The optimization is carried out using the optimization `method`
(defaults to `RAdam(0.1)`) from `Optimisers.jl`. Other methods can also be
used, such as Adam and Momentum.
    The FCIQMC simulation is set up to terminate when the sum of the absolute
value of the gradient becomes smaller than `threshold` (default: `1e-3`).

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
                   TestTwoParticleDensityGradient); name=(:gradient_test_overlaps,
                   :coefficient_vector_overlaps, :parameter), 
                   testfunction = nothing, parameter);

julia> oops = OverlapwithOptimization(gop; step = 5, threshold = 1e-2);

julia> p = ProjectorMonteCarloProblem(h; n_replicas=3, global_step_actions=(oops,));

julia> solve(p);
```
"""
mutable struct OverlapwithOptimization{threshold,T} <: GlobalStepAction
    gradientoverlap::GlobalStepAction
    Setup::T
    Step::Int
end

function OverlapwithOptimization(gradientoverlap; 
        method = RAdam(0.1), step=100, threshold = 1e-3)
    _setup = setup(method, (x = destructure(gradientoverlap.parameter)[1],))
    return OverlapwithOptimization{threshold, typeof(_setup)}(gradientoverlap,
        _setup, step)
end

function (ooa::OverlapwithOptimization)(state::ReplicaState) 
    return (ooa.gradientoverlap)(state)
end

function (ooa::OverlapwithOptimization{threshold})(df::DataFrame) where threshold
    # destructure the parameters of each spectral state to a single Vector.
    para = (x = destructure(ooa.gradientoverlap.parameter)[1],)
    v, re = destructure(grad_rrs(ooa, df))
    if isnan(sum(abs.(v))) # to ignore NaN as parameter
        return false
    end
    _setup, para = update(ooa.Setup, para, (x = -v,));
    
    # restructure (re) the parameters of each spectral state to a separate SVector. 
    ooa.gradientoverlap.parameter = re(para.x)
    ooa.Setup = _setup
    return sum(abs.(v)) < threshold
end

@inline function grad_rrs(ooa::OverlapwithOptimization, df)
    A = sum(df[!,ooa.gradientoverlap.name[1]])
    A_d = sum(df[!,ooa.gradientoverlap.name[2]])
    return [sum(A[:,s])./sum(A_d[:,s]) for s in 1:length(A[1,:])]
end
