"""
    Rimu
Random Integrator for Many-Body Quantum Systems
"""
module Rimu

using Reexport, Parameters, LinearAlgebra, DataFrames
import MPI, DataStructures

include("FastBufs.jl")
using .FastBufs
include("DictVectors/DictVectors.jl")
@reexport using .DictVectors
include("BitStringAddresses.jl")
@reexport using .BitStringAddresses
include("ConsistentRNG.jl")
@reexport using .ConsistentRNG
include("Hamiltonians.jl")
@reexport using .Hamiltonians
include("Blocking.jl")
@reexport using .Blocking

export lomc!
export fciqmc!, FciqmcRunStrategy, RunTillLastStep
export MemoryStrategy, NoMemory, DeltaMemory, ShiftMemory
export ProjectStrategy, NoProjection, NoProjectionTwoNorm, ThresholdProject, ScaledThresholdProject
export ShiftUpdateStrategy, LogUpdate, LogUpdateAfterTargetWalkers
export DontUpdate, DelayedLogUpdate, DelayedLogUpdateAfterTargetWalkers
export DoubleLogUpdate, DelayedDoubleLogUpdate, DoubleLogUpdateAfterTargetWalkers
export DelayedDoubleLogUpdateAfterTW
export DoubleLogUpdateAfterTargetWalkersSwitch
export HistoryLogUpdate
export ReportingStrategy, EveryTimeStep, EveryKthStep, ReportDFAndInfo
export TimeStepStrategy, ConstantTimeStep, OvershootControl
export StochasticStyle, IsStochastic, IsDeterministic
# export IsSemistochastic # is not yet ready
export IsStochasticNonlinear, IsStochasticWithThreshold
export threadedWorkingMemory, localpart

# exports for MPI functionality
export DistributeStrategy, MPIData, MPIDefault, MPIOSWin
export mpi_default, mpi_one_sided, fence, put, sbuffer, sbuffer!, targetrank
export free, mpi_no_exchange

include("strategies_and_params.jl")
include("helpers.jl")
include("mpi_helpers.jl")
include("fciqmc.jl")



export greet

"brief greeting"
greet() = print("Kia ora!")

end # module
