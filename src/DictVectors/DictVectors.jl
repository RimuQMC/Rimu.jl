"""
Module that provides concrete implementations of the [`AbstractDVec`](@ref) interface.

- [`DVec`](@ref): basic [`AbstractDVec`](@ref)
- [`PDVec`](@ref): parallel [`AbstractDVec`](@ref) with MPI and initiator support
- [`InitiatorDVec`](@ref): allows storing information about initiator status

See [`Interfaces`](@ref).
"""
module DictVectors
using Folds: Folds
using LinearAlgebra: LinearAlgebra, I, dot, ⋅, mul!, normalize!, rank
using Random: Random
using StaticArrays: SVector
using VectorInterface: VectorInterface, add, add!, inner, norm, scalartype,
    scale, scale!, zerovector, zerovector!, zerovector!!

import MPI

using ..Interfaces: Interfaces, AbstractDVec, AdjointUnknown,
    CompressionStrategy, IsDiagonal, LOStructure,
    apply_column!, apply_operator!, compress!,
    diagonal_element, offdiagonals, step_stats, AbstractHamiltonian, AbstractOperator,
    AbstractObservable, dot_from_right, operator_column
using ..StochasticStyles: StochasticStyles, IsDeterministic
using Statistics: Statistics, mean, std
import ..Interfaces: deposit!, storage, StochasticStyle, default_style, freeze, localpart,
    working_memory, sum_mutating!

export deposit!, storage, walkernumber, walkernumber_and_length, dot_from_right
export DVec, InitiatorDVec, PDVec

export InitiatorRule, Initiator, SimpleInitiator, NonInitiator, CoherentInitiator

export AbstractProjector, NormProjector, Norm2Projector, UniformProjector, Norm1ProjectorPPop
export LoadBalancedCommunicator, LoadBalancer


# The idea is to do linear algebra with data structures that are not
# subtyped to AbstractVector, much in the spirit of VectorInterface.jl.
# In particular we provide concrete data structures with the aim of being
# suitable for use with KrylovKit.

include("delegate.jl")
include("abstractdvec.jl")
include("projectors.jl")

include("initiators.jl")
include("communicators.jl")

include("dvec.jl")
include("initiatordvec.jl")

include("pdvec.jl")
include("pdworkingmemory.jl")
include("LoadBalancer.jl")

end # module
