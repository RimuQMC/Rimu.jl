"""
    module Interfaces

This module contains interfaces that can be used to extend and modify the algorithms and
behaviours of `Rimu`.

# Interfaces
Follow the links for the definitions of the interfaces!
* [`AbstractHamiltonian`](@ref) for defining [`Hamiltonians`](@ref Main.Hamiltonians)
* [`AbstractOperator`](@ref) for defining observable operators
* [`AbstractObservable`](@ref) for defining observables
* [`AbstractOperatorColumn`](@ref) for defining operator columns
* [`AbstractDVec`](@ref) for defining data structures for `Rimu` as in [
    `DictVectors`](@ref Main.DictVectors)
* [`StochasticStyle`](@ref) for controlling the stochastic algorithms used by
    [`ProjectorMonteCarloProblem`](@ref Main.ProjectorMonteCarloProblem) as implemented in
    [`StochasticStyles`](@ref Main.StochasticStyles)
* [`AbstractFockAddress`](@ref) for defining Fock states, see also
    [`BitStringAddresses`](@ref Main.BitStringAddresses).

# Additional exports

## Interface functions for[`AbstractHamiltonian`](@ref)s:
* [`starting_address`](@ref)
* [`operator_column`](@ref)
* [`parent_operator`](@ref)
* [`diagonal_element`](@ref)
* [`random_offdiagonal`](@ref)
* [`offdiagonals`](@ref).
* [`num_offdiagonals`](@ref)
* [`get_offdiagonal`](@ref)
* [`LOStructure`](@ref)
* [`allows_address_type`](@ref)
* [`undo_transform`](@ref)
* [`has_random_offdiagonal`](@ref)
* [`has_iterable_offdiagonals`](@ref)
* [`maximum_mode_occupation`](@ref)

## working with  [`AbstractDVec`](@ref)s and [`StochasticStyle`](@ref)
* [`deposit!`](@ref)
* [`default_style`](@ref)
* [`CompressionStrategy`](@ref)
* The interface from [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl).

## Functions Rimu.jl uses to do FCIQMC:

* [`apply_column!`](@ref)
* [`apply_operator!`](@ref)
* [`step_stats`](@ref)

## Functions for retrieving information from DataFrames:

* [`num_replicas`](@ref)
* [`num_spectral_states`](@ref)
* [`num_overlaps`](@ref)

## Functions for working with [`AbstractFockAddress`](@ref)s:
* [`num_particles`](@ref)
* [`num_modes`](@ref)
* [`num_components`](@ref)
* [`maximum_mode_occupation`](@ref)
"""
module Interfaces

using LinearAlgebra: LinearAlgebra, diag
using VectorInterface: VectorInterface, add, add!, zerovector!, scalartype
using DataFrames: DataFrame, metadata

import OrderedCollections: freeze

export
    AbstractFockAddress, num_particles, num_modes, num_components, num_modes_check_equal,
    maximum_mode_occupation
export
    StochasticStyle, default_style, StyleUnknown, apply_column!, step_stats,
    CompressionStrategy, NoCompression, compress!
export
    AbstractDVec, deposit!, storage, localpart, freeze, working_memory,
    apply_operator!, sort_into_targets!, sum_mutating!
export
    AbstractHamiltonian, diagonal_element, num_offdiagonals, get_offdiagonal, offdiagonals,
    random_offdiagonal, starting_address, allows_address_type, undo_transform,
    LOStructure, IsDiagonal, IsHermitian, AdjointKnown, AdjointUnknown, has_adjoint,
    AbstractOperator, AbstractObservable, operator_column, OffdiagonalsOperatorColumn,
    AbstractOperatorColumn, parent_operator,
    has_random_offdiagonal, has_iterable_offdiagonals
export
    num_replicas, num_spectral_states, num_overlaps

include("abstractfockaddress.jl")
include("stochasticstyles.jl")
include("hamiltonians.jl")
include("dictvectors.jl")
include("dataframes.jl")

end
