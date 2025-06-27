```
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
```
module BitStringAddresses

using LinearAlgebra: LinearAlgebra, I, dot, ⋅
using Parameters: Parameters, @unpack
using Setfield: Setfield, @set, @set!, setindex
using SparseArrays: SparseArrays, SparseVector, nonzeros, rowvals, spzeros
using StaticArrays: StaticArrays, @MVector, FieldVector, MVector, SA, SVector

using Base.Cartesian

export AbstractFockAddress, SingleComponentFockAddress, BoseFS, BoseFS2C, FermiFS
export CompositeFS, FermiFS2C, time_reverse
export OccupationNumberFS
export BoseFSIndex, FermiFSIndex
export BitString, SortedParticleList
export num_particles, num_modes, num_components
export find_occupied_mode, find_mode, occupied_modes, unoccupied_modes, is_occupied, num_occupied_modes, num_unoccupied_modes
export excitation, near_uniform, ModeMap, OccupiedPairsMap, occupied_mode_map, unoccupied_mode_map
export onr, occupation_number_representation
export hopnextneighbour, bose_hubbard_interaction
export @fs_str

include("fockaddress.jl")
include("bitstring.jl")
include("sortedparticlelist.jl")
include("bosefs.jl")
include("fermifs.jl")
include("multicomponent.jl")
include("occupationnumberfs.jl")

end
