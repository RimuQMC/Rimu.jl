"""
    BitStringAddresses
Module with types and methods pertaining to bitstring addresses.
"""
module BitStringAddresses

using LinearAlgebra: LinearAlgebra, I, dot, ⋅
using Parameters: Parameters, @unpack
using Setfield: Setfield, @set, @set!, setindex
using SparseArrays: SparseArrays, SparseVector, nonzeros, rowvals, spzeros
using StaticArrays: StaticArrays, @MVector, FieldVector, MVector, SA, SVector

using Base.Cartesian

using ..Interfaces: Interfaces, AbstractFockAddress, num_particles, num_modes,
    num_components

export SingleComponentFockAddress, BoseFS, FermiFS, HardcoreBoseFS
export CompositeFS, FermiFS2C, time_reverse
export BoseFSIndex, FermiFSIndex
export BitString, SortedParticleList

export num_particles, num_modes, num_components
export find_occupied_mode, find_mode, occupied_modes, unoccupied_modes, each_mode
export is_occupied, num_occupied_modes, num_unoccupied_modes
export excitation, near_uniform, OccupiedPairsMap, occupied_mode_map, unoccupied_mode_map

export onr, occupation_number_representation
export hopnextneighbour, bose_hubbard_interaction
export @fs_str

include("fockaddress.jl")
include("bitstring.jl")
include("sortedparticlelist.jl")
include("bosefs.jl")
include("hardcorebosefs.jl")
include("fermifs.jl")
include("multicomponent.jl")
export OccupationNumberFS

"""
    OccupationNumberFS{M,T} <: SingleComponentFockAddress
Address type that stores the occupation numbers of a single component bosonic Fock state
with `M` modes. The occupation numbers must fit into the type `T <: Unsigned`. The number of
particles is runtime data, and can be retrieved with [`num_particles(address)`](@ref).

This is a deprecated type, and will be removed in a future release. The
type constructors currently return `BoseFS{missing}`. Use [`BoseFS{missing}`](@ref) instead.

# Constructors
- `OccupationNumberFS(val::Integer...)`: Construct from occupation numbers. Must be
  < 256 to fit into `UInt8`.
- `OccupationNumberFS(M, pairs::Pair...)`: Construct from a sparse representation with
  `M` modes and pairs of mode index and occupation number.
- `OccupationNumberFS{[M,T]}(onr)`: Construct from collection `onr` with `M` occupation
  numbers with optional type `T`.  `onr` may be a tuple, an array, or a generator.
- `OccupationNumberFS{M[,T]}()`: Construct a vacuum state with `M` modes. If `T` is
  unspecified, `UInt8` is used.
- `OccupationNumberFS(fs::BoseFS)`: Construct from [`BoseFS`](@ref).
- With short form macro [`@fs_str`](@ref). Specify the number of
  significant bits in braces. See example below.

See also: [`BoseFS`](@ref), [`HardcoreBoseFS`](@ref), [`FermiFS`](@ref),
[`SingleComponentFockAddress`](@ref), [`CompositeFS`](@ref), [`@fs_str`](@ref).
!!! warning
    The use of `OccupationNumberFS` is deprecated. Use [`BoseFS{missing}`](@ref) instead.
"""
struct OccupationNumberFS{M,T}
    OccupationNumberFS{M,T}() where {M,T} = BoseFS{missing,M}(; type=T)
end
function OccupationNumberFS(args...)
    Base.depwarn("OccupationNumberFS is deprecated, use BoseFS{missing} instead", :OccupationNumberFS)
    BoseFS{missing}(args...)
end
function OccupationNumberFS{M}(args...) where {M}
    Base.depwarn("OccupationNumberFS is deprecated, use BoseFS{missing} instead", :OccupationNumberFS)
    BoseFS{missing,M}(args...)
end
function OccupationNumberFS{M,U}(args...) where {M,U}
    Base.depwarn("OccupationNumberFS is deprecated, use BoseFS{missing} instead", :OccupationNumberFS)
    BoseFS{missing,M}(args...; type=U)
end

end
