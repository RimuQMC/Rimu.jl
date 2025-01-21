"""
    abstract type Geometry{D}

## Interface:

* `Base.size(::Geometry)`: return the size of the lattice.
* [`periodic_dimensions`](@ref)`(::Geometry)`: return a `D`-tuple of `Bool`s that signal which dimensions
  of the geometry are periodic.
* [`num_neighbors`](@ref)`(::Geometry)`: return the number of sites each site is connected
  to.
* [`neighbor_site`](@ref)`(::Geometry, site, chosen)`: pick a neighbor of site `site`.

See also [`CubicGrid`](@ref), [`HoneycombLattice`](@ref), and [`HexagonalLattice`](@ref) for
concrete implementations and [`HubbardRealSpace`](@ref) for a Hamiltonian that uses
[`Geometry`](@ref).

## Example

```julia
julia> geo = CubicGrid((2,3), (true,false))
CubicGrid{2}((2, 3), (true, false))

julia> geo[1]
(1, 1)

julia> geo[2]
(2, 1)

julia> geo[3]
(1, 2)

julia> geo[(1,2)]
3

julia> geo[(3,2)] # 3 is folded back into 1
3

julia> geo[(3,3)]
5

julia> geo[(3,4)] # returns 0 if out of bounds
0
```
"""
abstract type Geometry{D} end

Base.size(g::Geometry, i) = size(g)[i]
Base.length(g::Geometry) = prod(size(g))

"""
    num_neighbors(::Geometry)

Return the number of neighbors each site has in the geometry.
"""
num_neighbors

"""
    neighbor_site(::Geometry, site, i)

Find the `i`-th neighbor of `site` in the geometry. If the move is illegal, return 0.
"""
neighbor_site

"""
    periodic_dimensions(::Geometry{D})

Return a `D`-tuple of `Bool`s signaling which dimensions of the [`Geometry`](@ref) are
periodic.
"""
periodic_dimensions

"""
    num_dimensions(geom::Geometry)

Return the number of dimensions of the lattice in this geometry.
"""
num_dimensions(::Geometry{D}) where {D} = D

function Base.getindex(g::Geometry{D}, vec::Union{NTuple{D,Int},SVector{D,Int}}) where {D}
    return get(LinearIndices(size(g)), fold_vec(g, SVector(vec)), 0)
end
Base.getindex(g::Geometry, i::Int) = SVector(Tuple(CartesianIndices(size(g))[i]))

"""
    fold_vec(g::Geometry{D}, vec::SVector{D,Int}) -> SVector{D,Int}

Use the [`Geometry`](@ref) to fold the `vec` in each dimension. If folding is disabled in a
dimension, the vector is allowed to go out of bounds.

```julia
julia> geo = CubicGrid((2,3), (true,false))
CubicGrid{2}((2, 3), (true, false))

julia> fold_vec(geo, (3,1))
(1, 1)

julia> fold_vec(geo, (3,4))
(1, 4)
```
"""
function fold_vec(g::Geometry{D}, vec::SVector{D,Int}) where {D}
    (_fold_vec(Tuple(vec), periodic_dimensions(g), size(g)))
end
@inline _fold_vec(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline function _fold_vec((x, xs...), (f, fs...), (d, ds...))
    x = f ? mod1(x, d) : x
    return (x, _fold_vec(xs, fs, ds)...)
end

"""
    CubicGrid(dims::NTuple{D,Int}, fold::NTuple{D,Bool})

Represents a `D`-dimensional grid. Used to define a cubic lattice and boundary conditions
for some [`AbstractHamiltonian`](@ref)s, e.g. with the keyword argument `geometry` when
constructing a [`HubbardRealSpace`](@ref). The type instance can be used to convert between
cartesian vector indices (tuples or `SVector`s) and linear indices (integers). When indexed
with vectors, it folds them back into the grid if the out-of-bounds dimension is periodic and
0 otherwise (see example below).

* `dims` controls the size of the grid in each dimension.
* `fold` controls whether the boundaries in each dimension are periodic (or folded in the
  case of momentum space).

See also [`PeriodicBoundaries`](@ref), [`HardwallBoundaries`](@ref) and
[`LadderBoundaries`](@ref) for special-case constructors and [`HoneycombLattice`](@ref), and
[`HexagonalLattice`](@ref) for alternative lattice geometries.

See also [`HubbardRealSpace`](@ref) and [`G2RealSpace`](@ref).

A 3×2 `CubicGrid` is indexed as follows.
```
  │   │
─ 1 ─ 4 ─
  │   │
─ 2 ─ 5 ─
  │   │
─ 3 ─ 6 ─
  │   │
```
"""
struct CubicGrid{D,Dims,Fold} <: Geometry{D}
    function CubicGrid(
        dims::NTuple{D,Int}, fold::NTuple{D,Bool}=ntuple(Returns(true), Val(D))
    ) where {D}
        if any(≤(1), dims)
            throw(ArgumentError("All dimensions must be at least 2 in size"))
        end
        return new{D,dims,fold}()
    end
end
CubicGrid(args::Vararg{Int}) = CubicGrid(args)

"""
    PeriodicBoundaries(dims...) -> CubicGrid
    PeriodicBoundaries(dims) -> CubicGrid

Return a [`CubicGrid`](@ref) with all dimensions periodic. Equivalent to `CubicGrid(dims)`.
"""
function PeriodicBoundaries(dims::NTuple{D,Int}) where {D}
    return CubicGrid(dims, ntuple(Returns(true), Val(D)))
end
PeriodicBoundaries(dims::Vararg{Int}) = PeriodicBoundaries(dims)

"""
    HardwallBoundaries(dims...) -> CubicGrid
    HardwallBoundaries(dims) -> CubicGrid

Return a [`CubicGrid`](@ref) with all dimensions non-periodic. Equivalent to
`CubicGrid(dims, (false, false, ...))`.
"""
function HardwallBoundaries(dims::NTuple{D,Int}) where {D}
    return CubicGrid(dims, ntuple(Returns(false), Val(D)))
end
HardwallBoundaries(dims::Vararg{Int}) = HardwallBoundaries(dims)

"""
    LadderBoundaries(dims...) -> CubicGrid
    LadderBoundaries(dims) -> CubicGrid

Return a [`CubicGrid`](@ref) where the first dimension is dimensions non-periodic and the
rest are periodic. Equivalent to `CubicGrid(dims, (true, false, ...))`.
"""
function LadderBoundaries(dims::NTuple{D,Int}) where {D}
    return CubicGrid(dims, ntuple(i -> dims[i] > 2, Val(D)))
end
LadderBoundaries(dims::Vararg{Int}) = LadderBoundaries(dims)

function Base.show(io::IO, g::CubicGrid{<:Any,Dims,Fold}) where {Dims,Fold}
    print(io, "CubicGrid($Dims, $Fold)")
end

Base.size(g::CubicGrid{<:Any,Dims}) where {Dims} = Dims
periodic_dimensions(g::CubicGrid{<:Any,<:Any,Fold}) where {Fold} = Fold

"""
    Directions(D) <: AbstractVector{SVector{D,Int}}
    Directions(geometry::CubicGrid) <: AbstractVector{SVector{D,Int}}

Iterate over axis-aligned direction vectors in `D` dimensions.

```jldoctest; setup=:(using Rimu.Hamiltonians: Directions)
julia> Directions(3)
6-element Directions{3}:
 [1, 0, 0]
 [0, 1, 0]
 [0, 0, 1]
 [-1, 0, 0]
 [0, -1, 0]
 [0, 0, -1]

```

See also [`CubicGrid`](@ref).
"""
struct Directions{D} <: AbstractVector{SVector{D,Int}} end

Directions(D) = Directions{D}()
Directions(::Geometry{D}) where {D} = Directions{D}()

Base.size(::Directions{D}) where {D} = (2D,)

function Base.getindex(uv::Directions{D}, i) where {D}
    @boundscheck 0 < i ≤ length(uv) || throw(BoundsError(uv, i))
    if i ≤ D
        return SVector(_unit_vec(Val(D), i, 1))
    else
        return SVector(_unit_vec(Val(D), i - D, -1))
    end
end

@inline _unit_vec(::Val{0}, _, _) = ()
@inline function _unit_vec(::Val{I}, i, x) where {I}
    val = ifelse(i == I, x, 0)
    return (_unit_vec(Val(I-1), i, x)..., val)
end

"""
    Displacements(geometry::CubicGrid) <: AbstractVector{SVector{D,Int}}

Return all valid offset vectors in a [`CubicGrid`](@ref). If `center=true` the (0,0)
displacement is placed at the centre of the array.

```jldoctest; setup=:(using Rimu.Hamiltonians: Displacements)
julia> geometry = CubicGrid((3,4));

julia> reshape(Displacements(geometry), (3,4))
3×4 reshape(::Displacements{2, CubicGrid{2, (3, 4), (true, true)}}, 3, 4) with eltype StaticArraysCore.SVector{2, Int64}:
 [0, 0]  [0, 1]  [0, 2]  [0, 3]
 [1, 0]  [1, 1]  [1, 2]  [1, 3]
 [2, 0]  [2, 1]  [2, 2]  [2, 3]

julia> reshape(Displacements(geometry; center=true), (3,4))
3×4 reshape(::Displacements{2, CubicGrid{2, (3, 4), (true, true)}}, 3, 4) with eltype StaticArraysCore.SVector{2, Int64}:
 [-1, -1]  [-1, 0]  [-1, 1]  [-1, 2]
 [0, -1]   [0, 0]   [0, 1]   [0, 2]
 [1, -1]   [1, 0]   [1, 1]   [1, 2]

```
"""
struct Displacements{D,G<:CubicGrid{D}} <: AbstractVector{SVector{D,Int}}
    geometry::G
    center::Bool
end
Displacements(geometry; center=false) = Displacements(geometry, center)

Base.size(off::Displacements) = (length(off.geometry),)

@inline function Base.getindex(off::Displacements{D}, i) where {D}
    @boundscheck 0 < i ≤ length(off) || throw(BoundsError(off, i))
    geo = off.geometry
    vec = geo[i]
    if !off.center
        return vec - ones(SVector{D,Int})
    else
        return vec - SVector(ntuple(i -> cld(size(geo, i), 2), Val(D)))
    end
end

function neighbor_site(g::CubicGrid{D}, mode, chosen) where {D}
    # TODO: reintroduce LadderBoundaries small dimensions
    return g[g[mode] + Directions(D)[chosen]]
end

num_neighbors(::CubicGrid{D}) where {D} = 2D

function BitStringAddresses.onr(address, geom::CubicGrid{<:Any,S}) where {S}
    return SArray{Tuple{S...}}(onr(address))
end
function BitStringAddresses.onr(address::CompositeFS, geom::CubicGrid)
    return map(fs -> onr(fs, geom), address.components)
end

"""
    HoneycombLattice((height, width), fold=(true, true))

A honeycomb lattice where each site has three neighbors. If periodic, each dimension of the
lattice must be divisible by 2.

A 4×4 `HoneycombLattice` is indexed as follows.
```
  ╲    ╱    ╲    ╱
   1──5      9─13
  ╱    ╲    ╱    ╲
─2      6─10     14─
  ╲    ╱    ╲    ╱
   3──7     11─15
  ╱    ╲    ╱    ╲
─4      8─12     16─
  ╲    ╱    ╲    ╱
```
"""
struct HoneycombLattice{Dims,Fold} <: Geometry{2}
    function HoneycombLattice(dims::Tuple{Int,Int}, fold::Tuple{Bool,Bool}=(true, true))
        if any(<(2), dims)
            throw(ArgumentError("All dimensions must be at least 2 in size"))
        end
        if fold[1] && isodd(dims[1]) || fold[2] && isodd(dims[2])
            throw(ArgumentError("Periodic dimensions must be even in size"))
        end
        return new{dims,fold}()
    end
end
HoneycombLattice(h, w, fold::Tuple{Bool,Bool}=(true, true)) = HoneycombLattice((h, w), fold)

function Base.show(io::IO, geom::HoneycombLattice)
    h, w = size(geom)
    fold = periodic_dimensions(geom)
    print(io, "HoneycombLattice($h, $w, $fold)")
end

Base.size(::HoneycombLattice{Dims}) where {Dims} = Dims
periodic_dimensions(::HoneycombLattice{<:Any,Fold}) where {Fold} = Fold

function neighbor_site(geom::HoneycombLattice, mode, chosen)
    i, j = geom[mode]
    if chosen ≤ 2
        target = SVector(i + ifelse(chosen == 1, 1, -1), j)
    else
        target = SVector(i, j + ifelse(isodd(i + j), -1, 1))
    end
    source = geom[mode]
    return geom[target]
end

num_neighbors(::HoneycombLattice) = 3

"""
    HexagonalLattice((height, width), fold=(true, true))

A hexagonal lattice where each site has 6 neighbors.

A 3×2 `HexagonalLattice` is indexed as follows.
```
╲ │ ╲ │ ╲
─ 1 ─ 4 ─
╲ │ ╲ │ ╲
─ 2 ─ 5 ─
╲ │ ╲ │ ╲
─ 3 ─ 6 ─
╲ │ ╲ │ ╲
```
"""
struct HexagonalLattice{Dims,Fold} <: Geometry{2}
    function HexagonalLattice(dims::Tuple{Int,Int}, fold::Tuple{Bool,Bool}=(true,true))
        if any(<(2), dims)
            throw(ArgumentError("All dimensions must be at least 2 in size"))
        end
        return new{dims,fold}()
    end
end
HexagonalLattice(h, w, fold::Tuple{Bool,Bool}=(true, true)) = HexagonalLattice((h, w), fold)

function Base.show(io::IO, geom::HexagonalLattice)
    h, w = size(geom)
    fold = periodic_dimensions(geom)
    print(io, "HexagonalLattice($h, $w, $fold)")
end


Base.size(::HexagonalLattice{Dims}) where {Dims} = Dims
periodic_dimensions(::HexagonalLattice{<:Any,Fold}) where {Fold} = Fold

function neighbor_site(geom::HexagonalLattice, mode, chosen)
    if chosen ≤ 4
        # same as CubicGrid{2}
        offset = Directions(2)[chosen]
    else
        offset = SVector(1, 1) * ifelse(chosen == 5, 1, -1)
    end
    return geom[geom[mode] + offset]
end

num_neighbors(::HexagonalLattice) = 6
