"""
    local_interaction(::AbstractFockAddress, u)
    local_interaction(::AbstractFockAddress, ::AbstractFockAddress, v)

Return the sum of (mode-wise) local interactions ``\\frac{u}{2} \\sum_i n_i(n_i-1)`` of a
single component Fock state, or ``v \\sum_i n_{↑,i} n_{↓,i}`` between two Fock states. For a
multi-component Fock state, return the eigenvalue of

```math
\\frac{1}{2}\\sum_{i, σ, τ} u_{σ,τ} a^†_{σ,i}a^†_{τ,i}a_{τ,i}a_{σ,i} ,
```

where `u::SMatrix` is a symmetric matrix of interaction constants, `i` is a mode index,
and `σ`, `τ` are component indices.

See also [`BoseFS`](@ref), [`FermiFS`](@ref), [`CompositeFS`](@ref).
"""
local_interaction(b::SingleComponentFockAddress, u) = u * bose_hubbard_interaction(b) / 2
local_interaction(f::FermiFS, _) = 0
function local_interaction(a::SingleComponentFockAddress, b::SingleComponentFockAddress, u)
    return u * dot(occupied_modes(a), occupied_modes(b))
end

"""
    nearest_neighbour_interaction(f::SingleComponentFockAddress, σ, geometry::CubicGrid, map::ModeMap)
    nearest_neighbour_interaction(f1::SingleComponentFockAddress, f2::SingleComponentFockAddress, 
        Δ, geometry::CubicGrid, map1::ModeMap)

Calculate the nearest neighbour interaction ``\\σ \\sum_{⟨i,j⟩} n_i n_j`` for a single component 
Fock state `f` or ``Δ \\sum_{⟨i,j⟩} n_{↑,i} n_{↓,j}`` between two Fock states.For a
multi-component Fock state, return the eigenvalue of

```math
\\frac{1}{2}\\sum_{⟨i,j⟩, σ, τ} Δ_{σ,τ} a^†_{σ,i}a^†_{τ,j}a_{τ,j}a_{σ,i} ,
```

where `Δ::SMatrix` is a symmetric matrix of interaction constants, `i, j` are the mode indices,
and `σ`, `τ` are component indices.

See also [`BoseFS`](@ref), [`FermiFS`](@ref), [`CompositeFS`](@ref).
"""
function nearest_neighbour_interaction(f::SingleComponentFockAddress, Δ, 
    geometry::CubicGrid{D,S,B}, map::ModeMap) where {D,S,B}
    N = length(map)
    ONR = onr(f)
    ext_result = 0
    for i in 1:N
        for j in 1:D
            occ_mode = map[i].mode
            if !B[j] && check_boundary(occ_mode, j, S)
                continue # skip the dimension if it is hard wall
            end
            neigh = neighbor_site(geometry, occ_mode, j)
            ext_result += ONR[occ_mode] * ONR[neigh]
        end
    end
    return Δ*ext_result
end
function nearest_neighbour_interaction(f1::SingleComponentFockAddress, f2::SingleComponentFockAddress, 
    Δ, geometry::CubicGrid{D,S,B}, map1::ModeMap
    ) where {D,S,B}
    N1 = length(map1)
    ONR1 = onr(f1)
    ONR2 = onr(f2)
    ext_result = 0
    for i in 1:N1
        for j in 1:D
            occ_mode1 = map1[i].mode
            if !B[j] && check_boundary(occ_mode1, j, S)
                continue # skip the dimension if it is hard wall
            end
            neigh = neighbor_site(geometry, occ_mode1, j)
            if neigh in occupied_modes(f2)
                ext_result += ONR1[occ_mode1] * ONR2[neigh]
            end
        end
    end
    return Δ*ext_result
    
end

function check_boundary(mode, dim, S)
    P = prod(S[1:dim])
    _, C = fldmod1(mode,P)
    if dim == 1
        return C == S[1]
    else
        return C >= P - prod(S[1:dim-1])-1 && C <= P
    end
end

function _interactions_(fs::CompositeFS, u, Δ, geometry::CubicGrid) 
    return _interactions(fs.components, u, Δ, geometry)
end

"""
    _interaction_col(a, bs::Tuple, us::Tuple, Δs::Tuple, geometry::CubicGrid)

Sum the local interactions of the Fock state `a` with all states in `bs` using the onsite 
and nearest neighbour interaction constants in `us` and `Δs`. This is used to compute 
all interactions in the column below the diagonal of the interaction matrix. 
"""
@inline _interaction_col(a, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::CubicGrid) = 0
@inline function _interaction_col(a, (b, bs...), (u, us...), (Δ, Δs...), g::CubicGrid)
    return local_interaction(a, b, u) + _interaction_col(a, bs, us, Δs, g) +
        nearest_neighbour_interaction(a, b, Δ, g, occupied_mode_map(a))
end
@inline _interaction_col(a, ::Tuple{}, ::Tuple{}, ::Nothing, ::CubicGrid) = 0
@inline function _interaction_col(a, (b, bs...), (u, us...), Δ::Nothing, g::CubicGrid)
    return local_interaction(a, b, u) + _interaction_col(a, bs, us, Δ, g)
end
@inline _interaction_col(a, ::Tuple{}, ::Nothing, ::Tuple{}, ::CubicGrid) = 0
@inline function _interaction_col(a, (b, bs...), u::Nothing, (Δ, Δs...), g::CubicGrid)
    return _interaction_col(a, bs, u, Δs, g) +
        nearest_neighbour_interaction(a, b, Δ, g, occupied_mode_map(a))
end
"""
    _interactions(addresses, onsite_int_matrix, nearest_neighbour_int_matrix, geometry)

Compute all pairwise interactions in a tuple of `addresses`. The `onsite_int_matrix` and 
`nearest_neighbour_int_matrix` sets the intraction strengths of the onsite interaction and 
the nearest neighbour interaction.

The code is equivalent to the following.

```julia
acc = 0.0
for (i, a) in enumerate(addresses)
    acc += local_interaction(a, onsite_int_matrix[i, i]) +
        nearest_neighbour_int_matrix(a, nearest_neighbour_int_matrix[i,i],
        occupied_mode_map(a), geometry)
    for (j, b) in enumerate(addresses[i+1:end])
        acc += local_interaction(a, b, onsite_int_matrix[i, j]) +
            nearest_neighbour_int_matrix(a, b, nearest_neighbour_int_matrix[i,j],
            occupied_mode_map(a), geometry)
    end
end
return acc
```

It is implemented recursively to ensure type stability.
"""
@inline _interactions(::Tuple{}, ::SMatrix{0,0},::SMatrix{0,0}, ::CubicGrid) = 0.0
@inline function _interactions((a, as...)::NTuple{N,AbstractFockAddress}, 
    m::SMatrix{N,N}, σ::SMatrix{N,N}, g::CubicGrid
) where {N}
    # Split the matrix into the column we need now, and the rest.
    (u, u_column...) = Tuple(m[:, 1])
    (Δ, Δ_column...) = Tuple(σ[:, 1])
    # Type-stable way to subset SMatrix:
    m_rest = SMatrix{N-1,N-1}(view(m, 2:N, 2:N))
    σ_rest = SMatrix{N-1,N-1}(view(σ, 2:N, 2:N))
    # Get the self-interaction first.
    self = local_interaction(a, u) + 
        nearest_neighbour_interaction(a, Δ, g, occupied_mode_map(a))
    # Get the interactions for the rest of the row.
    row = _interaction_col(a, as, u_column, Δ_column, g)
    # Get the interaction for the rest of the rows.
    return self + row + _interactions(as, m_rest, σ_rest, g)
end
@inline _interactions(::Tuple{}, ::Nothing,::SMatrix{0,0}, ::CubicGrid) = 0.0
@inline function _interactions((a, as...)::NTuple{N,AbstractFockAddress}, 
    m::Nothing, σ::SMatrix{N,N}, g::CubicGrid
) where {N}
    # Split the matrix into the column we need now, and the rest.
    (Δ, Δ_column...) = Tuple(σ[:, 1])
    # Type-stable way to subset SMatrix:
    σ_rest = SMatrix{N-1,N-1}(view(σ, 2:N, 2:N))
    # Get the self-interaction first.
    self = nearest_neighbour_interaction(a, Δ, g, occupied_mode_map(a))
    # Get the interactions for the rest of the row.
    row = _interaction_col(a, as, u_column, Δ_column, g)
    # Get the interaction for the rest of the rows.
    return self + row + _interactions(as, m, σ_rest, g)
end
@inline _interactions(::Tuple{}, ::SMatrix{0,0},::Nothing, ::CubicGrid) = 0.0
@inline function _interactions((a, as...)::NTuple{N,AbstractFockAddress}, 
    m::SMatrix{N,N}, σ::Nothing, g::CubicGrid
) where {N}
    # Split the matrix into the column we need now, and the rest.
    (u, u_column...) = Tuple(m[:, 1])
    # Type-stable way to subset SMatrix:
    m_rest = SMatrix{N-1,N-1}(view(m, 2:N, 2:N))
    # Get the self-interaction first.
    self = local_interaction(a, u) 
    # Get the interactions for the rest of the row.
    row = _interaction_col(a, as, u_column, σ, g)
    # Get the interaction for the rest of the rows.
    return self + row + _interactions(as, m_rest, σ, g)
end

"""
    external_potential(add::AbstractFockAddress, pot)

Calculate the value of a diagonal single particle operator (e.g. a trap potential) at
the address `add`.
```math
\\sum_{iσ} v_{iσ} n_{iσ}
```
The (precomputed) potential energy per particle at each mode passed as `pot` should be
a length `M` vector for a [`SingleComponentFockAddress`](@ref), or a `M×C` matrix for
a [`CompositeFS `](@ref), where `M` is the number of modes and `C` the number of
components.
"""
Base.@propagate_inbounds function external_potential(add::SingleComponentFockAddress, pot)
    pe = 0.0
    @boundscheck checkbounds(pot, 1:num_modes(add))
    for (n,i) in occupied_modes(add)
        pe += n * pot[i]
    end
    return pe
end

function external_potential(add::CompositeFS, pot::Matrix)
    pe = 0.0
    @boundscheck checkbounds(pot, 1:num_modes(add), 1:num_components(add))
    for (i,c) in enumerate(add.components)
        @inbounds pe += external_potential(c, @view pot[:,i])
    end
    return pe
end

###
### HubbardRealSpace
###
"""
    HubbardRealSpace(address; geometry=PeriodicBoundaries(M,), t=ones(C, D), u=ones(C, C), Δ=ones(C, C), v=zeros(C, D))

Hubbard model in real space. Supports single or multi-component Fock state
addresses (with `C` components) and various (rectangular) lattice geometries
in `D` dimensions.

```math
  \\hat{H} = -\\sum_{\\langle i,j\\rangle,σ} t_{iσ} a^†_{iσ} a_{jσ} +
  \\frac{1}{2}\\sum_{i,σ,σ'} u_{σσ'} n_{iσ} (n_{iσ'} - δ_{σ,σ'}) + 
  \\sum_{⟨i,j⟩,σ,σ'} Δ_{iσ,jσ'} n_{iσ} n_{jσ} +
  \\sum_{i,σ≠τ}u_{στ} n_{iσ} n_{iτ}
```

If `v` is nonzero then this calculates ``\\hat{H} + \\hat{V}`` by adding the
harmonic trapping potential
```math
    \\hat{V} = \\sum_{i,σ,d} v_{σd} x_{di}^2 n_{iσ}
```
where ``x_{di}`` is the distance of site ``i`` from the centre of the trap
along dimension ``d``.

## Address types

* [`BoseFS`](@ref): Single-component Bose-Hubbard model.
* [`FermiFS`](@ref): Single-component Fermi-Hubbard model.
* [`CompositeFS`](@ref): For multi-component models.

Note that a single component of fermions cannot interact with itself. A warning
is produced if `address`is incompatible with the interaction parameters `u` and `Δ`.

## Geometries

Implemented [`CubicGrid`](@ref)s for keyword `geometry`

* [`PeriodicBoundaries`](@ref)
* [`HardwallBoundaries`](@ref)
* [`LadderBoundaries`](@ref)

Default is `geometry=PeriodicBoundaries(M,)`, i.e. a one-dimensional lattice with the
number of sites `M` inferred from the number of modes in `address`.

## Other parameters

* `t`: the hopping strengths. Must be a matrix of length `C × D `. The `i`-th and `j`-th element of the
  matrix corresponds to the hopping strength of the `i`-th component and `j`-th direction.
* `u`: the on-site interaction parameters. Must be a symmetric matrix. `u[i, j]`
  corresponds to the interaction between the `i`-th and `j`-th component. `u[i, i]`
  corresponds to the interaction of a component with itself. Note that `u[i,i]` must
  be zero for fermionic components.
* `Δ`: the nearest neighbour interaction parameters. Must be a symmetric matrix. `Δ[i, j]`
  corresponds to the interaction between the `i`-th and `j`-th component. `Δ[i, i]`
  corresponds to the interaction of a component with itself.
  `Δ[i, j]` corresponds to the interaction between the `i`-th and `j`-th component.
* `v`: the trap potential strengths. Must be a matrix of size `C × D`. `v[i,j]` is
  the strength of the trap for component `i` in the `j`th dimension.
"""
struct HubbardRealSpace{
    TT,
    C, # components
    A<:AbstractFockAddress,
    G<:CubicGrid,
    D, # dimension
    # The following need to be type params.
    T<:SMatrix{C,D,TT},
    U<:Union{SMatrix{C,C,TT},Nothing},
    DELTA<:Union{SMatrix{C,C,TT},Nothing},
    V<:Union{SMatrix{C,D,TT},Nothing},
    P<:Union{Matrix{TT},Nothing}
} <: AbstractHamiltonian{TT}
    address::A
    t::T # hopping strengths
    u::U # interactions
    Δ::DELTA # nearest neighbour interactions
    v::V # trap strengths
    potential::P # potential energy of each component at each lattice site
    geometry::G
end

function HubbardRealSpace(
    address::AbstractFockAddress;
    geometry::CubicGrid=PeriodicBoundaries((num_modes(address),)),
    t=ones(num_components(address), num_dimensions(geometry)),
    u=ones(num_components(address), num_components(address)),
    Δ=ones(num_components(address), num_components(address)),
    v=zeros(num_components(address), num_dimensions(geometry))
)
    C = num_components(address)
    D = num_dimensions(geometry)
    S = size(geometry)

    # Sanity checks
    if prod(size(geometry)) ≠ num_modes(address)
        throw(ArgumentError("`geometry` does not have the correct number of sites"))
    elseif length(u) ≠ 1 && !issymmetric(u)
        throw(ArgumentError("`u` must be symmetric"))
    elseif length(u) ≠ C * C
        throw(ArgumentError("`u` must be a $C × $C matrix"))
    elseif length(Δ) ≠ 1 && !issymmetric(Δ)
        throw(ArgumentError("`u` must be symmetric"))
    elseif length(Δ) ≠ C * C
        throw(ArgumentError("`u` must be a $C × $C matrix"))
    elseif length(t) ≠ C*D
        throw(ArgumentError("`t` must be a $C × $D matrix"))
    elseif length(v) ≠ C * D
        throw(ArgumentError("`v` must be a $C × $D matrix"))
    elseif !(address isa SingleComponentFockAddress || address isa CompositeFS)
        throw(ArgumentError(
            "unsupported address type detected use `CompositeFS` or `<: SingleComponentFockAddress`"
        ))
    end
    warn_fermi_interaction(address, u)

    TT = eltype(t)==Int ? Float64 : eltype(t)
    t_mat = SMatrix{C,D,TT}(t)
    u_mat = iszero(u) ? nothing : SMatrix{C,C,TT}(u)
    Δ_mat = iszero(Δ) ? nothing : SMatrix{C,C,TT}(Δ)

    # Precompute the trap potential terms
    if iszero(v)
        v_mat = nothing
        pot_vec = nothing
    else
        v_mat = SMatrix{C,D,Float64}(v)
        ranges = Tuple(range(-fld(M,2); length=M) for M in S)
        x_sq = map(x -> Tuple(x).^2, CartesianIndices(ranges))
        pot_vec = zeros(prod(S), C) # or undef...
        for c in 1:C
            pot_vec[:,c] .= vec(map(x -> sum(v_mat[c,:] .* x), x_sq))
        end
    end

    return HubbardRealSpace{TT,C,typeof(address),typeof(geometry),D,typeof(t_mat),typeof(u_mat),typeof(Δ_mat),typeof(v_mat),typeof(pot_vec)}(
        address, t_mat, u_mat, Δ_mat, v_mat, pot_vec, geometry,
    )
end

"""
    warn_fermi_interaction(address, u)

Warn if interaction matrix `u` does not make sense for `address`.
"""
function warn_fermi_interaction(address::CompositeFS, u)
    C = num_components(address)
    for c in 1:C
        if address.components[c] isa FermiFS && u ≠ ones(C,C) && u[c,c] ≠ 0
            @warn "component $(c) is fermionic, but was given a self-interaction " *
                "strength of $(u[c,c])" maxlog=1
        end
    end
end
function warn_fermi_interaction(address::FermiFS, u)
    if u ≠ ones(1, 1) && u[1, 1] ≠ 0
        @warn "address is fermionic, but was given a self-interaction " *
            "strength of $(u[1,1])" maxlog=1
    end
end
warn_fermi_interaction(_, _) = nothing

LOStructure(::Type{<:HubbardRealSpace}) = IsHermitian()

function Base.show(io::IO, h::HubbardRealSpace{TT,C}) where {TT,C}
    io = IOContext(io, :compact => true)
    println(io, "HubbardRealSpace(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  t = ", TT.(h.t), ",")
    if isnothing(h.u)
        println(io, "  u = ", zeros(C,C), ",")
    else
        println(io, "  u = ", TT.(h.u), ",")
    end
    if isnothing(h.Δ)
        println(io, "  Δ = ", zeros(C,C), ",")
    else
        println(io, "  Δ = ", TT.(h.Δ), ",")
    end
    !isnothing(h.v) && println(io, "  v = ", TT.(h.v), ",")
    print(io, ")")
end

# Overload equality due to stored potential energy arrays.
Base.:(==)(H::HubbardRealSpace, G::HubbardRealSpace) = all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(h::HubbardRealSpace) = h.address

dimension(::HubbardRealSpace, address) = number_conserving_dimension(address)

function diagonal_element(h::HubbardRealSpace{TT}, address) where {TT}
    int = (isnothing(h.u) && isnothing(h.Δ))  ? 0.0 : _interactions_(address, h.u, h.Δ, h.geometry)
    pot = isnothing(h.v) ? 0.0 : external_potential(address, h.potential)
    return convert(TT,int + pot)
end
function diagonal_element(h::HubbardRealSpace{TT,1}, address) where {TT}
    int = isnothing(h.u) ? 0.0 : local_interaction(address, h.u[1])
    int_NN = isnothing(h.Δ) ? 0.0 : nearest_neighbour_interaction(address, h.Δ[1], 
        h.geometry, occupied_mode_map(address))
    pot = if isnothing(h.v)
            0.0
        else
            @boundscheck checkbounds(h.potential, 1:num_modes(address), 1)
            @inbounds external_potential(address, @view h.potential[:,1])
        end
    return convert(TT,int + pot + int_NN)
end

###
### Offdiagonals
###
# This may be an inefficient implementation, but it is not actually used anywhere in the
# main algorithm.
get_offdiagonal(h::HubbardRealSpace, add, i) = offdiagonals(h, add)[i]
num_offdiagonals(h::HubbardRealSpace, add) = length(offdiagonals(h, add))

"""
    HubbardRealSpaceCompOffdiagonals{TT,D,G,A} <: AbstractOffdiagonals{A,TT}

Offdiagonals for a single address component of D dimensional CubicGrid. Used with [`HubbardRealSpace`](@ref) model
with a single-component address, or a component of a [`CompositeFS`](@ref).
"""
struct HubbardRealSpaceCompOffdiagonals{TT,D,G,A} <: AbstractOffdiagonals{A,TT}
    geometry::G
    address::A
    t::SMatrix{1,D,TT}
    length::Int
end

function offdiagonals(h::HubbardRealSpace{TT}, comp, add) where {TT}
    D = num_dimensions(h.geometry)
    return HubbardRealSpaceCompOffdiagonals{TT, D, typeof(h.geometry), typeof(add)}(
        h.geometry, add, h.t[comp,:], num_occupied_modes(add) * D * 2
    )
end

Base.size(o::HubbardRealSpaceCompOffdiagonals) = (o.length,)

@inline function Base.getindex(o::HubbardRealSpaceCompOffdiagonals{TT,D}, chosen) where {TT,D}
    particle, neigh = fldmod1(chosen, 2 * D)
    src_index = find_occupied_mode(o.address, particle)
    n_neigh = neighbor_site(o.geometry, src_index.mode, neigh)

    if n_neigh == 0
        return o.address, 0.0
    else
        dst_index = find_mode(o.address, n_neigh)
        new_add, value = excitation(o.address, (dst_index,), (src_index,))
        if neigh > D
            return new_add, convert(TT,conj(-o.t[neigh - D] * value))
        else        
            return new_add, convert(TT,-o.t[neigh] * value)
        end
    end
end

# For simple models with one component.
offdiagonals(h::HubbardRealSpace{<:Any,1,A}, add::A) where {A} = offdiagonals(h, 1, add)

# Multi-component part
"""
    HubbardRealSpaceOffdiagonals{TT,A,T<:Tuple} <: AbstractOffdiagonals{A,TT}

Offdiagonals of a [`HubbardRealSpace`](@ref) model with a [`CompositeFS`](@ref) address.
"""
struct HubbardRealSpaceOffdiagonals{TT,A,T<:Tuple} <: AbstractOffdiagonals{A,TT}
    address::A
    parts::T
    length::Int
end

"""
    get_comp_offdiags(h::HubbardRealSpace, add)

Get offdiagonals of all components of address in a type-stable manner.
"""
@inline function get_comp_offdiags(h::HubbardRealSpace, address)
    return _get_comp_offdiags(address.components, h, Val(1))
end

# All steps of recursive function (should) get inlined, creating a type-stable tuple of
# offdiagonals.
@inline function _get_comp_offdiags((a,as...), h, ::Val{I}) where {I}
    return (offdiagonals(h, I, a), _get_comp_offdiags(as, h, Val(I+1))...)
end
@inline _get_comp_offdiags(::Tuple{}, h, ::Val) = ()

function offdiagonals(h::HubbardRealSpace{TT,C,A}, address::A) where {TT,C,A<:CompositeFS}
    parts = get_comp_offdiags(h, address)
    return HubbardRealSpaceOffdiagonals{TT,A,typeof(parts)}(address, parts, sum(length, parts))
end

Base.size(o::HubbardRealSpaceOffdiagonals) = (o.length,)

# Becomes type unstable without inline for lots of components. Recursive function is used
# because the type of the result of `o.parts[i]` can not be inferred.
@inline function Base.getindex(o::HubbardRealSpaceOffdiagonals{A}, chosen) where {A}
    return _getindex(o.parts, o.address, chosen, Val(1))
end
@inline function _getindex((p, ps...), address::A, chosen, comp::Val{I}) where {A,I}
    if chosen ≤ length(p)
        new_add, val = p[chosen]
        return BitStringAddresses.update_component(address, new_add, comp), val
    else
        chosen -= length(p)
        return _getindex(ps, address, chosen, Val(I + 1))
    end
end
