"""
    index_apply(f, tuple, i, args...)

Return `f(tuple[i], args...)` in a type-stable manner when `tuple` is a heterogeneous tuple,
but `f` always returns a value of the same type.
"""
@inline function index_apply(f::F, tuple, i, args...) where {F}
    @boundscheck if i < 1 || i > length(tuple)
        throw(BoundsError(tuple, i))
    end
    return _index_apply(f, tuple, i, 1, args...)
end
@inline function _index_apply(f::F, (t, ts...), chosen, current, args...) where {F}
    if current == chosen
        return f(t, args...)
    end
    return _index_apply(f, ts, chosen, current + 1, args...)
end

"""
    local_interaction(::AbstractFockAddress, u, occ)
    local_interaction(::AbstractFockAddress, ::AbstractFockAddress, v, occ)

Return the sum of (mode-wise) local interactions ``\\frac{u}{2} \\sum_i n_i(n_i-1)`` of a
single component Fock state, or ``v \\sum_i n_{↑,i} n_{↓,i}`` between two Fock states. For a
multi-component Fock state, return the eigenvalue of

```math
\\frac{1}{2}\\sum_{i, σ, τ} u_{σ,τ} a^†_{σ,i}a^†_{τ,i}a^†_{τ,i}a^†_{σ,i} ,
```

where `u::SMatrix` is a symmetric matrix of interaction constants, `i` is a mode index,
and `σ`, `τ` are component indices.
`occ` is a [`ModeMap`](@ref) for single-component fock addresses or a tuple of
[`ModeMap`](@ref)s for composite addresses.

See also [`BoseFS`](@ref), [`FermiFS`](@ref), [`CompositeFS`](@ref).
"""
@inline function local_interaction(b::SingleComponentFockAddress, u, occs::Tuple)
    return local_interaction(b, u, only(occs))
end
@inline function local_interaction(b::SingleComponentFockAddress, u, occ::ModeMap)
    bh_interaction = sum(occ) do index
        index.occnum * (index.occnum - 1)
    end
    return bh_interaction * u[1] / 2
end
@inline local_interaction(f::FermiFS, _, ::Tuple) = 0
@inline local_interaction(f::FermiFS, _, ::ModeMap) = 0

@inline function local_interaction(
    a::SingleComponentFockAddress, b::SingleComponentFockAddress, u, occ_a, occ_b
)
    return u * dot(occ_a, occ_b)
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
function nearest_neighbour_interaction(onr1::SVector, onr2::SVector, 
    Δ, geometry::CubicGrid{D,S,B}, map1::ModeMap
    ) where {D,S,B}
    N1 = length(map1)
    ext_result = 0
    for i in 1:N1
        for j in 1:D
            occ_mode1 = map1[i].mode
            neigh = neighbor_site(geometry, occ_mode1, j)
            if !iszero(neigh)
                ext_result += onr1[occ_mode1] * onr2[neigh]
            end
        end
    end
    return Δ*ext_result
end

function _interactions_(fs::CompositeFS, u, Δ, occs, geometry::CubicGrid) 
    return _interactions(fs.components, u, Δ, occs, geometry)
end

"""
    _interaction_col(a, bs::Tuple, us::Tuple, Δs::Tuple, occ::ModeMap, occs::Tuple, geometry::CubicGrid)
    _interaction_col(a, bs::Tuple, us::Tuple, Δs::Nothing, occ::ModeMap, occs::Tuple, geometry::CubicGrid)
    _interaction_col(a, bs::Tuple, us::Nothing, Δs::Tuple, occ::ModeMap, occs::Tuple, geometry::CubicGrid)

Sum the local interactions of the Fock state `a` with all states in `bs` using the onsite 
and nearest neighbour interaction constants in `us` and `Δs`. This is used to compute 
all interactions in the column below the diagonal of the interaction matrix. 
"""
@inline _interaction_col(a, ::Tuple{}, ::Tuple{}, ::Tuple{},_,_, ::CubicGrid) = 0
@inline function _interaction_col(a, (b, bs...), (u, us...), (Δ, Δs...), occ_a, (occ_b, occs...), g::CubicGrid)
    return local_interaction(a, b, u, occ_a, occ_b) + _interaction_col(a, bs, us, Δs, occ_a, occs, g) +
        nearest_neighbour_interaction(onr(a), onr(b), Δ, g, occ_a)
end
@inline _interaction_col(a, ::Tuple{}, ::Tuple{}, ::Nothing,_,_, ::CubicGrid) = 0
@inline function _interaction_col(a, (b, bs...), (u, us...), Δ::Nothing, g::CubicGrid)
    return local_interaction(a, b, u, occ_a, occ_b) + _interaction_col(a, bs, us, Δs, occ_a, occs, g)
end
@inline _interaction_col(a, ::Tuple{}, ::Nothing, ::Tuple{},_,_, ::CubicGrid) = 0
@inline function _interaction_col(a, (b, bs...), u::Nothing, (Δ, Δs...), g::CubicGrid)
    return _interaction_col(a, bs, us, Δs, occ_a, occs, g) +
        nearest_neighbour_interaction(onr(a), onr(b), Δ, g, occ_a)
end
"""
    _interactions(addresses, onsite_int_matrix, nearest_neighbour_int_matrix, occs, geometry)

Compute all pairwise interactions in a tuple of `addresses`. The `onsite_int_matrix` and 
`nearest_neighbour_int_matrix` sets the intraction strengths of the onsite interaction and 
the nearest neighbour interaction. Moreover, `occs` holds the occupied modes of the adresses

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
@inline _interactions(::Tuple{}, ::SMatrix{0,0},::SMatrix{0,0}, _, ::CubicGrid) = 0.0
@inline function _interactions((a, as...)::NTuple{N,AbstractFockAddress}, 
    m::SMatrix{N,N}, σ::SMatrix{N,N}, (occ, occs...), g::CubicGrid
) where {N}
    # Split the matrix into the column we need now, and the rest.
    (u, u_column...) = isnothing(m) ? (nothing, nothing) : Tuple(m[:, 1])
    (Δ, Δ_column...) = isnothing(σ) ? (nothing, nothing) : Tuple(σ[:, 1])
    # Type-stable way to subset SMatrix:
    m_rest = isnothing(m) ? nothing : SMatrix{N-1,N-1}(view(m, 2:N, 2:N))
    σ_rest = isnothing(σ) ? nothing : SMatrix{N-1,N-1}(view(σ, 2:N, 2:N))
    # Get the self-interaction first.
    self = local_interaction(a, u, occ) + 
        nearest_neighbour_interaction(onr(a),onr(a), Δ, g, occ)
    # Get the interactions for the rest of the row.
    row = _interaction_col(a, as, u_column, Δ_column, occ, occs, g)
    # Get the interaction for the rest of the rows.
    return self + row + _interactions(as, m_rest, σ_rest, occs, g)
end

"""
    external_potential(address::AbstractFockAddress, pot, occ)

Calculate the value of a diagonal single particle operator (e.g. a trap potential) at
the address `address` whose occupied modes are stored in `occ`.
```math
\\sum_{iσ} v_{iσ} n_{iσ}
```
The (precomputed) potential energy per particle at each mode passed as `pot` should be
a length `M` vector for a [`SingleComponentFockAddress`](@ref), or a `M×C` matrix for
a [`CompositeFS `](@ref), where `M` is the number of modes and `C` the number of
components.
"""
@inline function external_potential(::SingleComponentFockAddress, potential, occ::ModeMap)
    return sum(occ) do index
        index.occnum * potential[index.mode]
    end
end
@inline function external_potential(addr::SingleComponentFockAddress, potential, occ::Tuple)
    return external_potential(addr, potential, only(occ))
end
@inline function external_potential(address::CompositeFS, potential, occs)
    return _external_potential(address.components, potential, occs, 1)
end
@inline function _external_potential(::Tuple{}, _, ::Tuple{}, _)
    return 0.0
end
@inline function _external_potential((a, as...), potential, (occ, occs...), i)
    pot = external_potential(a, view(potential, :, i), occ)
    return pot + _external_potential(as, potential, occs, i + 1)
end

# struct ================================================================================ #
"""
    HubbardRealSpace(address; geometry=PeriodicBoundaries(M,), t=ones(C, D), u=ones(C, C), Δ=zeros(C, C), v=zeros(C, D))

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
    Δ=zeros(num_components(address), num_components(address)),
    v=zeros(num_components(address), num_dimensions(geometry))
)
    C = num_components(address)
    D = num_dimensions(geometry)
    S = size(geometry)
    if t isa Vector && D != 1
        t_n = zeros(eltype(t), length(t), D)
        t_n .= t
    else
        t_n = t
    end
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
        throw(ArgumentError("`Δ` must be a $C × $C matrix"))
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
    t_mat = SMatrix{C,D,TT}(t_n)
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

# offdiaonals =========================================================================== #
# Holds the offdiagonals for a single-component nearest neighbour one-body term. It's
# structured like a matric where the first index determines the occupied site in the adress
# and the second index determines the site the particle will hop to.
struct HubbardRealSpaceComponentData{TT,D,I,G,A,C,O} <: AbstractMatrix{Pair{A,TT}}
    geometry::G
    parent_address::A
    address::C
    t::SVector{D,TT}
    occmap::O

    function HubbardRealSpaceComponentData{TT,D,I}(
        geometry::G,
        parent::A,
        address::C,
        t::SVector{D,TT},
        occmap::O=occupied_mode_map(address),
    ) where {TT,D,I,G,A,C,O}
        return new{TT,D,I,G,A,C,O}(geometry, parent, address, t, occmap)
    end
end

function Base.size(data::HubbardRealSpaceComponentData)
    return (length(data.occmap), 2 * num_dimensions(data.geometry))
end

component_index(::HubbardRealSpaceComponentData{<:Any,<:Any,I}) where {I} = I

function Base.getindex(data::HubbardRealSpaceComponentData{TT, D}, particle, direction) where {TT, D}
    @boundscheck if !(0 < particle ≤ size(data, 1)) || !(0 < direction ≤ size(data, 2))
        throw(BoundsError(data, (particle, direction)))
    end
    src = data.occmap[particle]
    neighbor = neighbor_site(data.geometry, src.mode, direction)
    if neighbor == 0
        return data.parent_address => 0.0
    else
        dst = find_mode(data.address, neighbor, data.occmap)
        new_add, val = excitation(data.address, (dst,), (src,))
        if data.parent_address isa CompositeFS
            new_parent = BitStringAddresses.update_component(
                data.parent_address, new_add, Val(component_index(data))
            )
        else
            new_parent = new_add
        end
        if direction > D
            return new_parent => convert(TT,conj(-data.t[direction - D] * val))
        else
            return new_parent => convert(TT, data.t[direction] * val)
        end
    end
end

# column ================================================================================= #
struct HubbardRealSpaceColumn{TT,H,G,A,C<:Tuple} <: AbstractOperatorColumn{A,TT,H}
    hamiltonian::H
    geometry::G
    address::A
    components::C
    num_offdiagonals::Int
end

parent_operator(column::HubbardRealSpaceColumn) = column.hamiltonian
starting_address(column::HubbardRealSpaceColumn) = column.address

function diagonal_element(col::HubbardRealSpaceColumn{TT}) where {TT}
    h = col.hamiltonian
    occmaps = map(c -> c.occmap, col.components)
    int = isnothing(h.u) && isnothing(h.Δ) ? 0.0 : _interactions_(col.address, h.u, h.Δ, occmaps, h.geometry)
    pot = isnothing(h.v) ? 0.0 : external_potential(col.address, h.potential, occmaps)

    return convert(TT, int + pot)
end

function operator_column(h::HubbardRealSpace{TT,<:Any,A,G}, address) where {TT,A,G}
    components = _column_components(h, address)
    return HubbardRealSpaceColumn{TT,typeof(h),G,A,typeof(components)}(
        h, h.geometry, address, components, sum(length, components)
    )
end

# Collect HubbardRealSpaceComponentData for each component of the address.
@inline function _column_components(h::HubbardRealSpace{TT}, address::SingleComponentFockAddress) where {TT}
    D = num_dimensions(h.geometry)
    return (HubbardRealSpaceComponentData{TT,D,1}(h.geometry, address, address, h.t[1,:]),)
end
@inline function _column_components(h::HubbardRealSpace, address::CompositeFS)
    return _column_components(h, address, address.components, Val(1))
end
@inline function _column_components(::HubbardRealSpace, _, ::Tuple{}, ::Val)
    return ()
end
@inline function _column_components(
    h::HubbardRealSpace{TT}, address, (a, as...), ::Val{I}
) where {TT,I}
    D = num_dimensions(h.geometry)
    data = HubbardRealSpaceComponentData{TT,D,I}(h.geometry, address, a, h.t[I,:])
    rest = _column_components(h, address, as, Val(I + 1))
    return (data, rest...)
end

function _split_index_component(column, i)
    components = column.components
    chosen_component = 0
    while i > 0
        chosen_component += 1
        i -= index_apply(length, components, chosen_component)
    end
    i += index_apply(length, components, chosen_component)
    return chosen_component, i
end

function random_offdiagonal(column::HubbardRealSpaceColumn)
    directions = 2 * num_dimensions(column.hamiltonian.geometry)
    random_number = rand(1:num_offdiagonals(column))
    component, remainder = _split_index_component(column, random_number)

    addr, val = index_apply(getindex, column.components, component, remainder)
    return addr, 1/column.num_offdiagonals, val
end

struct HubbardRealSpaceColumnOffdiagonals{TT,A,G,C<:Tuple} <: AbstractVector{Pair{A,TT}}
    address::A
    geometry::G
    components::C
    num_offdiagonals::Int
end

function offdiagonals(column::HubbardRealSpaceColumn{TT,<:Any,G,A,C}) where {TT,G,A,C}
    return HubbardRealSpaceColumnOffdiagonals{TT,A,G,C}(
        column.address,
        column.hamiltonian.geometry,
        column.components,
        column.num_offdiagonals,
    )
end
num_offdiagonals(column) = column.num_offdiagonals

@inline function Base.iterate(ods::HubbardRealSpaceColumnOffdiagonals, state=(1,1,1))
    i, j, k = state
    if k > 2 * num_dimensions(ods.geometry)
        k = 1
        j += 1
    end
    if j > index_apply(size, ods.components, i, 1)
        j = 1
        i += 1
    end
    if i > length(ods.components)
        return nothing
    else
        result = index_apply(getindex, ods.components, i, j, k)
        return result, (i, j, k + 1)
    end
end
Base.size(ods::HubbardRealSpaceColumnOffdiagonals) = (ods.num_offdiagonals,)
Base.eltype(::HubbardRealSpaceColumnOffdiagonals{TT,A}) where {TT,A} = Pair{A,TT}

function Base.getindex(column::HubbardRealSpaceColumnOffdiagonals, i)
    chosen, i = _split_index_component(column, i)
    return index_apply(getindex, column.components, chosen, i)
end
