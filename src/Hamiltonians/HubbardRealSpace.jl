"""
    index_apply(f, tuple, i, args...)

Return `f(tuple[i], args...)` in a type-stable manner when `tuple` is a heterogeneous tuple,
but `f` always returns a value of the same type.
"""
function index_apply(f::F, tuple, i, args...) where {F}
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
    local_interaction(::AbstractFockAddress, u)
    local_interaction(::AbstractFockAddress, ::AbstractFockAddress, v)

Return the sum of (mode-wise) local interactions ``\\frac{u}{2} \\sum_i n_i(n_i-1)`` of a
single component Fock state, or ``v \\sum_i n_{↑,i} n_{↓,i}`` between two Fock states. For a
multi-component Fock state, return the eigenvalue of

```math
\\frac{1}{2}\\sum_{i, σ, τ} u_{σ,τ} a^†_{σ,i}a^†_{τ,i}a^†_{τ,i}a^†_{σ,i} ,
```

where `u::SMatrix` is a symmetric matrix of interaction constants, `i` is a mode index,
and `σ`, `τ` are component indices.

See also [`BoseFS`](@ref), [`FermiFS`](@ref), [`CompositeFS`](@ref).
"""
function local_interaction(b::SingleComponentFockAddress, u, occs::Tuple)
    return local_interaction(b, u, only(occs))
end
function local_interaction(b::SingleComponentFockAddress, u, occ::ModeMap)
    bh_interaction = sum(occ) do index
        index.occnum * (index.occnum - 1)
    end
    return bh_interaction * u[1] / 2
end
local_interaction(f::FermiFS, _, ::Tuple) = 0
local_interaction(f::FermiFS, _, ::ModeMap) = 0

function local_interaction(
    a::SingleComponentFockAddress, b::SingleComponentFockAddress, u, occ_a, occ_b
)
    return u * dot(occ_a, occ_b)
end
function local_interaction(fs::CompositeFS, u, occs)
    return _interactions(fs.components, u, occs)
end

"""
    _interaction_col(a, bs::Tuple, us::Tuple)#TODO

Sum the local interactions of the Fock state `a` with all states in `bs` using the
interaction constants in `us`. This is used to compute all interactions in the column
below the diagonal of the interaction matrix.
"""
@inline _interaction_col(a, ::Tuple{}, ::Tuple{}, _, _) = 0
@inline function _interaction_col(a, (b, bs...), (u, us...), occ_a, (occ_b, occs...))
    return local_interaction(a, b, u, occ_a, occ_b) + _interaction_col(a, bs, us, occ_a, occs)
end

"""
    _interactions(addresses, interaction_matrix) #TODO

Compute all pairwise interactions in a tuple of `addresses`. The `interaction_matrix` sets the
intraction strengths.

The code is equivalent to the following.

```julia
acc = 0.0
for (i, a) in enumerate(addresses)
    acc += local_interaction(a, interaction_matrix[i, i])
    for (j, b) in enumerate(addresses[i+1:end])
        acc += local_interaction(a, b, interaction_matrix[i, j])
    end
end
return acc
```

It is implemented recursively to ensure type stability.
"""
@inline _interactions(::Tuple{}, ::SMatrix{0,0}, ::Tuple{}) = 0.0
@inline function _interactions(
    (a, as...)::NTuple{N,AbstractFockAddress}, m::SMatrix{N,N}, (occ, occs...)
) where {N}
    # Split the matrix into the column we need now, and the rest.
    (u, column...) = Tuple(m[:, 1])
    # Type-stable way to subset SMatrix:
    rest = SMatrix{N-1,N-1}(view(m, 2:N, 2:N))

    # Get the self-interaction first.
    self = local_interaction(a, u, occ)
    # Get the interactions for the rest of the row.
    row = _interaction_col(a, as, column, occ, occs)
    # Get the interaction for the rest of the rows.
    return self + row + _interactions(as, rest, occs)
end

"""
    external_potential(add::AbstractFockAddress, pot) #TODO

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
function external_potential(::SingleComponentFockAddress, potential, occ::ModeMap)
    return sum(occ) do index
        index.occnum * potential[index.mode]
    end
end
function external_potential(addr::SingleComponentFockAddress, potential, occ::Tuple)
    return external_potential(addr, potential, only(occ))
end
function external_potential(address::CompositeFS, potential, occs)
    return _external_potential(address.components, potential, occs, 1)
end
@inline function _external_potential(::Tuple{}, _, ::Tuple{}, _)
    return 0.0
end
@inline function _external_potential((a, as...), potential, (occ, occs...), i)
    pot = external_potential(a, view(potential, :, i), occ)
    return pot + _external_potential(as, potential, occs, i + 1)
end

###
### HubbardRealSpace
###
"""
    HubbardRealSpace(address; geometry=PeriodicBoundaries(M,), t=ones(C), u=ones(C, C), v=zeros(C, D))

Hubbard model in real space. Supports single or multi-component Fock state
addresses (with `C` components) and various (rectangular) lattice geometries
in `D` dimensions.

```math
  \\hat{H} = -\\sum_{\\langle i,j\\rangle,σ} t_σ a^†_{iσ} a_{jσ} +
  \\frac{1}{2}\\sum_{i,σ} u_{σσ} n_{iσ} (n_{iσ} - 1) +
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
is produced if `address`is incompatible with the interaction parameters `u`.

## Geometries

Implemented [`CubicGrid`](@ref)s for keyword `geometry`

* [`PeriodicBoundaries`](@ref)
* [`HardwallBoundaries`](@ref)
* [`LadderBoundaries`](@ref)

Default is `geometry=PeriodicBoundaries(M,)`, i.e. a one-dimensional lattice with the
number of sites `M` inferred from the number of modes in `address`.

## Other parameters

* `t`: the hopping strengths. Must be a vector of length `C`. The `i`-th element of the
  vector corresponds to the hopping strength of the `i`-th component.
* `u`: the on-site interaction parameters. Must be a symmetric matrix. `u[i, j]`
  corresponds to the interaction between the `i`-th and `j`-th component. `u[i, i]`
  corresponds to the interaction of a component with itself. Note that `u[i,i]` must
  be zero for fermionic components.
* `v`: the trap potential strengths. Must be a matrix of size `C × D`. `v[i,j]` is
  the strength of the trap for component `i` in the `j`th dimension.
"""
struct HubbardRealSpace{
    C, # components
    A<:AbstractFockAddress,
    G<:CubicGrid,
    D, # dimension
    # The following need to be type params.
    T<:SVector{C,Float64},
    U<:Union{SMatrix{C,C,Float64},Nothing},
    V<:Union{SMatrix{C,D,Float64},Nothing},
    P<:Union{Matrix{Float64},Nothing}
} <: AbstractHamiltonian{Float64}
    address::A
    t::T # hopping strengths
    u::U # interactions
    v::V # trap strengths
    potential::P # potential energy of each component at each lattice site
    geometry::G
end

function HubbardRealSpace(
    address::AbstractFockAddress;
    geometry::CubicGrid=PeriodicBoundaries((num_modes(address),)),
    t=ones(num_components(address)),
    u=ones(num_components(address), num_components(address)),
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
    elseif size(t) ≠ (C,)
        throw(ArgumentError("`t` must be a vector of length $C"))
    elseif length(v) ≠ C * D
        throw(ArgumentError("`v` must be a $C × $D matrix"))
    elseif !(address isa SingleComponentFockAddress || address isa CompositeFS)
        throw(ArgumentError(
            "unsupported address type detected use `CompositeFS` or `<: SingleComponentFockAddress`"
        ))
    end
    warn_fermi_interaction(address, u)

    t_vec = SVector{C,Float64}(t)
    u_mat = iszero(u) ? nothing : SMatrix{C,C,Float64}(u)

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

    return HubbardRealSpace{C,typeof(address),typeof(geometry),D,typeof(t_vec),typeof(u_mat),typeof(v_mat),typeof(pot_vec)}(
        address, t_vec, u_mat, v_mat, pot_vec, geometry,
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

function Base.show(io::IO, h::HubbardRealSpace{C}) where C
    io = IOContext(io, :compact => true)
    println(io, "HubbardRealSpace(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  t = ", Float64.(h.t), ",")
    if isnothing(h.u)
        println(io, "  u = ", zeros(C,C), ",")
    else
        println(io, "  u = ", Float64.(h.u), ",")
    end
    !isnothing(h.v) && println(io, "  v = ", Float64.(h.v), ",")
    print(io, ")")
end

# Overload equality due to stored potential energy arrays.
Base.:(==)(H::HubbardRealSpace, G::HubbardRealSpace) = all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(h::HubbardRealSpace) = h.address

dimension(::HubbardRealSpace, address) = number_conserving_dimension(address)


# offdiagonals =========================================================================== #
struct HubbardRealSpaceComponentData{I,G,A,C,O,M} <: AbstractMatrix{Pair{A,Float64}}
    geometry::G
    parent_address::A
    address::C
    t::Float64
    occmap::O
    modemap::M

    function HubbardRealSpaceComponentData{I}(
        geometry::G,
        parent::A,
        address::C,
        t::Float64,
        occmap::O=occupied_mode_map(address),
        modemap::M=nothing,
    ) where {I,G,A,C,O,M}
        return new{I,G,A,C,O,M}(geometry, parent, address, t, occmap, modemap)
    end
end

function _collect_modemap(address::BoseFS{<:Any,<:Any,<:BitString})
    iter = each_mode(address)
    result = MVector{length(iter),eltype(iter)}(undef)
    @inbounds for (i, index) in enumerate(iter)
        result[i] = index
    end
    return Tuple(result)
end
_collect_modemap(_) = nothing

function attach_modemap(data::HubbardRealSpaceComponentData{I}) where {I}
    modemap = _collect_modemap(data.address)

    return HubbardRealSpaceComponentData{I}(
        data.geometry, data.parent_address, data.address, data.t, data.occmap, modemap
    )
end

function Base.size(data::HubbardRealSpaceComponentData)
    return (length(data.occmap), 2 * num_dimensions(data.geometry))
end

component_index(::HubbardRealSpaceComponentData{I}) where {I} = I

function Base.getindex(data::HubbardRealSpaceComponentData, particle, direction)
    src = data.occmap[particle]
    neighbor = neighbor_site(data.geometry, src.mode, direction)
    if neighbor == 0
        return data.parent_address => 0.0
    else
        if !isnothing(data.modemap)
            dst = data.modemap[neighbor]
        else
            dst = find_mode(data.address, neighbor)
        end
        new_add, val = excitation(data.address, (dst,), (src,))
        if data.parent_address isa CompositeFS
            new_parent = BitStringAddresses.update_component(
                data.parent_address, new_add, Val(component_index(data))
            )
        else
            new_parent = new_add
        end
        return new_parent => -data.t * val
    end
end

# column ===================================================================================
struct HubbardRealSpaceColumn{H,G,A,C<:Tuple} <: AbstractOperatorColumn{A,Float64,H}
    hamiltonian::H
    geometry::G
    address::A
    components::C
    num_offdiagonals::Int
end

parent_operator(column::HubbardRealSpaceColumn) = column.hamiltonian
starting_address(column::HubbardRealSpaceColumn) = column.address

function diagonal_element(col::HubbardRealSpaceColumn)
    h = col.hamiltonian
    occmaps = map(c -> c.occmap, col.components)
    int = isnothing(h.u) ? 0.0 : local_interaction(col.address, h.u, occmaps)
    pot = isnothing(h.v) ? 0.0 : external_potential(col.address, h.potential, occmaps)

    return int + pot
end

function operator_column(h::HubbardRealSpace, address)
    components = _column_components(h, address)
    return HubbardRealSpaceColumn(
        h, h.geometry, address, components, sum(length, components)
    )
end

@inline function _column_components(h::HubbardRealSpace, address::SingleComponentFockAddress)
    return (HubbardRealSpaceComponentData{1}(h.geometry, address, address, h.t[1]),)
end
@inline function _column_components(h::HubbardRealSpace, address::CompositeFS)
    return _column_components(h, address, address.components, Val(1))
end
@inline function _column_components(::HubbardRealSpace, _, ::Tuple{}, ::Val)
    return ()
end
@inline function _column_components(
    h::HubbardRealSpace, address, (a, as...), ::Val{I}
) where {I}
    data = HubbardRealSpaceComponentData{I}(h.geometry, address, a, h.t[I])
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

struct HubbardRealSpaceColumnOffdiagonals{A,G,C<:Tuple} <: AbstractVector{Pair{A,Float64}}
    address::A
    geometry::G
    components::C
    num_offdiagonals::Int
end

function offdiagonals(column::HubbardRealSpaceColumn)
    components = map(attach_modemap, column.components)

    return HubbardRealSpaceColumnOffdiagonals(
        column.address,
        column.hamiltonian.geometry,
        components,
        column.num_offdiagonals,
    )
end
num_offdiagonals(column) = column.num_offdiagonals

function Base.iterate(ods::HubbardRealSpaceColumnOffdiagonals, state=(1,1,1))
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
Base.eltype(::HubbardRealSpaceColumnOffdiagonals{A}) where {A} = Pair{A,Float64}

function Base.getindex(column::HubbardRealSpaceColumnOffdiagonals, i)
    chosen, i = _split_index_component(column, i)
    return index_apply(getindex, column.components, chosen, i)
end
