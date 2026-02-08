"""
    dispersion_mom_space(ks, geometry, t)

    Dispersion relation for a given set of k values, lattice geometry and hopping strengths t. Returns
        ``-2(\\sum_{\\bar{k}} \\Re(t_{\\bar{k}}) \\cos(k_{\\bar{k}}) + \\Im(t_{\\bar{k}}) \\sin(k_{\\bar{k}}))``
    where ``\\bar{k}`` runs over all dimensions of the lattice.
"""
function dispersion_mom_space(ks::SVector{D}, geometry::CubicGrid{D, S}, t::SMatrix) where {D, S}
    # Calculate the dispersion relation for a given set of k values and hopping strength t.
    C,_ = size(t)
    M = prod(S)
    kes_mat = zeros(Float64,C,M)
    ks_mat = zeros(Float64,D,M)
    phase = atan.(imag.(t)./real.(t))
    for j in 1:C
        for i in 1:M
            mom_val = value_of_mom_mode(M-i+1, ks, geometry)
            kes_mat[j,i] = convert(Float64,hub_dis_mom_space(t[j,:], mom_val)[1])
            ks_mat[:,i] = mom_val + phase[j,:]
        end
    end
    return SMatrix{C,M,Float64}(kes_mat), SMatrix{D,M,Float64}(ks_mat)
end
function value_of_mom_mode(add_index::Int, ks::SVector{D}, geometry::CubicGrid{D}) where {D}
    mom_mode = geometry[add_index]
    return [ks[i][mode] for (i, mode) in enumerate(mom_mode)]
end

function hub_dis_mom_space(t::SVector, k::Vector)
    # Calculate the dispersion relation for a given k value and hopping strength t.
    return -2 * (real.(t') * cos.(k) - imag.(t') * sin.(k))
end

"""
    _mom_hopping(kes, address)

    Calculate the hopping term for a single-component or multi-component Fock state address in momentum space.
"""
@inline function _mom_hopping(kes::SMatrix{C}, address::CompositeFS{C}) where {C}
    # Calculate the hopping term for a single component.
    comp = address.components
    onproduct = 0.0
    for i in 1:C
        onproduct += dot(kes[i,:], occupied_mode_map(comp[i]))
    end
    return onproduct
end

@inline _mom_hopping(kes::SMatrix{1}, address::SingleComponentFockAddress) = dot(kes[1,:], occupied_mode_map(address))

"""
    mom_transfer_mom_space(add, chosen, map, g; fold=true)
    mom_transfer_mom_space(add1, add2, chosen, map1, map2, g; fold=true)

Get the momentum transfer for a given excitation in a same or two different components of a
multi-component Fock state address in momentum space, i.e., `add` or between `add1` and `add2`. 
`map`, `map1` and `map2` are the occupied mode maps for the relevant components of the 
multi-component Fock state. `chosen` is an integer that determines which excitation and `g` is
a geometry of the lattice.

See also [`extended_mom_transfer_diag`](@ref). 
"""
@inline function mom_transfer_mom_space(
    add::SingleComponentFockAddress{<:Any, M}, chosen::Int, map::ModeMap,
    g::CubicGrid{D,S}; fold=true) where {M, D, S}
    # Get the momentum transfer for a given excitation.
    singlies = length(map) # number of at least singly occupied modes

    double = chosen - singlies * (singlies - 1) * (M - 2)

    if double > 0
        # Both moves from the same mode.
        double, mom_change = fldmod1(double, M - 1)
        idx = first(map) # placeholder
        for i in map
            double -= i.occnum ≥ 2
            if double == 0
                idx = i
                break
            end
        end
        src_indices = (idx, idx)
    else
        # Moves from different modes.
        pair, mom_change = fldmod1(chosen, M - 2)
        fst, snd = fldmod1(pair, singlies - 1) # where the holes are to be made
        if snd < fst # put them in ascending order
            f_hole = snd
            s_hole = fst
        else
            f_hole = fst
            s_hole = snd + 1 # as we are counting through all singlies
        end
        src_indices = (map[f_hole], map[s_hole])
    end
    src_modes = (src_indices[1].mode, src_indices[2].mode)
    src_loc = (g[src_modes[1]], g[src_modes[2]])
    Q = g[mom_change+1] - g[1]
    dst_loc = (src_loc[1]+Q, src_loc[2]-Q)
    if fold
        dst_loc = (mod1.(dst_loc[1], S) , mod1.(dst_loc[2], S))
        if dst_loc == src_loc || reverse(dst_loc) == src_loc
            # If the momentum transfer is out of bounds, we return the original address.
            Q = g[M] - g[1]
            dst_loc = (src_loc[1]+Q, src_loc[2]-Q)
            dst_loc = (mod1.(dst_loc[1], S) , mod1.(dst_loc[2], S))
        end
    elseif !(all(ones(Int, D) .≤ dst_loc[2] .≤ S) && all(ones(Int, D) .≤ dst_loc[2] .≤ S))
        Q .-= S
        dst_loc .= [SRC[1]+Q, SRC[2]-Q]
        if !(all(ones(Int, D) .≤ dst_loc[2] .≤ S) && all(ones(Int, D) .≤ dst_loc[2] .≤ S))
            return add, 0.0, src_modes..., -Q
        end
    end
    dst_indices = find_mode(add, (g[dst_loc[1]], g[dst_loc[2]]))
    return excitation(add, dst_indices, reverse(src_indices))..., src_modes..., -Q
end

@inline function mom_transfer_mom_space(
    add1::SingleComponentFockAddress{<:Any, M}, add2::SingleComponentFockAddress{<:Any, M}, 
    chosen::Int, map1::ModeMap, map2::ModeMap, g::CubicGrid{D,S}; fold=true) where {M, D, S}
    # Get the momentum transfer for a given excitation.
    singlies = length(map2)

    pair, mom_change = fldmod1(chosen, M - 1)

    f_hole, s_hole = fldmod1(pair, singlies) # where the holes are to be made
    src_indices = (map1[f_hole], map2[s_hole])
    src_modes = (src_indices[1].mode, src_indices[2].mode)
    src_loc = (g[src_modes[1]], g[src_modes[2]])
    Q = g[mom_change+1] - g[1]
    dst_loc = (src_loc[1]+Q, src_loc[2]-Q)

    if fold
        dst_loc = (mod1.(dst_loc[1], S) , mod1.(dst_loc[2], S))
    elseif !(all(x -> 1 ≤ x ≤ S, dst_loc[1]) && all(x -> 1 ≤ x ≤ S, dst_loc[2]))
        Q .-= S
        dst_loc .= [SRC[1]+Q, SRC[2]-Q]
        if !(all(x -> 1 ≤ x ≤ S, dst_loc[1]) && all(x -> 1 ≤ x ≤ S, dst_loc[2]))
            return add, 0.0, src_modes..., -Q
        end
    end
    return excitation(add1, find_mode(add1, (g[dst_loc[1]],)), (src_indices[1],))..., 
        excitation(add2, find_mode(add2, (g[dst_loc[2]],)), (src_indices[2],))..., src_modes..., -Q
end

"""
    extended_mom_transfer_diag(map, g, u, w)
    extended_mom_transfer_diag(map1, map2, g, u, w)

Calculate the extended momentum transfer diagonal between a given same component occupied mode map `map` or
between two different component occupied mode maps `map1` and `map2`. `g` is the geometry of the lattice.
`u` and `w` are the on-site and nearest neighbour interaction parameters respectively.

"""
@inline function extended_mom_transfer_diag(map::BoseOccupiedModeMap, g::CubicGrid{D,S}, u, w) where {D, S}

    onproduct = 0
    for i in 1:length(map)
        occ_i = map[i].occnum
        onproduct += occ_i * (occ_i - 1) * (u/2 + w*D)
        for j in 1:i-1
            occ_j = map[j].occnum
            q = g[map[i].mode] - g[map[j].mode]
            onproduct += 2 * occ_i * occ_j * (u + w * (D + _cosin_sum(q, S)))
        end
    end
    return onproduct
end

@inline function extended_mom_transfer_diag(map::BoseOccupiedModeMap, g::CubicGrid{D,S}, ::Nothing, w) where {D, S}
    
    onproduct = 0
    for i in 1:length(map)
        occ_i = map[i].occnum
        onproduct += occ_i * (occ_i - 1) * (w*D)
        for j in 1:i-1
            occ_j = map[j].occnum
            q = g[map[i].mode] - g[map[j].mode]
            onproduct += 2 * occ_i * occ_j * (D + _cosin_sum(q, S))
        end
    end
    return onproduct * w
end

@inline function extended_mom_transfer_diag(map::BoseOccupiedModeMap, ::CubicGrid, u, ::Nothing)

    onproduct = 0
    for i in 1:length(map)
        occ_i = map[i].occnum
        onproduct += occ_i * (occ_i - 1) / 2
        for j in 1:i-1
            occ_j = map[j].occnum
            onproduct += 2 * occ_i * occ_j
        end
    end
    return onproduct * u
end

@inline function extended_mom_transfer_diag(map::FermiOccupiedModeMap, g::CubicGrid{D,S}, _, w) where {D, S}

    onproduct = 0
    for i in 1:length(map)
        occ_i = map[i].occnum
        onproduct += occ_i * (occ_i - 1)
        for j in 1:i-1
            occ_j = map[j].occnum
            q = g[map[i].mode] - g[map[j].mode]
            onproduct += 2 * occ_i * occ_j * (D - _cosin_sum(q, S))
        end
    end
    return onproduct*w
end

@inline extended_mom_transfer_diag(::FermiOccupiedModeMap, ::CubicGrid, _, ::Nothing) = 0

@inline function extended_mom_transfer_diag(map1::ModeMap, map2::ModeMap, ::CubicGrid{D}, u, w) where D
    onproduct = 0
    for i in map1
        occ_i = i.occnum
        for j in map2
            occ_j = j.occnum
            onproduct += occ_i * occ_j
        end
    end
    return onproduct * _interaction_parameter_diag(u, w, D)
end

@inline function extended_mom_transfer_diag(map1::FermiOccupiedModeMap, map2::FermiOccupiedModeMap, 
    ::CubicGrid{D}, u, w) where D
    return length(map1) * length(map2) * _interaction_parameter_diag(u, w, D)
end

function _cosin_sum(q::SVector{D}, S::NTuple{D}) where {D}
    onproduct = 0.0
    for i in 1:D
        onproduct += cos(q[i] * 2π / S[i])
    end
    return onproduct
end

"""
    _mom_interactions_diag(component, g)
Calculate the interaction terms for a given component of a multi-component Fock state address in momentum space.
`component` is a tuple of all combination between the pair of relevant components of the multi-component Fock state 
for which the interaction term needs to be calculated,

'''math
    \\hat{H}_\\text{int} = \\frac{1}{2}\\sum_{p,q,σ,σ'} V_{σσ'} \\hat{b}^†_{pσ} \\hat{b}^†_{qσ'} \\hat{b}^†_{qσ'} \\hat{b}_{pσ}
'''

where `V_{σσ}' is the interaction coefficent that depends on  `u_{σσ'}' and `w_{σσ'}'. `g` is the geometry of the lattice.

"""

function _mom_interactions_diag(component::Tuple, g::CubicGrid)
    onproduct = 0
    for data in component
        if !(isnothing(data.u) && isnothing(data.w)) 
            if component_index(data)[3]
                # If the occupied modes are the same, we can use the extended mom transfer.
                onproduct += extended_mom_transfer_diag(data.occmap1, g, data.u, data.w)
            else
                # Otherwise we need to calculate the interaction between two different occupied modes.
                onproduct += extended_mom_transfer_diag(data.occmap1, data.occmap2, g, data.u, data.w)
            end
        end
    end
    return onproduct
end

@inline _interaction_parameter_diag(u::Float64, w::Float64, D::Int) = u + 2 * w * D
@inline _interaction_parameter_diag(::Nothing, w::Float64, D::Int) = 2 * w * D
@inline _interaction_parameter_diag(u::Float64, ::Nothing, _) = u

"""
    HubbardMomSpace(address; geometry=PeriodicBoundaries(M,), t=ones(C, D), u=ones(C, C), w=zeros(C, C))

Hubbard model in momentum space. Supports single or multi-component Fock state
addresses (with `C` components) and various (rectangular) lattice geometries
in `D` dimensions and of `M` volume.

```math
  \\hat{H} = -\\sum_{k,σ} ϵ_{kσ} n_{kσ} +
  \\sum_{p,q,k,σ,σ'} V_{σσ'} a^†_{p+k,σ} a^†_{q-k,σ'} a_{q,σ'} a_{p,σ}
```
where ``ϵ_{kσ} = -2 (\\sum_{d=1}^{D} \\Re(t_{σ,d}) \\cos(k_d) - \\Im(t_{σ,d}) \\sin(k_d))`` and
``V_{σσ'} = (u_{σσ'}(1- \\frac{δ_{σσ'}}{2}) + w_{σσ'} \\sum_{d=1}^{D} \\cos(q_d))/M``

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

* `t`: the hopping strengths. Must be a matrix of length `C × D `. The `i`-th and `j`-th element of the
  matrix corresponds to the hopping strength of the `i`-th component and `j`-th direction.
* `u`: the on-site interaction parameters. Must be a symmetric matrix. `u[i, j]`
  corresponds to the interaction between the `i`-th and `j`-th component. `u[i, i]`
  corresponds to the interaction of a component with itself. Note that `u[i,i]` mustadd
  be zero for fermionic components.
* `w`: the nearest neighbour interaction parameters. Must be a symmetric matrix.
  `w[i, j]` corresponds to the interaction between the `i`-th and `j`-th component.
"""
struct HubbardMomSpace{
    C, # components    
    D, # dimension
    A<:AbstractFockAddress,
    G<:CubicGrid,
    # The following need to be type params.
    KS<:SMatrix{D,<:Any, Float64}, # k values
    KES<:SMatrix{C,<:Any,Float64},
    T<:SMatrix{C,D,<:Any}, # hopping strengths
    U<:Union{SMatrix{C,C,Float64},Nothing},
    W<:Union{SMatrix{C,C,Float64},Nothing},
} <: AbstractHamiltonian{Float64}
    address::A
    ks::KS # k values
    kes::KES # kinetic energy values
    t::T
    u::U # interactions
    w::W # nearest neighbour interactions
    geometry::G
end

function HubbardMomSpace(
    address::AbstractFockAddress;
    geometry::CubicGrid=PeriodicBoundaries((num_modes(address),)),
    t=ones(num_components(address), num_dimensions(geometry)),
    u=ones(num_components(address), num_components(address)),
    w=zeros(num_components(address), num_components(address)),
)
    C = num_components(address)
    D = num_dimensions(geometry)
    S = size(geometry)

    # Sanity checks
    if prod(size(geometry)) ≠ num_modes(address)
        throw(ArgumentError("`geometry` does not have the correct number of sites"))
    elseif !(address isa SingleComponentFockAddress || address isa CompositeFS)
        throw(ArgumentError(
            "unsupported address type detected use `CompositeFS` or 
            `<: SingleComponentFockAddress`"
        ))
    end

    t_mat = _t_or_v_to_matrix(:t, t, C, D; zero_is_nothing=false)
    u_mat = _u_or_w_to_matrix(:u, u, C)
    w_mat = _u_or_w_to_matrix(:w, w, C)

    warn_fermi_interaction(address, u_mat)
    ks_mat = Array{Float64}[]
    for i in eachindex(S)
        step = 2π/S[i]
        if isodd(S[i])
            start = -π*(1+1/S[i]) + step
        else
            start = -π + step
        end
        kr = range(start; step = step, length = S[i])
        if isodd(S[i])
            push!(ks_mat,[j for j in kr])
        else
            push!(ks_mat,reverse([j for j in kr]))
        end
    end
    kes, ks =     dispersion_mom_space(SVector{D}(ks_mat), geometry, t_mat)

    return HubbardMomSpace{C,D,typeof(address),typeof(geometry),typeof(ks),typeof(kes),
    typeof(t_mat),typeof(u_mat),typeof(w_mat)}(
        address, ks, kes, t_mat, u_mat, w_mat, geometry,
    )
end

LOStructure(::Type{<:HubbardMomSpace}) = IsHermitian()

function Base.show(io::IO, h::HubbardMomSpace{C}) where {C}
    io = IOContext(io, :compact => true)
    println(io, "HubbardMomSpace(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  t = ", (h.t), ",")
    if isnothing(h.u)
        println(io, "  u = ", zeros(C,C), ",")
    else
        println(io, "  u = ", Float64.(h.u), ",")
    end
    if isnothing(h.w)
        println(io, "  w = ", zeros(C,C), ",")
    else
        println(io, "  w = ", Float64.(h.w), ",")
    end
    print(io, ")")
end

# Overload equality due to stored potential energy arrays.
Base.:(==)(H::HubbardMomSpace, G::HubbardMomSpace) = 
    all(map(p -> getproperty(H, p) == getproperty(G, p), propertynames(H)))

starting_address(h::HubbardMomSpace) = h.address

dimension(::HubbardMomSpace, address) = number_conserving_dimension(address)

# offdiaonals =========================================================================== #
# Holds the offdiagonals for a single-component nearest neighbour one-body term. It's
# structured like a matric where the first index determines the occupied site in the adress
# and the second index determines the site the particle will hop to.
struct HubbardMomSpaceComponentData{
    C,I1,I2,D,G,A,A1,A2,U<:Union{Float64,Nothing},W<:Union{Float64,Nothing},O1,O2
} <: AbstractMatrix{Pair{A,Float64}}
    geometry::G
    parent_address::A
    address1::A1
    address2::A2
    u::U # interaction strength
    w::W # nearest neighbour interaction strength
    occmap1::O1
    occmap2::O2
    function HubbardMomSpaceComponentData{C,I1,I2,D}(
        geometry::G,
        parent::A,
        address1::A1,
        address2::A2,
        u::U,
        w::W,
        occmap1::O1=occupied_mode_map(address1),
        occmap2::O2=occupied_mode_map(address2),
    ) where {C,I1,I2,D,G,A,A1,A2,U,W,O1,O2}
        return new{C,I1,I2,D,G,A,A1,A2,U,W,O1,O2}(
            geometry, parent, address1, address2, u, w, occmap1, occmap2
        )
    end
end

function Base.size(data::HubbardMomSpaceComponentData{<:Any,I,I}) where {I}
    if isnothing(data.u) && isnothing(data.w)
        return 0
    else
        M = num_modes(data.address1)
        s1, d1 = num_singly_doubly_occupied_sites(data.address1)
        return  s1 * (s1 - 1) * (M - 2) + d1 * (M - 1)
    end
end

function Base.size(data::HubbardMomSpaceComponentData)
    if isnothing(data.u) && isnothing(data.w)
        return 0
    else
        M = num_modes(data.address1)
        s1 = length(data.occmap1)
        s2 = length(data.occmap2)
        return s1 * s2 * (M - 1)
    end
end

component_index(::HubbardMomSpaceComponentData{<:Any,I1,I2}) where {I1,I2} = (I1, I2, I1 == I2)

function Base.getindex(data::HubbardMomSpaceComponentData{C,I,I,D}, chosen::Int) where {C,I,D}
    geometry = data.geometry
    S = size(geometry)
    M = prod(S)
    map1 = data.occmap1
    new_add, onproduct,_,_,q = mom_transfer_mom_space(data.address1, chosen, map1, geometry)
    if data.parent_address isa CompositeFS
        new_parent = BitStringAddresses.update_component(
            data.parent_address, new_add, Val(I)
        )
    else
        new_parent = new_add
    end
    return new_parent => _interaction_parameter(data.u, data.w, q, S) * onproduct / M
end

function Base.getindex(data::HubbardMomSpaceComponentData{C,I1,I2,D}, chosen::Int) where {C,I1,I2,D}
    geometry = data.geometry
    S = size(geometry)
    M = prod(S)
    new_add1, onproduct1,new_add2, onproduct2,_,_,q = 
        mom_transfer_mom_space(data.address1,data.address2,chosen,data.occmap1,data.occmap2,data.geometry)
    new_parent = BitStringAddresses.update_component(
        data.parent_address, new_add1, Val(I1)
    )
    new_parent = BitStringAddresses.update_component(
        new_parent, new_add2, Val(I2)
    )
    return new_parent =>  2 * _interaction_parameter(data.u, data.w, q, S) * onproduct1 * onproduct2 / M
end

@inline _interaction_parameter(u::Float64, w::Float64, q::SVector, S::NTuple) = u/2 + w * _cosin_sum(q, S)
@inline _interaction_parameter(::Nothing, w::Float64, q::SVector, S::NTuple) = w * _cosin_sum(q, S)
@inline _interaction_parameter(u::Float64, ::Nothing, ::SVector, ::NTuple) = u/2

# column ================================================================================= #
struct HubbardMomSpaceColumn{H,G,A,C<:Tuple} <: AbstractOperatorColumn{A,Float64,H}
    hamiltonian::H
    geometry::G
    address::A
    components::C
    num_offdiagonals::Int
end

parent_operator(column::HubbardMomSpaceColumn) = column.hamiltonian
starting_address(column::HubbardMomSpaceColumn) = column.address

function diagonal_element(col::HubbardMomSpaceColumn)
    return _mom_hopping(col.hamiltonian.kes, col.address) + 
        _mom_interactions_diag(col.components, col.geometry)/num_modes(col.address)
end

function operator_column(h::HubbardMomSpace{<:Any,<:Any,A,G}, address) where {A,G}
    components = _column_components(h, address)
    return HubbardMomSpaceColumn{typeof(h),G,A,typeof(components)}(
        h, h.geometry, address, components, sum(length, components)
    )
end

# Collect HubbardMomSpaceComponentData for each component of the address.
@inline function _column_components(h::HubbardMomSpace{1,D}, 
        address::SingleComponentFockAddress) where {D}
    if isnothing(h.w) && isnothing(h.u)
        return (HubbardMomSpaceComponentData{1,1,1,D}(h.geometry, address, address, 
            address, nothing, nothing),)
    elseif isnothing(h.w) && !isnothing(h.u)
        return (HubbardMomSpaceComponentData{1,1,1,D}(h.geometry, address, address, 
            address, h.u[1], nothing),)
    elseif !isnothing(h.w) && isnothing(h.u)
        return (HubbardMomSpaceComponentData{1,1,1,D}(h.geometry, address, address, 
            address, nothing, h.w[1]),)
    else
        return (HubbardMomSpaceComponentData{1,1,1,D}(h.geometry, address, address, 
            address, h.u[1], h.w[1]),)
    end
end

@inline function _column_components(h::HubbardMomSpace{1,D}, address::FermiFS) where {D}
    if !isnothing(h.w)
        return (HubbardMomSpaceComponentData{1,1,1,D}(h.geometry, address, address, 
            address, nothing, h.w[1,1]),)
    else
        return (HubbardMomSpaceComponentData{1,1,1,D}(h.geometry, address, address, 
            address, nothing, nothing),)
    end
end

@inline function _column_components(h::HubbardMomSpace, address::CompositeFS)
    return _column_components(h, address, address.components, h.u, h.w, Val(1))
end

@inline function _column_components(::HubbardMomSpace, _, ::Tuple{}, ::Union{SMatrix{0,0},Nothing}, 
    ::Union{SMatrix{0,0},Nothing}, ::Val)
    return ()
end
@inline function _column_components(
    h::HubbardMomSpace{C,D}, address, (a, as...),
    m::Union{SMatrix{N,N},Nothing}, σ::Union{SMatrix{N,N},Nothing},
    ::Val{I1}) where {C,D,N,I1}
    # Split the matrix into the column we need now, and the rest.
    (u, u_column...) = isnothing(m) ? (nothing, nothing) : Tuple(m[:, 1])
    (w, w_column...) = isnothing(σ) ? (nothing, nothing) : Tuple(σ[:, 1])
    # Type-stable way to subset SMatrix:
    m_rest = isnothing(m) ? nothing : SMatrix{N-1,N-1}(view(m, 2:N, 2:N))
    σ_rest = isnothing(σ) ? nothing : SMatrix{N-1,N-1}(view(σ, 2:N, 2:N))
    
    return (HubbardMomSpaceComponentData{C,I1,I1,D}(h.geometry, address, a, a, u, w), 
        _mom_interactions_col(h,address,a,as,u_column,w_column,Val(I1),Val(I1+1))...,
        _column_components(h, address, as, m_rest, σ_rest, Val(I1+1) )...,)
end

@inline _mom_interactions_col(::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress,
    ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Val, ::Val) = ()
@inline function _mom_interactions_col(h::HubbardMomSpace{C,D}, address::AbstractFockAddress, 
    a::SingleComponentFockAddress, (b,as...)::NTuple{N}, (u, us...)::NTuple{N}, (w, ws...)::NTuple{N}, 
    ::Val{I1}, ::Val{I2}) where {C,D,N,I1,I2}
    return (HubbardMomSpaceComponentData{C,I1,I2,D}(h.geometry, address, a, b, u, w), 
            _mom_interactions_col(h, address, a, as, us, ws, Val(I1), Val(I2+1))...)
end

@inline _mom_interactions_col(::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress, 
    ::Tuple{}, ::Tuple{}, ::Tuple{Nothing}, ::Val, ::Val) = ()
@inline function _mom_interactions_col(h::HubbardMomSpace{C,D}, address::AbstractFockAddress, 
    a::SingleComponentFockAddress, (b,as...)::NTuple{N}, m::NTuple{N}, σ::Tuple{Nothing}, ::Val{I1}, 
    ::Val{I2}) where {C,D,N,I1,I2}
    return (HubbardMomSpaceComponentData{C,I1,I2,D}(h.geometry, address, a, b, m[1], σ[1]), 
            _mom_interactions_col(h, address, a, as, m[2:N], σ, Val(I1), Val(I2+1))...)
end

@inline _mom_interactions_col(::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress, 
    ::Tuple{},::Tuple{Nothing}, ::Tuple{}, ::Val, ::Val) = ()
@inline function _mom_interactions_col(h::HubbardMomSpace{C,D}, address::AbstractFockAddress, 
    a::SingleComponentFockAddress, (b,as...)::Tuple{N}, m::Tuple{Nothing}, σ::NTuple{N}, ::Val{I1}, 
    ::Val{I2}) where {C,D,N,I1,I2}
    return (HubbardMomSpaceComponentData{C,I1,I2,D}(h.geometry, address, a, b, m[1], σ[1]), 
            _mom_interactions_col(h, address, a, as, m, σ[2:N], Val(I1), Val(I2+1))...)
end

@inline _mom_interactions_col(::HubbardMomSpace, ::AbstractFockAddress, ::SingleComponentFockAddress, 
    ::Tuple{},::Tuple{Nothing}, ::Tuple{Nothing}, ::Val, ::Val) = ()
@inline function _mom_interactions_col(h::HubbardMomSpace{C,D}, address::AbstractFockAddress, 
    a::SingleComponentFockAddress, (b,as...)::Tuple{N}, m::Tuple{Nothing}, σ::Tuple{Nothing}, ::Val{I1}, 
    ::Val{I2}) where {C,D,N,I1,I2}
    return (HubbardMomSpaceComponentData{C,I1,I2,D}(h.geometry, address, a, b, m[1], σ[1]), 
            _mom_interactions_col(h, address, a, as, m, σ, Val(I1), Val(I2+1))...)
end

# Split one-dimensional array `index` that indexes over many components simultaneously into
# a two-dimensional one. The dimension of the new index picks the component, while the
# second picks the offdiagonal within the component.

function random_offdiagonal(column::HubbardMomSpaceColumn)
    random_number = rand(1:column.num_offdiagonals)
    component, remainder = _split_component_from_index(column, random_number)

    addr, val = index_apply(getindex, column.components, component, remainder)
    return addr, 1/column.num_offdiagonals, val
end

struct HubbardMomSpaceColumnOffdiagonals{A,G,C<:Tuple} <: AbstractVector{Pair{A,Float64}}
    address::A
    geometry::G
    components::C
    num_offdiagonals::Int
end

function offdiagonals(column::HubbardMomSpaceColumn{<:Any,G,A,C}) where {G,A,C}
    return HubbardMomSpaceColumnOffdiagonals{A,G,C}(
        column.address,
        column.hamiltonian.geometry,
        column.components,
        column.num_offdiagonals,
    )
end

@inline function Base.iterate(ods::HubbardMomSpaceColumnOffdiagonals, state=(1,1))
    component_index, chosen = state
    component_index = _test_interaction_coeff(ods, component_index)
    if chosen > index_apply(size, ods.components, component_index)
        chosen = 1
        component_index += 1
    end
    if component_index > length(ods.components)
        return nothing
    else
        result = index_apply(
            getindex, ods.components, component_index, chosen
        )
        return result, (component_index, chosen + 1)
    end
end

@inline function _test_interaction_coeff(ods::HubbardMomSpaceColumnOffdiagonals, component_index::Int)
    if index_apply(size, ods.components, component_index) == 0 && component_index < length(ods.components)
        return _test_interaction_coeff(ods, component_index + 1)
    else
        return component_index
    end
end

Base.size(ods::HubbardMomSpaceColumnOffdiagonals) = (ods.num_offdiagonals,)
Base.eltype(::HubbardMomSpaceColumnOffdiagonals{A}) where {A} = Pair{A,Float64}

function Base.getindex(column::HubbardMomSpaceColumnOffdiagonals, index)
    component_index, inner_index = _split_component_from_index(column, index)
    return index_apply(getindex, column.components, component_index, inner_index)
end

########################################################################################################
# Momentum operator in momentum space
########################################################################################################

struct MomentumMomSpace{T,C,D,H<:AbstractHamiltonian{T}} <: AbstractHamiltonian{Vector{T}}
    ham::H
end
LOStructure(::Type{MomentumMomSpace{H,T}}) where {H,T <: Real} = IsDiagonal()
num_offdiagonals(ham::MomentumMomSpace, _) = 0
function diagonal_element(mom::MomentumMomSpace{<:Any,1,D}, address::SingleComponentFockAddress) where D
    return [dot(mom.ham.ks[i,:], occupied_mode_map(address)) for i in 1:D]
end
function diagonal_element(mom::MomentumMomSpace{<:Any,C,D}, address::CompositeFS) where {C,D}
    return [sum(dot(mom.ham.ks[i,:], occupied_mode_map(c)) for c in address.components) for i in 1:D]
end
# fold into (-π, π]
starting_address(mom::MomentumMomSpace) = starting_address(mom.ham)

momentum(ham::HubbardMomSpace{C,D}) where {C,D} = MomentumMomSpace{Float64,C,D,typeof(ham)}(ham)
