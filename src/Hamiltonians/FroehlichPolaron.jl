"""
    FroehlichPolaron(address::BoseFS{missing,M}; kwargs...) <: AbstractHamiltonian


The Froehlich polaron Hamiltonian for a `D` dimensional box of volume`l^D` with `M ` momentum modes is given by

```math
H = (𝐩̂_f - 𝐩)^2/2m + ωN̂  +  Σ_𝐤 v_𝐤(â^†_𝐤 + â_{-𝐤})
```

where ``𝐩`` is the total momentum vector, ``𝐩̂_f = Σ_𝐤 𝐤 â^†_𝐤 â_𝐤`` is the momentum operator
for the bosons, and the ``𝐤`` is the single-phonon momentum on a lattice with separation
``2π/l``. ``N̂` = Σ_𝐤 â^†_𝐤 â_𝐤` is the number operator for the bosons.

The coupling constant in ``D`` dimensions is given by
```math
v_k = (-1)\\sqrt{\\frac{(\\Gamma(\\frac{D-1}{2})  \\alpha  2^{D-1}  \\pi^{\\frac{D-1}{2}}  \\omega^2)} { (\\sqrt{m \\omega})}(\\frac{1}{k^{D-1}V})}
```

# Keyword Arguments

* `p=[0.0,...]`: the total momentum vector ``𝐩``.
* `D = 1`: the dimension of the Hamiltonian.
* `alpha=1.0`: the dimensionless coupling strength ``\\alpha``.
* `two_m=1.0`: twice the particle mass ``2m``.
* `omega=1.0`: the oscillation frequency of the phonons ``ω``.
* `l=[1.0,...]`: the ``D`` dimensional hypercube dimensions in real space ``l``. Provides scale parameter of the momentum
    lattice.
* `momentum_cutoff=nothing`: the maximum boson momentum allowed for an address.
* `mode_cutoff`: the maximum number of bosons in each momentum mode. Defaults to the maximum
    value supported by the address type [`BoseFS{missing}`](@ref).
* `twist=zeros(D)`: twist the boundary conditions in each dimension by the given value.

# Examples
```jldoctest
julia> fs = BoseFS{missing}(0,0,0,0)
BoseFS{missing}(0, 0, 0, 0)

julia> ham = FroehlichPolaron(fs; D = 2,alpha = 1)
FroehlichPolaron(
  fs"|0 0 0 0⟩{}",
  geometry = CubicGrid((2, 2), (true, true)),
  alpha = 1.0, D = 2, two_m = 1.0, omega = 1.0,
  l = [1.0, 1.0], p = [0.0, 0.0],
  mode_cutoff = 255,
  twist = [0.0, 0.0]
)

julia> dimension(ham)
4294967296

julia> dimension(FroehlichPolaron(fs; alpha = 1,D = 2, mode_cutoff=5))
1296
```

See also [`BoseFS`](@ref), [`dimension`](@ref), [`AbstractHamiltonian`](@ref).
"""
struct FroehlichPolaron{
        T,
        M, #num_modes
        D, #dimension
        A<:BoseFS{missing,M},
        MC, # momentum cutoff indicating type
        G<:CubicGrid,
        TW
    } <: AbstractHamiltonian{T}
    address::A
    geometry::G
    alpha::T
    two_m::T
    omega::T
    l::SVector{D, T}
    p::SVector{D, T}
    ks::SVector{M, SVector{D, T}}
    momentum_cutoff::MC
    mode_cutoff::Int
    vk_constant::T
    twist::TW
end

function FroehlichPolaron(
    address::BoseFS{missing};
    alpha = 1,
    two_m = 1,
    omega = 1,
    mass=nothing, # deprecated keyword, use `two_m` instead
    kwargs...
)
    if !isnothing(mass)
        @warn "The keyword argument `mass` is deprecated. Use `two_m` instead."
        two_m = mass
    end

    T = float(promote_type(typeof(alpha), typeof(two_m), typeof(omega)))
    return FroehlichPolaron{T}(address; alpha, two_m, omega, kwargs...)
end

function FroehlichPolaron{T}(
    address::BoseFS{missing,M,SVector{M,AT}};
    D::Int = 1,
    geometry::CubicGrid = PeriodicBoundaries(ntuple(Returns(round(Int, M^(1/D))), D)),
    alpha = 1,
    two_m = 1,
    omega = 1,
    l = ones(D),
    p = zeros(D),
    momentum_cutoff = nothing,
    mode_cutoff = nothing,
    twist = nothing,
    mass=nothing, # deprecated keyword, use `two_m` instead
    v = nothing,
) where {T, M, AT}
    D = dimension(geometry) # geometry overrides D keyword argument
    if !isnothing(mass)
        @warn "The keyword argument `mass` is deprecated. Use `two_m` instead."
        two_m = mass
    end
    alpha, two_m, omega = T.((alpha, two_m, omega))

    if !isnothing(v)
        if D != 1
            throw(ArgumentError("v is only supported for D=1"))
        end
        @warn "The keyword argument `v` is deprecated. Use `alpha` instead."
        alpha = T(only(l) * T(v)^2 * sqrt(two_m * omega)/(2 * omega^2))
    end

    if D != 1
        vk_constant = sqrt((gamma(((D-1)/T(2))) * alpha * 2^(D-1) * pi^((D-1)/T(2)) *
            omega^2) / (sqrt(two_m * omega)*prod(l))
        )
    else
        vk_constant = sqrt(2 * alpha /only(l))
    end

    if length(l) != D || any(x -> x <= 0, l)
        throw(ArgumentError("`l` must be a positive-valued vector of length $D"))
    end

    if abs(M^(1/D) - round(M^(1/D))) > 0.01
        throw(ArgumentError("num_modes(address)==$M must be an integer power with exponent $D"))
    end
    if M != length(geometry)
        throw(ArgumentError("num_modes(address)==$M must equal length(geometry)==$(length(geometry))"))
    end

    l = SVector{D,T}(T.(l))
    p = SVector{D,T}(T.(p))

    ks_tmp = Vector{SVector{D,T}}(undef, M)

    twist = isnothing(twist) ? nothing : SVector{D,T}(T.(twist))

    linear_dimension = size(geometry)[1]
    for idx_mode in 1:M
        idx = Tuple(geometry[idx_mode])
        kv = ntuple(d -> begin
            m = isnothing(twist) ? zero(T) : twist[d]
            if isodd(linear_dimension)
                m += idx[d] - 1 - div(linear_dimension, 2)
            else
                m = idx[d] - div(linear_dimension, 2)
            end
            (2π / l[d]) * m
        end, D)
        ks_tmp[idx_mode] = SVector{D,T}(kv)
    end

    ks = SVector{M,SVector{D,T}}(Tuple(ks_tmp))

    if !isnothing(momentum_cutoff)
        momentum_cutoff = T(momentum_cutoff)
        momentum = norm(_p_phonon(ks, address))
        if momentum > momentum_cutoff
            throw(ArgumentError("Starting address has momentum $momentum which cannot exceed momentum_cutoff $momentum_cutoff"))
        end
    end

    if isnothing(mode_cutoff)
        mode_cutoff = Int(typemax(AT))
    end
    mode_cutoff = floor(Int, mode_cutoff)::Int
    if _exceed_mode_cutoff(mode_cutoff, address)
        throw(ArgumentError("Starting address cannot have occupations that exceed mode_cutoff"))
    end

    return FroehlichPolaron{
        T, M, D, typeof(address), typeof(momentum_cutoff), typeof(geometry), typeof(twist)
    }(
        address, geometry, alpha, two_m, omega,
        l, p, ks,
        momentum_cutoff, mode_cutoff, vk_constant,
        twist
    )
end

function Base.show(io::IO, h::FroehlichPolaron{T,M,D}) where {T,M,D}  #put D is the show function
    io = IOContext(io, :compact => true)
    print(io, "FroehlichPolaron")
    eltype(h) === Float64 || print(io, "{$(eltype(h))}")
    println(io, "(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  alpha = ", h.alpha,", D = ", D,  ", two_m = ", h.two_m, ", omega = ", h.omega,",")
    println(io, "  l = $(h.l), p = $(h.p),")
    isnothing(h.momentum_cutoff) || println(io, "  momentum_cutoff = ", h.momentum_cutoff, ",")
    println(io, "  mode_cutoff = ",  h.mode_cutoff, ",")
    isnothing(h.twist) || println(io, "  twist = $(h.twist),")
    print(io, ")")
end

LOStructure(::Type{<:FroehlichPolaron}) = IsHermitian()

starting_address(h::FroehlichPolaron) = h.address

function dimension(h::FroehlichPolaron, address)
    M = num_modes(address)
    n = h.mode_cutoff
    return BigInt(n + 1)^BigInt(M)
end

struct FroehlichPolaronColumn{A,T,H} <: AbstractOperatorColumn{A,T,H}
    hamiltonian::H
    address::A
    num_offdiagonals::Int
end

function operator_column(h::FroehlichPolaron, address)
    M = num_modes(address)
    T = eltype(h)
    return FroehlichPolaronColumn{typeof(address),T,typeof(h)}(h, address, 2M)
end

function diagonal_element(col::FroehlichPolaronColumn)
    h = col.hamiltonian
    occ = onr(col.address)

    # calculate the phonon momentum and number of phonons
    p_phonon = -h.p
    n_phonon = 0
    for m in 1:num_modes(col.address)
        nm = occ[m]
        n_phonon += nm
        p_phonon += h.ks[m] * nm
    end

    return (h.omega * n_phonon) + dot(p_phonon, p_phonon) / (h.two_m)
end

# Calculate the total phonon momentum for a given address using the momentum modes `ks`.
function _p_phonon(ks, addr)
    occ = onr(addr)
    p_f = zero(ks[1])
    for m in 1:num_modes(addr)
        p_f += ks[m] * occ[m]
    end
    return p_f
end
struct FroehlichPolaronOffdiagonals{A,T,H} <: AbstractVector{Pair{A,T}}
    address::A
    h::H
    num_offdiagonals::Int
end

function offdiagonals(col::FroehlichPolaronColumn{A,T,H}) where {A,T,H}
    FroehlichPolaronOffdiagonals{A,T,H}(col.address, col.hamiltonian, col.num_offdiagonals)
end

Base.size(ods::FroehlichPolaronOffdiagonals) = (ods.num_offdiagonals,)

calc_vk(h::FroehlichPolaron{<:Any,<:Any,1}, _) = h.vk_constant
@inline function calc_vk(h::FroehlichPolaron{T,M,D}, kidx) where {T,M,D}
    knorm = norm(h.ks[kidx]) # D > 1
    if iszero(knorm)
        return  zero(T)
    else
        return h.vk_constant * sqrt(1/ (knorm^(D-1)))
    end
end

"""
    phonon_op(h::FroehlichPolaron, addr, chosen)
The phonon_op function applies the creation and annihilation operators on a chosen address
and returns the new offdiagonal element.
"""
@inline function phonon_op(h::FroehlichPolaron{T,M,D}, addr, chosen) where {T,M,D}
    if chosen ≤ M
        if !isnothing(h.mode_cutoff) && onr(addr)[chosen] ≥ h.mode_cutoff
            return addr => zero(T)
        end
        new_addr, val = excitation(T, addr, (chosen,), ())
    else
        k = chosen - M
        new_addr, val = excitation(T, addr, (), (k,))
    end
    vk = calc_vk(h, chosen)
    amp = -vk * val

    if !isnothing(h.momentum_cutoff) && norm(_p_phonon(h.ks, new_addr)) > h.momentum_cutoff
        return addr => zero(T)
    end
    return new_addr => amp
end

function Base.getindex(ods::FroehlichPolaronOffdiagonals, i::Int)
    return phonon_op(ods.h, ods.address, i)
end

function random_offdiagonal(col::FroehlichPolaronColumn{<:Any,T}) where {T}
    M2 = col.num_offdiagonals
    i = rand(1:M2)
    addr_val = phonon_op(col.hamiltonian, col.address, i)
    return first(addr_val), 1/T(M2), last(addr_val)
end

parent_operator(col::FroehlichPolaronColumn) = col.hamiltonian
starting_address(col::FroehlichPolaronColumn) = col.address
num_offdiagonals(col::FroehlichPolaronColumn) = col.num_offdiagonals
