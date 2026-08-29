"""
    FroehlichPolaronND(address::BoseFS{missing,M}; kwargs...) <: AbstractHamiltonian


The Froehlich polaron Hamiltonian for a `D` dimensional box of volume`l^D` with `M ` momentum modes is given by

```math
H = \\frac{(\\hat{p}_f - \\mathbf{p})^2}{2m} + \\omega \\hbar \\sum_{\\mathbf{k}}\\hat{N}_{\\mathbf{k}} +  \\sum_{\\mathbf{k}}(v_{\\mathbf{k}}^{\\ast} \\hat{a}_{\\mathbf{k}}^\\dagger + v_{\\mathbf{k}} \\hat{a}_{\\mathbf{k}})
```

where ``p`` is the total momentum, ``p̂_f = Σ_k k âₖ^† âₖ`` is the momentum operator for the
bosons, and ``k`` part of the momentum lattice with separation ``2π/l``. ``N̂`` is the number
operator for the bosons.

We have in ``D`` dimensions,

```math
V_k = (-1)\\sqrt{\\frac{(\\Gamma(\\frac{D-1}{2})  \\alpha  2^{D-1}  \\pi^{\\frac{D-1}{2}}  \\omega^2)} { (\\sqrt{m \\omega})}(\\frac{1}{k^{D-1}V})}
```
All of the components of ``V_k`` that are not dependent on ``\\mathbf{k}`` are stored as a field in the Hamiltonian ``vk_constant``
# Keyword Arguments

* `p=[0.0,...]`: the total momentum vector ``p``.
* `D = 1`: the dimension of the Hamiltonian
* `alpha=1.0`: the dimensionless coupling strength ``\\alpha``.
* `mass=1.0`: the particle mass ``2m``.
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

julia> ham = FroehlichPolaronND(fs; D = 2,alpha = 1)
FroehlichPolaronND(
  fs"|0 0 0 0⟩{}",
  geometry = CubicGrid((2, 2), (true, true)),
  alpha = 1.0, D = 2, mass = 1.0, omega = 1.0,
  l = [1.0, 1.0], p = [0.0, 0.0],
  mode_cutoff = 255,
  twist = [0.0, 0.0]
)

julia> dimension(ham)
4294967296

julia> dimension(FroehlichPolaronND(fs; alpha = 1,D = 2, mode_cutoff=5))
1296
```

See also [`BoseFS`](@ref), [`dimension`](@ref), [`AbstractHamiltonian`](@ref).
"""
struct FroehlichPolaronND{
        T,
        M, #num_modes
        D, #dimension
        A<:BoseFS{missing,M},
        MC,
        G<:CubicGrid
    } <: AbstractHamiltonian{T}
    address::A
    geometry::G
    alpha::T
    mass::T
    omega::T
    l::SVector{D, Float64}
    p::SVector{D, Float64}
    ks::SVector{M, SVector{D, T}}
    momentum_cutoff::Union{MC, Nothing}
    mode_cutoff::Union{Int, Nothing}
    vk_constant::T
    twist::SVector{D, Float64}
end


function FroehlichPolaronND(
    address::BoseFS{missing,M,SVector{M,AT}};
    D::Int = 1,
    geometry::CubicGrid = PeriodicBoundaries(ntuple(Returns(round(Int, M^(1/D))), D)),
    alpha = 1,
    mass = 1,
    omega = 1,
    l = ones(D),
    p = zeros(D),
    momentum_cutoff = nothing,
    mode_cutoff = nothing,
    twist = zeros(D)

) where {M, AT}
    if D != 1
        vk_constant = sqrt((gamma(((D-1)/2)) * alpha * 2^(D-1) * pi^((D-1)/2) * omega^2) / (sqrt(mass * omega)*prod(l)))
    else
        vk_constant = (2 * alpha /(l[1]))^0.5
    end

    if length(l) != D || any(x -> x <= 0, l)
        throw(ArgumentError("`l` must be a positive-valued vector of length $D"))
    end

    if abs(M^(1/D) - round(M^(1/D))) > 0.01
        throw(ArgumentError("num_modes(address)==$M is not an integer power with exponent $D"))
    end

    alpha, mass, omega = promote(float(alpha), float(mass), float(omega))
    l = SVector{D,Float64}(float.(l))
    p = SVector{D,Float64}(float.(p))

    ks_tmp = Vector{SVector{D,Float64}}(undef, M)

    for idx_mode in 1:M
        idx = Tuple(geometry[idx_mode])

        kv = ntuple(d -> begin
            if isodd(round(M^(1/D)))

                m = idx[d] - 1 - div(round(M^(1/D)), 2) + twist[d]
            else

                m = idx[d] - div(round(M^(1/D)) ,2) + twist[d]
            end
            (2π / l[d]) * m
        end, D)
        ks_tmp[idx_mode] = SVector{D,Float64}(kv)
    end

    ks = SVector{M,SVector{D,Float64}}(Tuple(ks_tmp))

    if !isnothing(momentum_cutoff)
        momentum_cutoff = typeof(alpha)(momentum_cutoff)
        momentum = dot(ks,onr(address))
        if abs(momentum) > momentum_cutoff
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

    return FroehlichPolaronND{typeof(alpha), M,D, typeof(address), typeof(momentum_cutoff),typeof(geometry)}(
        address ,geometry, alpha, mass, omega, l, p, ks, momentum_cutoff, mode_cutoff,vk_constant,twist)
end


function Base.show(io::IO, h::FroehlichPolaronND{T,M,D}) where {T,M,D}  #put D is the show function
    io = IOContext(io, :compact => true)
    println(io, "FroehlichPolaronND(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  alpha = ", h.alpha,", D = ", D,  ", mass = ", h.mass, ", omega = ", h.omega,",")
    println(io, "  l = ", Float64.(h.l), ", p = ", Float64.(h.p),  ",")
    !isnothing(h.momentum_cutoff) && println(io, "  momentum_cutoff = ", h.momentum_cutoff, ",")
    println(io, "  mode_cutoff = ", isnothing(h.mode_cutoff) ? "nothing" : string(h.mode_cutoff), ",")
    println(io, "  twist = ", h.twist)
    print(io, ")")
end

LOStructure(::Type{<:FroehlichPolaronND}) = IsHermitian()

starting_address(h::FroehlichPolaronND) = h.address

function dimension(h::FroehlichPolaronND, address)
    M = num_modes(address)
    n = h.mode_cutoff
    return BigInt(n + 1)^BigInt(M)
end

struct FroehlichPolaronNDColumn{H,A} <: AbstractOperatorColumn{A,Float64,H}
    hamiltonian::H
    address::A
    num_offdiagonals::Int
end

function operator_column(h::FroehlichPolaronND, address)
    M = num_modes(address)
    return FroehlichPolaronNDColumn(h, address, 2M)
end

function diagonal_element(col::FroehlichPolaronNDColumn)
    h = col.hamiltonian
    occ = onr(col.address)

    Pphonon = zero(h.ks[1])
    Nphonon = 0
    for m in 1:num_modes(col.address)
        nm = occ[m]
        Nphonon += nm
        Pphonon += h.ks[m] * nm
    end
    ek = dot(h.p - Pphonon, h.p - Pphonon) / (h.mass)
    return (h.omega * Nphonon)+ ek
end

struct FroehlichPolaronNDOffdiagonals{A,H} <: AbstractVector{Pair{A,Float64}}
    address::A
    h::H
    num_offdiagonals::Int
end

offdiagonals(col::FroehlichPolaronNDColumn) = FroehlichPolaronNDOffdiagonals(col.address, col.hamiltonian, col.num_offdiagonals)


Base.size(ods::FroehlichPolaronNDOffdiagonals) = (ods.num_offdiagonals,)


@inline function calc_vk(h::FroehlichPolaronND{T,M,D}, kidx::Int) where {T,M,D}
    knorm = sqrt(dot(h.ks[kidx], h.ks[kidx]))

    if D == 1
        return h.vk_constant
    else
        if knorm == 0.0
            return  0.0
        else
            return h.vk_constant * sqrt(1/ (knorm^(D-1)))
        end
    end
end

"""
    phonon_op(h::FroehlichPolaronND, addr, chosen)
The phonon_op function applies the creation and annihilation operators on a chosen address
and returns the new offdiagonal element.
"""
@inline function phonon_op(h::FroehlichPolaronND{T,M,D}, addr, chosen) where {T,M,D}
    if chosen ≤ M
        if !isnothing(h.mode_cutoff) && onr(addr)[chosen] ≥ h.mode_cutoff
            return addr => 0.0
        end
        new_addr, val = excitation(addr, (chosen,), ())
        vk = calc_vk(h, chosen)
        amp = -vk * val

    else
        k = chosen - M
        new_addr, val = excitation(addr, (), (k,))
        vk = calc_vk(h, k)
        amp = -vk * val

    end

    if !isnothing(h.momentum_cutoff)
        occ = onr(new_addr)
        phononmom = zero(h.ks[1])
        for m in 1:M

            phononmom += h.ks[m] * occ[m]

        end
        if norm(phononmom) ≤ h.momentum_cutoff
            return new_addr => amp
        else
            return addr => 0.0
        end
    else
        return new_addr => amp
    end
end


function Base.getindex(ods::FroehlichPolaronNDOffdiagonals, i::Int)
    return phonon_op(ods.h, ods.address, i)
end


function random_offdiagonal(col::FroehlichPolaronNDColumn)
    M2 = col.num_offdiagonals
    i = rand(1:M2)
    addr_val = phonon_op(col.hamiltonian, col.address, i)
    return first(addr_val), 1/M2, last(addr_val)
end

parent_operator(col::FroehlichPolaronNDColumn) = col.hamiltonian
starting_address(col::FroehlichPolaronNDColumn) = col.address
num_offdiagonals(col::FroehlichPolaronNDColumn) = col.num_offdiagonals
