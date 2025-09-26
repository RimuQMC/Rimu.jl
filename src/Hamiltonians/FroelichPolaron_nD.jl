using SpecialFunctions

#structure 
struct FroehlichPolaron_nD{
    T,
    M,
    D,
    A<:OccupationNumberFS{M}, 
    MC,
    G<:CubicGrid
    } <: AbstractHamiltonian{T}
    address::A
    d:: Int64
    geometry::G
    alpha::T
    mass::T
    omega::T
    l::SVector{D, Float64}
    p::SVector{D, Float64}
    ks::SVector{M, SVector{D, T}}
    momentum_cutoff::Union{MC, Nothing}
    mode_cutoff::Union{Int, Nothing}
end

#function for nD Froehlich Polaron
function FroehlichPolaron_nD(
    address::OccupationNumberFS{M,AT};
    D::Int = 2,
    geometry::CubicGrid = PeriodicBoundaries(ntuple(_ -> Int(round(num_modes(address)^(1/D))), D)),
    alpha = 1,
    mass = 1,
    omega = 1,
    l = ones(D),
    p = zeros(D),
    momentum_cutoff::Union{Nothing,Real} = nothing,
    mode_cutoff::Union{Nothing,Int} = nothing,
) where {M, AT}

    # Check positive l
    if length(l) != D || any(x -> x <= 0, l)
        throw(ArgumentError("`l` must be a positive vector of length $D"))
    end

    #check m_lin is possible
    if abs(M^(1/D) - round(M^(1/D))) >0.01  
        throw(ArgumentError("M = $M is not a perfect $D th power"))
    end

    alpha, mass, omega = promote(float(alpha), float(mass), float(omega))
    l = SVector{D,Float64}(float.(l))
    p = SVector{D,Float64}(float.(p))

    ks_tmp = Vector{SVector{D,Float64}}(undef, M)

    #build k grid
    for idx_mode in 1:M
    idx = Tuple(geometry[idx_mode])
    kv = ntuple(d -> begin
        m = idx[d] - 1 - div(round(M^(1/D)),2)   
        (2π / l[d]) * m
    end, D)
    ks_tmp[idx_mode] = SVector{D,Float64}(kv)
    end

    ks = SVector{M,SVector{D,Float64}}(Tuple(ks_tmp))


    return FroehlichPolaron_nD{typeof(alpha), M, D, typeof(address), typeof(momentum_cutoff),typeof(geometry)}(
        address,D, geometry, alpha, mass, omega, l, p, ks, momentum_cutoff, mode_cutoff)
end


function Base.show(io::IO, h::FroehlichPolaron_nD)
    io = IOContext(io, :compact => true)
    println(io, "FroehlichPolaron_nD(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  alpha = ", h.alpha, ", mass = ", h.mass, ", omega = ", h.omega, ", Dimention = ", h.d,",")
    println(io, "  l = ", Float64.(h.l), ", p = ", Float64.(h.p), ",")
    !isnothing(h.momentum_cutoff) && println(io, "  momentum_cutoff = ", h.momentum_cutoff, ",")
    println(io, "  mode_cutoff = ", isnothing(h.mode_cutoff) ? "nothing" : string(h.mode_cutoff))
    print(io, ")")
end

LOStructure(::Type{<:FroehlichPolaron_nD}) = IsHermitian()

starting_address(h::FroehlichPolaron_nD) = h.address

function dimension(h::FroehlichPolaron_nD, address)
    M = num_modes(address)
    n = isnothing(h.mode_cutoff) ? typemax(Int) : h.mode_cutoff
    return BigInt(n + 1)^BigInt(M)
end

struct FroehlichPolaron_nDColumn{H,A} <: AbstractOperatorColumn{A,Float64,H}
    hamiltonian::H
    address::A
    num_offdiagonals::Int
end


#diagonal elements calculator
function diagonal_element(col::FroehlichPolaron_nDColumn)
    h = col.hamiltonian
    occ = onr(col.address)

    Pph = zero(h.ks[1])
    Nph = 0
    for m in 1:num_modes(col.address)
        nm = occ[m]
        Nph += nm
        Pph += h.ks[m] * nm
    end
    ek = dot(h.p - Pph, h.p - Pph) / (h.mass)
    return h.omega * Nph + ek
end

function operator_column(h::FroehlichPolaron_nD, address)
    M = num_modes(address)
    return FroehlichPolaron_nDColumn(h, address, 2M)
end

struct FroehlichPolaron_nDOffdiagonals{A,H} <: AbstractVector{Pair{A,Float64}}
    address::A
    h::H
    num_offdiagonals::Int
end

offdiagonals(col::FroehlichPolaron_nDColumn) = FroehlichPolaron_nDOffdiagonals(col.address, col.hamiltonian, col.num_offdiagonals)

Base.size(ods::FroehlichPolaron_nDOffdiagonals) = (ods.num_offdiagonals,)
Base.eltype(::FroehlichPolaron_nDOffdiagonals{A}) where {A} = Pair{A,Float64}


#calculate V_k for a specific k. Edge case of D = 1.


@inline function calc_vk(h::FroehlichPolaron_nD, kidx::Int)
    knorm = sqrt(dot(h.ks[kidx], h.ks[kidx]))
    vol = prod(h.l)
    if h.d == 1
        return (2 * h.alpha /(h.l[1]))^0.5
    else
        if knorm == 0.0
            return 0.0
        else
            return (-1) * sqrt((gamma(((h.d-1)/2)) * h.alpha * 2^(h.d-(3/2)) * pi^((h.d-1)/2) * h.omega^2) / (sqrt(h.mass * h.omega) * (knorm^(h.d-1) * vol)))
        end
    end
end

@inline function phonon_op(h::FroehlichPolaron_nD, addr, chosen)
    M = num_modes(addr)
    if chosen ≤ M
        if !isnothing(h.mode_cutoff) && onr(addr)[chosen] ≥ h.mode_cutoff
            return addr => 0.0
        end
        new_addr, val = excitation(addr, (chosen,), ())
        vk = calc_vk(h, chosen)
        amp = -vk * val
        return new_addr => amp
    else
        k = chosen - M
        new_addr, val = excitation(addr, (), (k,))
        vk = calc_vk(h, k)
        amp = -vk * val
        return new_addr => amp
    end

    if isnothing(h.momentum_cutoff)
        return new_addr => amp
    end

    occ = onr(new_addr)
    Pph = zero(h.ks[1])
    for m in 1:M
        Pph += h.ks[m] * occ[m]
    end
    if norm(Pph) ≤ h.momentum_cutoff
        return new_addr => amp
    else
        return addr => 0.0
    end
end

function Base.getindex(ods::FroehlichPolaron_nDOffdiagonals, i::Int)
    return phonon_op(ods.h, ods.address, i)
end

function Base.iterate(ods::FroehlichPolaron_nDOffdiagonals, state::Int=1)
    state > ods.num_offdiagonals && return nothing
    return ods[state], state + 1
end

function random_offdiagonal(col::FroehlichPolaron_nDColumn)
    M2 = col.num_offdiagonals
    i = rand(1:M2)
    addr_val = phonon_op(col.hamiltonian, col.address, i)
    return first(addr_val), 1/M2, last(addr_val)
end


parent_operator(col::FroehlichPolaron_nDColumn) = col.hamiltonian
starting_address(col::FroehlichPolaron_nDColumn) = col.address
num_offdiagonals(col::FroehlichPolaron_nDColumn) = col.num_offdiagonals