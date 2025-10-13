using SpecialFunctions


struct FroehlichPolaronND{
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

function operator_column(h::FroehlichPolaronND, address)
    M = num_modes(address)
    return FroehlichPolaronNDColumn(h, address, 2M)
end


function FroehlichPolaronND(
    address::OccupationNumberFS{M,AT};
    D::Int = 1,
    geometry::CubicGrid = PeriodicBoundaries(ntuple(Returns(round(Int, M^(1/D))), D)),
    alpha = 1,
    mass = 1,
    omega = 1,
    l = ones(D),
    p = zeros(D),
    momentum_cutoff = nothing,
    mode_cutoff = nothing,
) where {M, AT}

    
    if length(l) != D || any(x -> x <= 0, l)
        throw(ArgumentError("`l` must be a positive vector of length $D"))
    end

    
    if abs(M^(1/D) - round(M^(1/D))) > 0.01 
        throw(ArgumentError("M = $M is not a perfect $D th power"))
    end

    alpha, mass, omega = promote(float(alpha), float(mass), float(omega))
    l = SVector{D,Float64}(float.(l))
    p = SVector{D,Float64}(float.(p))

    ks_tmp = Vector{SVector{D,Float64}}(undef, M)

    
    for idx_mode in 1:M
        idx = Tuple(geometry[idx_mode])
        
        kv = ntuple(d -> begin
            if isodd(round(M^(1/D)))
                
                m = idx[d] - 1 - div(round(M^(1/D)), 2)
            else
                
                m = idx[d] - div(round(M^(1/D)) ,2)
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


    return FroehlichPolaronND{typeof(alpha), M, D, typeof(address), typeof(momentum_cutoff),typeof(geometry)}(
        address,D, geometry, alpha, mass, omega, l, p, ks, momentum_cutoff, mode_cutoff)
end


function Base.show(io::IO, h::FroehlichPolaronND)
    io = IOContext(io, :compact => true)
    println(io, "FroehlichPolaronND(")
    println(io, "  ", starting_address(h), ",")
    println(io, "  geometry = ", h.geometry, ",")
    println(io, "  alpha = ", h.alpha, ", mass = ", h.mass, ", omega = ", h.omega, ", Dimention = ", h.d,",")
    println(io, "  l = ", Float64.(h.l), ", p = ", Float64.(h.p), ",")
    !isnothing(h.momentum_cutoff) && println(io, "  momentum_cutoff = ", h.momentum_cutoff, ",")
    println(io, "  mode_cutoff = ", isnothing(h.mode_cutoff) ? "nothing" : string(h.mode_cutoff))
    print(io, ")")
end

LOStructure(::Type{<:FroehlichPolaronND}) = IsHermitian()

starting_address(h::FroehlichPolaronND) = h.address

function dimension(h::FroehlichPolaronND, address)
    # takes into account `mode_cutoff` but not `momentum_cutoff`
    M = num_modes(address)
    n = h.mode_cutoff
    return BigInt(n + 1)^BigInt(M)
end

struct FroehlichPolaronNDColumn{H,A} <: AbstractOperatorColumn{A,Float64,H}
    hamiltonian::H
    address::A
    num_offdiagonals::Int
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
    return h.omega * Nphonon + ek
end


struct FroehlichPolaronNDOffdiagonals{A,H} <: AbstractVector{Pair{A,Float64}}
    address::A
    h::H
    num_offdiagonals::Int
end

offdiagonals(col::FroehlichPolaronNDColumn) = FroehlichPolaronNDOffdiagonals(col.address, col.hamiltonian, col.num_offdiagonals)


Base.size(ods::FroehlichPolaronNDOffdiagonals) = (ods.num_offdiagonals,)


@inline function calc_vk(h::FroehlichPolaronND, kidx::Int)
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

@inline function phonon_op(h::FroehlichPolaronND, addr, chosen)
    M = num_modes(addr)
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
    

    if h.momentum_cutoff != nothing
        occ = onr(new_addr)
        phononmom = zeros(h.d)
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


function Base.iterate(ods::FroehlichPolaronNDOffdiagonals, state::Int=1)
    state > ods.num_offdiagonals && return nothing
    return ods[state], state + 1
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