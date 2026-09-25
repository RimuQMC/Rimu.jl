"""
    DickeIsing1D(address::CompositeFS{BoseFS{missing,1}, HardcoreBoseFS{missing,N}}; kwargs...) <: AbstractHamiltonian


The Dicke-Ising Hamiltonian where the cavity mode is represented as a bosonic mode with the spins as a hardcore bosonic mode. The address is a [`CompositeFS`](@ref) with two components, a [`BoseFS`](@ref) and a [`HardcoreBoseFS`](@ref), representing the occupation of the cavity and the two-level system.
representing the occupation of the cavity and the spin system.

The Hamiltonian is given by

```math
H = ωâ†â + (λ/√N)∑(â† + â)(σ+ + σ-) - h∑(σz) - J∑σyσy+1
```

where ``ω0`` is the splitting frequency of the two-level system, ``ω`` is the frequency of the cavity mode, 
``λ`` is the coupling strength between the cavity and the two-level system, ``â`` and ``â†`` are the 
annihilation and creation operators for the cavity mode, and ``σ+`` and ``σ-`` are the raising and 
lowering operators for the spins of the two-level systems. ``σz`` is the angular momentum operator along 
the z-axis. ``σy`` is the angular momentum operator along the y-axis.

# Keyword Arguments

* `lambda=1.0`: the coupling strength ``λ``.
* `omegacav=1.0`: the frequency of the cavity mode ``ω``.
* `hfield=0.0`: the external magnetic field ``h``.
* `J=0.0`: the Ising interaction strength ``J``.
"""
struct DickeIsing1D{T} <: AbstractHamiltonian{T}
    addr::CompositeFS
    lambda::T
    omegacav::T
    hfield::T
    J::T
end

function DickeIsing1D(
    addr::CompositeFS;
    lambda=1.0,
    omegacav=1.0,
    hfield=0.0,
    J=0.0,
)
    if omegacav < 0
        throw(ArgumentError("Cavity frequency must be positive"))
    end

    if hfield < 0
        throw(ArgumentError("External field must be positive"))
    end

    comp1, comp2 = addr.components
    comp1 isa BoseFS{missing,1} || throw(ArgumentError(
        "first component must be BoseFS{missing,1}"
    ))
    comp2 isa HardcoreBoseFS{missing,<:Any} || throw(ArgumentError(
        "second component must be HardcoreBoseFS{missing,N}"
    ))
    

    lambda, omegacav, hfield, J = promote(float(lambda), float(omegacav), float(hfield), float(J))
    return DickeIsing1D{typeof(lambda)}(addr, lambda, omegacav, hfield, J)
end

function Base.show(io::IO, h::DickeIsing1D)
    compact_addr = repr(h.addr, context=:compact => true) # compact print address
    print(io, "DickeIsing1D($compact_addr; ")
    print(io, "lambda=$(h.lambda), omegacav=$(h.omegacav), hfield=$(h.hfield), J=$(h.J))")
end

function starting_address(h::DickeIsing1D)
    return h.addr
end

LOStructure(::Type{<:DickeIsing1D{<:Real}}) = IsHermitian()

function dimension(h::DickeIsing1D, address)
    starting = h.addr
    d1 = maximum_mode_occupation(starting.components[1]) # number of photons in the cavity mode
    d2 = num_modes(starting.components[2]) # number of angular momentum states
    return BigInt((2^(BigInt(d2))*BigInt(d1 + 1)))
end

@inline function diagonal_element(h::DickeIsing1D, addr::CompositeFS)
    n1 = num_particles(addr.components[1])
    spins = onr(addr.components[2])
    n_up = sum(spins)
    nsites = length(spins)
    return h.omegacav * n1 - 2*h.hfield * (n_up - nsites / 2) 
end

function _get_offdiagonal_dickeising1D(h, addr::CompositeFS, chosen)
    cavity = addr.components[1]
    spins = addr.components[2]
    N=num_modes(spins)

    if chosen ≤ N  # a†σ+
        naddress1, value1 = excitation(cavity, (1,), ())
        i = find_mode(spins, (chosen))
        naddress2, value2=excitation(spins, (i,), ())
        valueJ=0.0

    elseif N < chosen ≤ 2*N  # a†σ-
        naddress1, value1 = excitation(cavity, (1,), ())
        i = find_mode(spins, (chosen-N))
        naddress2, value2=excitation(spins, (), (i,))
        valueJ=0.0

    elseif 2*N < chosen ≤ 3*N   # aσ+
        naddress1, value1 = excitation(cavity, (), (1,))
        i = find_mode(spins, (chosen-2*N))
        naddress2, value2=excitation(spins, (i,), ())
        valueJ=0.0

    elseif 3*N < chosen ≤ 4*N   # aσ-
        naddress1, value1 = excitation(cavity, (), (1,))
        i = find_mode(spins, (chosen-3*N))
        naddress2, value2=excitation(spins, (), (i,))
        valueJ=0.0

    elseif 4*N < chosen ≤ 5*N   # σ+iσ-i+1
        naddress1, value1 = cavity, 1.0
        bond = chosen - 4*N
        next_bond = bond == N ? 1 : bond + 1
        i,j = find_mode(spins, (bond,next_bond))
        value2=0.0
        naddress2, valueJ=excitation(spins, (i,), (j,))

    elseif 5*N < chosen ≤ 6*N   # σ-iσ+i+1
        naddress1, value1 = cavity, 1.0
        bond = chosen - 5*N
        next_bond = bond == N ? 1 : bond + 1
        i,j = find_mode(spins, (bond,next_bond))
        value2=0.0
        naddress2, valueJ=excitation(spins, (j,), (i,))
    
    elseif 6*N < chosen ≤ 7*N   # σ+iσ+i+1
        naddress1, value1 = cavity, 1.0
        bond = chosen - 6*N
        next_bond = bond == N ? 1 : bond + 1
        i,j = find_mode(spins, (bond,next_bond))
        value2=0.0
        naddress2, valueI=excitation(spins, (i,j,), (),)
        valueJ=-valueI

    elseif 7*N < chosen ≤ 8*N   # σ-iσ-i+1
        naddress1, value1 = cavity, 1.0
        bond = chosen - 7*N
        next_bond = bond == N ? 1 : bond + 1
        i,j = find_mode(spins, (bond,next_bond))
        value2=0.0
        naddress2, valueI=excitation(spins, (), (i,j,),)
        valueJ=-valueI
    else
        throw(ArgumentError("Invalid chosen value"))
    end
    naddress = CompositeFS(naddress1, naddress2)
    return naddress, value1*value2*h.lambda/sqrt(N) -h.J*valueJ
end

function get_offdiagonal(h::DickeIsing1D, addr::CompositeFS, chosen)
    return _get_offdiagonal_dickeising1D(h, addr, chosen)
end

function num_offdiagonals(h::DickeIsing1D, addr::CompositeFS)
    N = num_modes(addr.components[2])
    return 8 * N
end
