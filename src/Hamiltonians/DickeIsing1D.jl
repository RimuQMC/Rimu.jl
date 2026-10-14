"""
    DickeIsing1D(address::CompositeFS(BoseFS{missing,1}, HardcoreBoseFS{missing,M}); kwargs...) <: AbstractHamiltonian


The Dicke-Ising Hamiltonian with a single bosonic cavity mode and `M` spins represented as  
hardcore boson modes. The address is a [`CompositeFS`](@ref) with two components, a 
[`BoseFS`](@ref) and a [`HardcoreBoseFS`](@ref) with `M` modes.  

The Hamiltonian is given by

```math
H = ωâ^†â + (λ/√N)∑_{j=1}^M(â^† + â)(σ_j^+ + σ_j^-) - h∑_{j=1}^Mσ_j^z - J∑_{j=1}^Mσ_j^y σ_{j+1}^y  
```

where ``ω`` is the frequency of the cavity mode, ``λ`` is the coupling strength between the cavity 
and the two-level system, ``â`` and ``â^†`` are the annihilation and creation operators for the cavity mode, 
and ``σ_+`` and ``σ_-`` are the raising and lowering operators for the spins of the two-level systems. 
``σ_z`` is the angular momentum operator along the z-axis. ``σ_y`` is the angular momentum operator along 
the y-axis.

# Keyword Arguments

* `lambda=1.0`: the coupling strength ``λ``.
* `omegacav=1.0`: the frequency of the cavity mode ``ω``.
* `hfield=0.0`: the external magnetic field ``h``.
* `J=0.0`: the Ising interaction strength ``J``.

# Examples
```jldoctest
julia> fs = CompositeFS(BoseFS{missing}(0), HardcoreBoseFS{missing}(1,0,0))
CompositeFS(
  BoseFS{missing}(0,),
  HardcoreBoseFS{missing}(1, 0, 0),
)

julia> ham = DickeIsing1D(fs; lambda=1.0)
DickeIsing1D(fs"|0⟩{} ⊗ |●∘∘⟩{}"; lambda=1.0, omegacav=1.0, hfield=0.0, J=0.0)

julia> dimension(ham)
2048
```"""
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
        throw(ArgumentError("Cavity frequency must be non-negative"))
    end

    if hfield < 0
        throw(ArgumentError("External field must be non-negative"))
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

struct DickeIsing1DColumn{T,H<:DickeIsing1D{T},A<:CompositeFS} <: AbstractOperatorColumn{A,T,H}
    hamiltonian::H
    address::A
    num_offdiagonals::Int
    diagonal::T
end

parent_operator(column::DickeIsing1DColumn) = column.hamiltonian
starting_address(column::DickeIsing1DColumn) = column.address
diagonal_element(column::DickeIsing1DColumn) = column.diagonal
num_offdiagonals(column::DickeIsing1DColumn) = column.num_offdiagonals

struct DickeIsing1DOffDiagonals{T,A<:CompositeFS,H<:DickeIsing1D{T}}
    hamiltonian::H
    address::A
    num_offdiagonals::Int
end

Base.length(ods::DickeIsing1DOffDiagonals) = ods.num_offdiagonals
Base.size(ods::DickeIsing1DOffDiagonals) = (ods.num_offdiagonals,)
Base.eltype(::DickeIsing1DOffDiagonals{T,A,H}) where {T,A,H} = Pair{A,T}

@inline function Base.iterate(ods::DickeIsing1DOffDiagonals, state=1)
    if state > ods.num_offdiagonals
        return nothing
    end
    new_address, value = get_offdiagonal(ods.hamiltonian, ods.address, state)
    return new_address => value, state + 1
end

function Base.getindex(ods::DickeIsing1DOffDiagonals, index)
    return get_offdiagonal(ods.hamiltonian, ods.address, index)
end

function operator_column(h::DickeIsing1D{T}, addr::CompositeFS) where {T}
    offs = 6 * num_modes(addr.components[2]) + 2 * num_occupied_modes(addr.components[2])
    return DickeIsing1DColumn{T,typeof(h),typeof(addr)}(h, addr, offs, convert(T, diagonal_element(h, addr)))
end

function offdiagonals(column::DickeIsing1DColumn)
    return DickeIsing1DOffDiagonals(column.hamiltonian, column.address, column.num_offdiagonals)
end

function random_offdiagonal(column::DickeIsing1DColumn)
    chosen = rand(1:column.num_offdiagonals)
    new_address, value = get_offdiagonal(column.hamiltonian, column.address, chosen)
    return new_address, 1.0 / column.num_offdiagonals, value
end

LOStructure(::Type{<:DickeIsing1D{<:Real}}) = IsHermitian()

function dimension(h::DickeIsing1D)
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
    return h.omegacav * n1 - 2 * h.hfield * (n_up - nsites / 2)
end

@inline function _get_offdiagonal_dickeising1D(h, addr::CompositeFS, chosen)
    cavity = addr.components[1]
    spins = addr.components[2]
    M=num_modes(spins)
    num_exchange_offdiagonals = 2 * num_occupied_modes(spins)

    if chosen ≤ M  # a†σ+
        naddress1, value1 = excitation(cavity, (1,), ())
        i = find_mode(spins, (chosen))
        naddress2, value2=excitation(spins, (i,), ())
        valueJ=0.0

    elseif M < chosen ≤ 2*M  # a†σ-
        naddress1, value1 = excitation(cavity, (1,), ())
        i = find_mode(spins, (chosen-M))
        naddress2, value2=excitation(spins, (), (i,))
        valueJ=0.0

    elseif 2*M < chosen ≤ 3*M  # aσ+
        naddress1, value1 = excitation(cavity, (), (1,))
        i = find_mode(spins, (chosen-2*M))
        naddress2, value2=excitation(spins, (i,), ())
        valueJ=0.0

    elseif 3*M < chosen ≤ 4*M   # aσ-
        naddress1, value1 = excitation(cavity, (), (1,))
        i = find_mode(spins, (chosen-3*M))
        naddress2, value2=excitation(spins, (), (i,))
        valueJ=0.0
    
    #σ_j^y σ_{j+1}^y 
    elseif 4*M < chosen ≤ 4*M + num_exchange_offdiagonals 
        naddress1, value1 = cavity, 1.0
        value2 = 0.0
        naddress2, valueJ = hopnextneighbour(spins, chosen - 4*M)

    elseif 4*M + num_exchange_offdiagonals < chosen ≤ 5*M + num_exchange_offdiagonals
        naddress1, value1 = cavity, 1.0
        bond = chosen - 4*M - num_exchange_offdiagonals
        next_bond = bond == M ? 1 : bond + 1
        value2 = 0.0
        i, j = find_mode(spins, (bond, next_bond))
        naddress2, valueI = excitation(spins, (i, j), ())
        valueJ = -valueI

    elseif 5*M + num_exchange_offdiagonals < chosen ≤ 6*M + num_exchange_offdiagonals
        naddress1, value1 = cavity, 1.0
        bond = chosen - 5*M - num_exchange_offdiagonals
        next_bond = bond == M ? 1 : bond + 1
        i, j = find_mode(spins, (bond, next_bond))
        value2 = 0.0
        naddress2, valueI = excitation(spins, (), (i, j))
        valueJ = -valueI
    else
        throw(ArgumentError("Invalid chosen value"))
    end
    naddress = CompositeFS(naddress1, naddress2)
    return naddress, value1 * value2 * h.lambda / sqrt(M) - h.J * valueJ
end

function get_offdiagonal(h::DickeIsing1D, addr::CompositeFS, chosen)
    return _get_offdiagonal_dickeising1D(h, addr, chosen)
end

function num_offdiagonals(h::DickeIsing1D, addr::CompositeFS)
    M = num_modes(h.addr.components[2])
    return 6 * M + 2 * num_occupied_modes(addr.components[2])
end