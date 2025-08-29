"""
    MolecularMolecularHamiltonian(fcidump_path::String[, starting_address::FermiFS2C])
    
    MolecularMolecularHamiltonian(
        fd::ElemCo.FciDumps.FDump[, 
            starting_address::FermiFS2C,
            specifier::String,
        ]
    )

Implements an electronic ab-initio Hamiltonian based on electron overlap integrals in
FCIDUMP format. It can be used to describe the electronic structure of a molecule
with fixed nuclear positions (i.e. under the Born-Oppenheimer approximation).
The parse of FCIDUMP file depends on [ElemCo.jl](https://elem.co.il). 

The expression for the Hamiltonian is
```math
\\begin{aligned}
    Ĥ  & = h₀ + Ĥ₁ + Ĥ₂, \\\\
    Ĥ₁ & = \\sum_{i,j,σ} h_{ij}  a^†_{j,σ} a_{i,σ}, \\\\
    Ĥ₂ & = ½ \\sum_{i,j,σ,τ} V_{ijkl} a^†_{i,σ} a^†_{j,τ} a_{l,τ} a_{k,σ}.
\\end{aligned}
```
where the constants ``h₀``, ``h_{ij}``, and ``V_{ijkl}`` are provided by the FCIDUMP `fd`.
See the documentation of [ElemCo.jl](https://elem.co.il) for generating FCIDUMP information
with a Hartree-Fock calculation, or parsing them from a file.

# Arguments
* `fcidump_path`: The path to FCIDUMP file of molecular.
* `fd`: The FCIDUMP structure defined in [ElemCo.jl](https://elem.co.il).
* `starting_address`: The starting address, defines number of particles and sites.
* `specifier`: The string used to identify the Molecular Hamiltonian, it stores the path
    to FCIDUMP file or a user specified name.

See also [`FermiFS2C`](@ref)

!!! warning
    The FCIDUMP file is assumed to be a Restricted Hartree-Fock(RHF) Hamiltonian.
"""
struct MolecularHamiltonian{T,A<:FermiFS2C,D<:FDump} <: AbstractHamiltonian{T}
    specifier::String
    starting_address::A
    fcidump::D
end

function MolecularHamiltonian(fcidump_path::String, starting_address::Union{Nothing,FermiFS2C}=nothing, specifier::String="")
    fd = read_fcidump(fcidump_path, Val(4))
    MolecularHamiltonian(fd, starting_address, fcidump_path)
end

function MolecularHamiltonian(
    fd::QFDump,
    starting_address::Union{Nothing,FermiFS2C{}}=nothing,
    specifier::String="",
)
    if !fd_exists(fd)
        throw(ArgumentError("invalid input FCIDUMP file"))
    end
    n_orb = headvar(fd, "NORB", Int)
    n_elec = headvar(fd, "NELEC", Int)
    ms2 = headvar(fd, "MS2", Int)

    if isnothing(n_orb) || isnothing(n_elec) || isnothing(ms2)
        throw(ArgumentError("input FCIDUMP file must have `NORB`, `NELEC`, and `MS2` defined in header"))
    end

    n_alpha_elec = (n_elec + ms2) ÷ 2
    n_beta_elec = (n_elec - ms2) ÷ 2

    if starting_address === nothing
        starting_address = FermiFS2C(
            near_uniform(FermiFS{n_alpha_elec,n_orb}),
            near_uniform(FermiFS{n_beta_elec,n_orb})
        )
    end

    if num_modes(starting_address) != n_orb
        throw(ArgumentError("starting_address provided must have same orbital numbers with FCIDUMP"))
    end

    val_type = typeof(fd.int0)
    addr_type = typeof(starting_address)
    fd_type = typeof(fd)
    MolecularHamiltonian{val_type,addr_type,fd_type}(specifier, starting_address, fd)
end

function Base.show(io::IO, h::MolecularHamiltonian)
    io = IOContext(io, :compact => true)
    print(io, "MolecularHamiltonian\n")
    if h.specifier != ""
        print(io, "specifier: ", h.specifier, "\n")
    end
    print(io, "starting_addresss: ")
    show(io, h.starting_address)
end

struct MolecularHamiltonianOperatorColumn{A<:FermiFS2C,T,O<:MolecularHamiltonian{T,A},M<:FermiFS2CModes} <: AbstractOperatorColumn{A,T,O}
    address::A # Contains address Psi_i, provided by operator_column
    op::O   # Represent Hamiltonian itself, provided by operator_column
    diag::T # < Psi_i | Psi_i >, calculated by diagonal_element
    modes::M # Store the modes of address
end

function starting_address(h::MolecularHamiltonian)
    return h.starting_address
end

function operator_column(h::MolecularHamiltonian{T,A,D}, a::A)::MolecularHamiltonianOperatorColumn where {T<:Number,A<:FermiFS2C,D}
    modes = full_mode_maps(a)

    diag = zero(T)
    diag_one_elec_int = one_electron_integral(h.fcidump.int1, modes.occupied)
    diag_two_elec_int = two_electron_integral(h.fcidump.int2, modes.occupied)
    diag = h.fcidump.int0 + diag_one_elec_int + diag_two_elec_int

    MolecularHamiltonianOperatorColumn{A,T,typeof(h),typeof(modes)}(a, h, diag, modes)
end

parent_operator(c::MolecularHamiltonianOperatorColumn) = c.op
starting_address(c::MolecularHamiltonianOperatorColumn) = c.address

function diagonal_element(column::MolecularHamiltonianOperatorColumn{A,T,O,OD}) where {A<:FermiFS2C,T,O,OD}
    return column.diag
end

function num_offdiagonals(column::MolecularHamiltonianOperatorColumn)
    return length(offdiagonals(column))
end

function offdiagonals(column::MolecularHamiltonianOperatorColumn)
    return MolecularHamiltonianOffDiagonals(column)
end

function random_offdiagonal(column::MolecularHamiltonianOperatorColumn)
    ods = offdiagonals(column)
    l = length(ods)
    ri = rand(1:l)
    state = linear_to_state(ods, ri)
    r = iterate(ods, state)
    if isnothing(r)
        return (column.address, 0, 0)
    end
    return r[1][1], 1 / l, r[1][2]
end

function random_offdiagonal_old(column::MolecularHamiltonianOperatorColumn)
    ods = offdiagonals(column)
    r = rand(collect(ods))
    return r[1], 1 / length(ods), r[2]
end

"""
    one_electron_integral(
        int1::Array{T,2},
        occ_modes::Tuple{AbstractVector{FermiFSIndex},AbstractVector{FermiFSIndex}}
    )::T where {T<:Number}

Calculate the one body operator diagonal term ``\\langle U | \\hat{H}_1 | U \\rangle``.

```math
\\begin{aligned}
    \\langle U | \\hat{H}_1| U \\rangle & = 
    \\sum_{i=1}^{N} h_{ii} \\langle \\psi_i | a^{\\dagger}_{i} a_{i} | \\psi_j \\rangle  \\\\
    & = \\sum_{i=1}^{n_{\\alpha}} h_{ii} 
        \\langle \\varphi_i | a^{\\dagger}_{i} a_{i} | \\varphi_i \\rangle 
      + \\sum_{i=1}^{n_{\\beta}} h_{ii} 
        \\langle \\varphi_i | a^{\\dagger}_{i} a_{i} | \\varphi_i \\rangle.
\\end{aligned}
```
"""
function one_electron_integral(
    int1::Array{T,2},
    occ_modes::Tuple{AbstractVector{FermiFSIndex},AbstractVector{FermiFSIndex}}
)::T where {T<:Number}
    one_elec_int = zero(T)
    for occ_mode in occ_modes
        for i in occ_mode
            one_elec_int += int1[i.mode, i.mode]
        end
    end
    one_elec_int
end

"""
    two_electron_integral(
        int2::Array{T,4},
        occ_modes::Tuple{AbstractVector{FermiFSIndex},AbstractVector{FermiFSIndex}}
    )::T where {T<:Number}

Calculate the two body operator diagonal term ``\\langle U | \\hat{H}_2| U \\rangle`` 
with Slater-Condon rules.

```math
\\begin{aligned}
    \\langle U | \\hat{H}_2| U \\rangle 
        & = \\sum_{i<j}^{n_\\alpha} [\\langle ij|ij\\rangle - \\langle ij|ji\\rangle] \\\\
        & + \\sum_{i<j}^{n_\\beta} [\\langle ij|ij\\rangle - \\langle ij|ji\\rangle] \\\\
        & + \\sum_{i}^{n_\\alpha}\\sum_{j}^{n_\\beta} \\langle ij|ij\\rangle. \\\\
\\end{aligned}
```
"""
function two_electron_integral(
    int2::Array{T,4},
    occ_modes::Tuple{AbstractVector{FermiFSIndex},AbstractVector{FermiFSIndex}}
)::T where {T<:Number}
    two_elec_int = zero(T)

    sum_alpha_alpha = zero(T)
    for i in occ_modes[1]
        for j in occ_modes[1]
            if i.mode ≠ j.mode
                sum_alpha_alpha += int2[i.mode, j.mode, i.mode, j.mode] -
                                   int2[i.mode, j.mode, j.mode, i.mode]
            end
        end
    end
    two_elec_int += 0.5 * sum_alpha_alpha

    sum_alpha_beta = zero(T)
    for i in occ_modes[1]
        for j in occ_modes[2]
            sum_alpha_beta += int2[i.mode, j.mode, i.mode, j.mode]
        end
    end
    two_elec_int += sum_alpha_beta

    sum_beta_beta = zero(T)
    for i in occ_modes[2]
        for j in occ_modes[2]
            if i.mode ≠ j.mode
                sum_beta_beta += int2[i.mode, j.mode, i.mode, j.mode] -
                                 int2[i.mode, j.mode, j.mode, i.mode]
            end
        end
    end
    two_elec_int += 0.5 * sum_beta_beta

    two_elec_int
end

"""
    flip_spin_components(component::Int)::Int

This is an inline function used to flip the spin component index. 
"""
@inline flip_spin_components(component::Int)::Int = 3 - component

"""
This struct is used internally to represent the iterator for off-diagonal 
terms generation, which is the returned value type of [`operator_column`](@ref) 
function when applying on [`MolecularHamiltonian`](@ref).
"""
struct MolecularHamiltonianOffDiagonals{
    T,A<:FermiFS2C,H<:MolecularHamiltonian{T,A},M<:FermiFS2CModes
}
    address::A
    op::H
    modes::M
    ind_cap::Tuple{Int,Int,Int,Int,Int}
end

function MolecularHamiltonianOffDiagonals(
    c::MolecularHamiltonianOperatorColumn
)
    ind_cap_beta_1 = length(c.modes.occupied[2]) * length(c.modes.unoccupied[2])
    ind_cap_alpha_1 = ind_cap_beta_1 + length(c.modes.occupied[1]) * length(c.modes.unoccupied[1])
    ind_cap_beta_2 = ind_cap_alpha_1 + binomial(length(c.modes.occupied[2]), 2) *
                                       binomial(length(c.modes.unoccupied[2]), 2)
    ind_cap_alpha_2 = ind_cap_beta_2 + binomial(length(c.modes.occupied[1]), 2) *
                                       binomial(length(c.modes.unoccupied[1]), 2)
    ind_cap_alpha_1_beta_1 = ind_cap_alpha_2 + length(c.modes.occupied[1]) * length(c.modes.unoccupied[1]) *
                                               length(c.modes.occupied[2]) * length(c.modes.unoccupied[2])
    ind_cap = (ind_cap_beta_1, ind_cap_alpha_1, ind_cap_beta_2, ind_cap_alpha_2, ind_cap_alpha_1_beta_1)
    return MolecularHamiltonianOffDiagonals(c.address, c.op, c.modes, ind_cap)
end

function Base.eltype(
    ::MolecularHamiltonianOffDiagonals{T,A}
) where {T,A}
    return Pair{A,T}
end

"""
    MolecularHamiltonianOffDiagonalsIteratorState

This struct is used internally to represent the state during 
off-diagonal terms generation.

* `excitations_per_channel`: Tuple contains 2 values represents the how many electrons 
are excited in alpha(1) and beta(2) channel. 
* `from_occupieds`: Records the indices of `FermiFS2CModes` arrays which modes electrons 
being excited from.
* `to_unoccupieds`: Records the indices of `FermiFS2CModes` arrays which modes electrons 
being excited to.

Both `from_occupieds == (0,0)` and `to_unoccupieds == (0,0)` represent a special "void"
state when there is no more state in current `n_excited` situation, this is used to 
notify upper caller that it should move to next valiad `excitations_per_channel`. 
This is checked by the function [`is_void_state`](@ref).

When used to represents one-electron-excitation cases, only `from_occupieds[1]` 
and `to_unoccupieds[1]` are used. 
When used to represents two-electrons-excitation cases, `from_occupieds[1]`, 
`to_unoccupieds[1]`, `from_occupieds[2]` and `to_unoccupieds[2]` are all used. 
When used to represents one-one-electrons-excitation case, index `1` represention
alpha spin channel and index `2` represention beta spin channel repectively.

See also [`FermiFS2CModes`](@ref).
"""
struct MolecularHamiltonianOffDiagonalsIteratorState
    excitations_per_channel::Tuple{Int,Int} # either 0, 1, 2
    from_occupieds::Tuple{Int,Int} # indices into FermiFS2CModes arrays
    to_unoccupieds::Tuple{Int,Int} # indices into FermiFS2CModes arrays
end

"""
    is_void_state(s::MolecularHamiltonianOffDiagonalsIteratorState)

When `from_occupieds == (0,0)` and `to_unoccupieds == (0,0)` are both 
satisfied, the state is considered a void state.
"""
function is_void_state(s::MolecularHamiltonianOffDiagonalsIteratorState)
    if s.from_occupieds == (0, 0) && s.to_unoccupieds == (0, 0)
        return true
    end
    return false
end

"""
    is_invalid_state(
        iter::MolecularHamiltonianOffDiagonalsIterator, 
        s::MolecularHamiltonianOffDiagonalsIteratorState
    )

This function is used to check if `s` is a valid state. It checks if its field
`from_occupieds` and `to_unoccupieds` tuples hold the indices within the range 
of corresponding `FermiFS2CModes` array.
"""
function is_invalid_state(
    iter::MolecularHamiltonianOffDiagonals,
    s::MolecularHamiltonianOffDiagonalsIteratorState
)
    if s.excitations_per_channel == (0, 1)
        if ((1 <= s.from_occupieds[1] <= length(iter.modes.occupied[2]))
            &&
            (1 <= s.to_unoccupieds[1] <= length(iter.modes.unoccupied[2])))
            return false
        else
            return true
        end
    elseif s.excitations_per_channel == (1, 0)
        if ((1 <= s.from_occupieds[1] <= length(iter.modes.occupied[1]))
            &&
            (1 <= s.to_unoccupieds[1] <= length(iter.modes.unoccupied[1])))
            return false
        else
            return true
        end
    elseif s.excitations_per_channel == (0, 2)
        if ((1 <= s.from_occupieds[1] <= length(iter.modes.occupied[2]))
            &&
            (1 <= s.from_occupieds[2] <= length(iter.modes.occupied[2]))
            &&
            (1 <= s.to_unoccupieds[1] <= length(iter.modes.unoccupied[2]))
            &&
            (1 <= s.to_unoccupieds[2] <= length(iter.modes.unoccupied[2])))
            return false
        else
            return true
        end
    elseif s.excitations_per_channel == (2, 0)
        if ((1 <= s.from_occupieds[1] <= length(iter.modes.occupied[1]))
            &&
            (1 <= s.from_occupieds[2] <= length(iter.modes.occupied[1]))
            &&
            (1 <= s.to_unoccupieds[1] <= length(iter.modes.unoccupied[1]))
            &&
            (1 <= s.to_unoccupieds[2] <= length(iter.modes.unoccupied[1])))
            return false
        else
            return true
        end
    elseif s.excitations_per_channel == (1, 1)
        if ((1 <= s.from_occupieds[1] <= length(iter.modes.occupied[1]))
            &&
            (1 <= s.from_occupieds[2] <= length(iter.modes.occupied[2]))
            &&
            (1 <= s.to_unoccupieds[1] <= length(iter.modes.unoccupied[1]))
            &&
            (1 <= s.to_unoccupieds[2] <= length(iter.modes.unoccupied[2])))
            return false
        else
            return true
        end
    else
        return true
    end
end

function MolecularHamiltonianOffDiagonalsIteratorState(
    n_excited::Tuple{Int,Int}, ii::Int, ij::Int
)
    return MolecularHamiltonianOffDiagonalsIteratorState(
        n_excited, (ii, 0), (ij, 0)
    )
end

function MolecularHamiltonianOffDiagonalsIteratorState(
    n_excited::Tuple{Int,Int},
    ii::Int, ij::Int, ik::Int, il::Int
)
    return MolecularHamiltonianOffDiagonalsIteratorState(
        n_excited, (ii, ij), (ik, il)
    )
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonals)
    initial = MolecularHamiltonianOffDiagonalsIteratorState((0, 1), (1, 0), (1, 0))
    return iterate(iter, initial)
end

function Base.iterate(
    iter::MolecularHamiltonianOffDiagonals,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    na, nb = state.excitations_per_channel
    if (na, nb) == (0, 1) # single electron in beta channel
        if is_void_state(state) || is_invalid_state(iter, state)
            return iterate(
                iter, MolecularHamiltonianOffDiagonalsIteratorState((1, 0), (1, 0), (1, 0))
            )
        end
        r = one_electron_excitation(2, iter.address, iter.op, iter.modes, state)
        nstate = one_electron_excitation_next(iter, state)
        return r, nstate
    elseif (na, nb) == (1, 0) # single electron excitation in alpha channel
        if is_void_state(state) || is_invalid_state(iter, state)
            return iterate(
                iter, MolecularHamiltonianOffDiagonalsIteratorState((0, 2), (1, 2), (1, 2))
            )
        end
        r = one_electron_excitation(1, iter.address, iter.op, iter.modes, state)
        nstate = one_electron_excitation_next(iter, state)
        return r, nstate
    elseif (na, nb) == (0, 2) # 2 electron excitation in beta channel
        if is_void_state(state) || is_invalid_state(iter, state)
            return iterate(
                iter, MolecularHamiltonianOffDiagonalsIteratorState((2, 0), (1, 2), (1, 2))
            )
        end
        r = two_electron_excitation(2, iter.address, iter.op, iter.modes, state)
        nstate = two_electron_excitation_next(iter, state)
        return r, nstate
    elseif (na, nb) == (2, 0) # 2 electron excitation in alpha channel
        if is_void_state(state) || is_invalid_state(iter, state)
            return iterate(
                iter, MolecularHamiltonianOffDiagonalsIteratorState((1, 1), (1, 1), (1, 1))
            )
        end
        r = two_electron_excitation(1, iter.address, iter.op, iter.modes, state)
        nstate = two_electron_excitation_next(iter, state)
        return r, nstate
    elseif (na, nb) == (1, 1)
        if is_void_state(state) || is_invalid_state(iter, state)
            return nothing
        end
        r = one_one_electron_excitation(iter.address, iter.op, iter.modes, state)
        nstate = one_one_electron_excitation_next(iter, state)
        return r, nstate
    end
end

function Base.length(iter::MolecularHamiltonianOffDiagonals)
    return length(iter.modes.occupied[1]) * length(iter.modes.unoccupied[1]) + # 1-alpha
           length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2]) + # 1-beta
           binomial(length(iter.modes.occupied[1]), 2) * 
           binomial(length(iter.modes.unoccupied[1]), 2) + # 2-alpha
           binomial(length(iter.modes.occupied[2]), 2) * 
           binomial(length(iter.modes.unoccupied[2]), 2) + # 2-beta
           (length(iter.modes.occupied[1]) * length(iter.modes.unoccupied[1]) *
            length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2])) # 1-alpha, 1-beta
end

function two_electron_excitation_next(
    iter::MolecularHamiltonianOffDiagonals,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    na, nb = state.excitations_per_channel
    chan = (na == 2) ? 1 : 2

    ii, ij = state.from_occupieds[1], state.from_occupieds[2]
    ik, il = state.to_unoccupieds[1], state.to_unoccupieds[2]

    il += 1
    if il > length(iter.modes.unoccupied[chan])
        ik += 1
        il = ik + 1
    end
    if ik > length(iter.modes.unoccupied[chan]) - 1
        ij += 1
        ik = 1
        il = ik + 1
    end
    if ij > length(iter.modes.occupied[chan])
        ii += 1
        ij = ii + 1
        ik = 1
        il = ik + 1
    end
    if ii > length(iter.modes.occupied[chan]) - 1
        return MolecularHamiltonianOffDiagonalsIteratorState((na, nb), 0, 0, 0, 0)
    end
    return MolecularHamiltonianOffDiagonalsIteratorState(
        (na, nb),
        ii, ij, ik, il
    )
end

function two_electron_excitation(
    chan::Int,
    addr::A,
    op::MolecularHamiltonian{T,A},
    m::FermiFS2CModes,
    state::MolecularHamiltonianOffDiagonalsIteratorState
) where {T,A<:FermiFS2C}
    fixed_chan = flip_spin_components(chan)

    ii, ij = state.from_occupieds[1], state.from_occupieds[2]
    ik, il = state.to_unoccupieds[1], state.to_unoccupieds[2]

    i, j = m.occupied[chan][ii], m.occupied[chan][ij]
    k, l = m.unoccupied[chan][ik], m.unoccupied[chan][il]
    new_address, sign = excitation(addr.components[chan], (k, l), (j, i))
    two_body = sign * (
        op.fcidump.int2[k.mode, l.mode, i.mode, j.mode]
        -
        op.fcidump.int2[k.mode, l.mode, j.mode, i.mode]
    )
    if chan == 1
        naddr = FermiFS2C(new_address, addr.components[fixed_chan])
    elseif chan == 2
        naddr = FermiFS2C(addr.components[fixed_chan], new_address)
    end
    return (naddr => two_body)
end

function one_electron_excitation_next(
    iter::MolecularHamiltonianOffDiagonals,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    na, nb = state.excitations_per_channel
    chan = (na == 1) ? 1 : 2

    ii = state.from_occupieds[1]
    ij = state.to_unoccupieds[1]

    ij += 1
    if ij > length(iter.modes.unoccupied[chan])
        ii += 1
        ij = 1
    end
    if ii > length(iter.modes.occupied[chan])
        return MolecularHamiltonianOffDiagonalsIteratorState((na, nb), 0, 0)
    end
    return MolecularHamiltonianOffDiagonalsIteratorState((na, nb), ii, ij)
end

function one_electron_excitation(
    chan::Int, addr::A, op::MolecularHamiltonian{T,A}, m::FermiFS2CModes,
    state::MolecularHamiltonianOffDiagonalsIteratorState
) where {T,A}
    # `addr` corresponds to the `mode`
    fixed_chan = flip_spin_components(chan)
    ii = state.from_occupieds[1]
    ij = state.to_unoccupieds[1]

    i = m.occupied[chan][ii]
    j = m.unoccupied[chan][ij]

    new_address, sign = excitation(addr.components[chan], (j,), (i,))
    one_body = op.fcidump.int1[j.mode, i.mode]
    two_body = zero(T)
    for k in m.occupied[chan]
        if k.mode ≠ i.mode
            two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode] -
                        op.fcidump.int2[j.mode, k.mode, k.mode, i.mode]
            # print("+ <$(j.mode), $(k.mode) ||  $(i.mode), $(k.mode)>")
        end
    end
    for k in m.occupied[flip_spin_components(chan)]
        two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode]
        # print("+ < $(j.mode), $(k.mode) | $(i.mode), $(k.mode) >")
    end
    interaction = sign * (one_body + two_body)

    if chan == 1
        naddr = FermiFS2C(new_address, addr.components[fixed_chan])
    elseif chan == 2
        naddr = FermiFS2C(addr.components[fixed_chan], new_address)
    end
    return (naddr => interaction)
end

function one_one_electron_excitation_next(
    iter::MolecularHamiltonianOffDiagonals,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    ii, ik = state.from_occupieds[1], state.to_unoccupieds[1]
    ij, il = state.from_occupieds[2], state.to_unoccupieds[2]

    il += 1
    if il > length(iter.modes.unoccupied[2])
        ij += 1
        il = 1
    end
    if ij > length(iter.modes.occupied[2])
        ik += 1
        ij = 1
    end
    if ik > length(iter.modes.unoccupied[1])
        ii += 1
        ik = 1
    end
    if ii > length(iter.modes.occupied[1])
        return MolecularHamiltonianOffDiagonalsIteratorState((1, 1), 0, 0, 0, 0)
    end
    nstate = MolecularHamiltonianOffDiagonalsIteratorState((1, 1), ii, ij, ik, il)
    return nstate
end

function one_one_electron_excitation(
    addr::A, op::MolecularHamiltonian{T,A,D}, m::FermiFS2CModes,
    state::MolecularHamiltonianOffDiagonalsIteratorState
) where {T,A,D}
    ii, ij = state.from_occupieds
    ik, il = state.to_unoccupieds

    i, k = m.occupied[1][ii], m.unoccupied[1][ik]
    j, l = m.occupied[2][ij], m.unoccupied[2][il]
    new_address_alpha, sign_alpha = excitation(addr.components[1], (k,), (i,))
    new_address_beta, sign_beta = excitation(addr.components[2], (l,), (j,))
    interaction = (sign_alpha * sign_beta) * op.fcidump.int2[k.mode, l.mode, i.mode, j.mode]

    return (FermiFS2C(new_address_alpha, new_address_beta) => interaction)
end


"""
    linear_to_state(
        iter::MolecularHamiltonianOffDiagonals, iter_index::Int
    )::MolecularHamiltonianOffDiagonalsIteratorState

This function takes `iter` and `iter_index` as arguments.
The iterator `iter` produces a sequence of `state` values in a fixed order,
establishing a bijection between indices and states. Given an `iter_index`,
the function generates the corresponding `state` by performing the following steps:

1. Identify excitation type
- As iter is traversed, it outputs states belonging to different excitation types
  in a specific order.
- Since the number of states for each excitation type is known in advance,
  we can determine which excitation type corresponds to the given `iter_index`
  by checking the index range it falls into.

2. Locate the state within the excitation type
- Once the excitation type is identified, compute the offset associated with that type.
- Using this offset, determine the exact state corresponding to the provided `iter_index`.
"""
function linear_to_state(
    iter::MolecularHamiltonianOffDiagonals, iter_index::Int
)::MolecularHamiltonianOffDiagonalsIteratorState
    if 1 <= iter_index <= iter.ind_cap[1] # 1 electron excitation in beta channel
        iter_offset = iter_index - 1
        from = iter_offset ÷ length(iter.modes.unoccupied[2]) + 1
        to = iter_offset % length(iter.modes.unoccupied[2]) + 1
        return MolecularHamiltonianOffDiagonalsIteratorState((0, 1), (from, 0), (to, 0))
    elseif iter.ind_cap[1] + 1 <= iter_index <= iter.ind_cap[2] # 1 electron excitation in alpha channel
        iter_offset = iter_index - iter.ind_cap[1] - 1
        from = iter_offset ÷ length(iter.modes.unoccupied[1]) + 1
        to = iter_offset % length(iter.modes.unoccupied[1]) + 1
        return MolecularHamiltonianOffDiagonalsIteratorState((1, 0), (from, 0), (to, 0))
    elseif iter.ind_cap[2] + 1 <= iter_index <= iter.ind_cap[3] # 2 electron excitation in beta channel
        iter_offset = iter_index - iter.ind_cap[2] - 1
        from = iter_offset ÷ binomial(length(iter.modes.unoccupied[2]), 2) + 1
        to = iter_offset % binomial(length(iter.modes.unoccupied[2]), 2) + 1
        from_tuple = unrank_combination(length(iter.modes.occupied[2]), from)
        to_tuple = unrank_combination(length(iter.modes.unoccupied[2]), to)
        return MolecularHamiltonianOffDiagonalsIteratorState((0, 2), from_tuple, to_tuple)
    elseif iter.ind_cap[3] + 1 <= iter_index <= iter.ind_cap[4] # 2 electron excitation in alpha channel
        iter_offset = iter_index - iter.ind_cap[3] - 1
        from = iter_offset ÷ binomial(length(iter.modes.unoccupied[1]), 2) + 1
        to = iter_offset % binomial(length(iter.modes.unoccupied[1]), 2) + 1
        from_tuple = unrank_combination(length(iter.modes.occupied[1]), from)
        to_tuple = unrank_combination(length(iter.modes.unoccupied[1]), to)
        return MolecularHamiltonianOffDiagonalsIteratorState((2, 0), from_tuple, to_tuple)
    elseif iter.ind_cap[4] + 1 <= iter_index <= iter.ind_cap[5] # 1 electron excitation in both channels
        iter_offset = iter_index - iter.ind_cap[4] - 1
        alpha_from = iter_offset ÷ (
            length(iter.modes.unoccupied[1]) * length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2])
        )
        iter_offset -= alpha_from * (
            length(iter.modes.unoccupied[1]) * length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2])
        )
        alpha_to = iter_offset ÷ (length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2]))
        iter_offset -= alpha_to * (length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2]))
        beta_from = iter_offset ÷ length(iter.modes.unoccupied[2])
        beta_to = iter_offset % length(iter.modes.unoccupied[2])
        return MolecularHamiltonianOffDiagonalsIteratorState(
            (1, 1), (alpha_from + 1, beta_from + 1), (alpha_to + 1, beta_to + 1)
        )
    end
    return MolecularHamiltonianOffDiagonalsIteratorState((1, 1), (0, 0), (0, 0))
end

"""
    unrank_combination(n::Int, i::Int)

Return the 2-element combination (in lex order) of set {1,...,n} with index i.
"""
function unrank_combination(n::Int, i::Int)
    idx = i - 1
    x = 0
    y = 0
    start = 1
    remaining = 2

    # Determine the 1st position in the combination and store it in x
    for j in start:n
        # Count the number of combinations exist if we fix 1st position to j
        count = binomial(n - j, remaining - 1)
        if idx < count
            x = j
            start = j + 1
            remaining -= 1
            break
        else
            idx -= count
        end
    end

    # Determine the 2nd position in the combination and store it in y
    for j in start:n
        count = binomial(n - j, remaining - 1)
        if idx < count
            y = j
            start = j + 1
            remaining -= 1
            break
        else
            idx -= count
        end
    end
    return (x, y)
end