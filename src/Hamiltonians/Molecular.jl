"""
    MolecularMolecularHamiltonian(fcidump::String)
    
    MolecularMolecularHamiltonian(fd::ElemCo.FciDumps.FDump)

Implements a Molecular Hamiltonian with ElemCo.jl.
"""
struct MolecularHamiltonian{T,A<:FermiFS2C,D<:FDump} <: AbstractHamiltonian{T}
    starting_address::A
    fcidump::D
end

function MolecularHamiltonian(fcidump::String, starting_address::Union{Nothing,FermiFS2C}=nothing)
    fd = read_fcidump(fcidump, Val(4))
    MolecularHamiltonian(fd, starting_address)
end

function MolecularHamiltonian(fd::QFDump, starting_address::Union{Nothing,FermiFS2C}=nothing)
    n_orb = headvar(fd, "NORB")
    n_elec = headvar(fd, "NELEC")
    ms2 = headvar(fd, "MS2")

    n_alpha_elec = (n_elec + ms2) ÷ 2
    n_beta_elec = (n_elec - ms2) ÷ 2

    if starting_address === nothing
        starting_address = FermiFS2C(near_uniform(FermiFS{n_alpha_elec,n_orb}), near_uniform(FermiFS{n_beta_elec,n_orb}))
    end

    if num_modes(starting_address) != n_orb
        throw(ArgumentError("starting_address provided must have same orbital numbers with FCIDUMP"))
    end

    val_type = typeof(fd.int0)
    addr_type = typeof(starting_address)
    fd_type = typeof(fd)
    MolecularHamiltonian{val_type,addr_type,fd_type}(starting_address, fd)
end

struct Modes{OA,OB,UA,UB,T<:FermiFSIndex}
    occupied::Tuple{ModeMap{OA,T},ModeMap{OB,T}}
    unoccupied::Tuple{ModeMap{UA,T},ModeMap{UB,T}}
end

function modes_extract(addr::FermiFS2C)
    occupied_modes = (occupied_mode_map(addr.components[1]), occupied_mode_map(addr.components[2]))
    unoccupied_modes = (unoccupied_mode_map(addr.components[1]), unoccupied_mode_map(addr.components[2]))
    Modes(occupied_modes, unoccupied_modes)
end

struct MolecularHamiltonianOperatorColumn{A<:FermiFS2C,T,O<:MolecularHamiltonian{T,A},M<:Modes} <: AbstractOperatorColumn{A,T,O}
    address::A # Contains address Psi_i, provided by operator_column
    op::O   # Represent Hamiltonian itself, provided by operator_column
    diag::T # < Psi_i | Psi_i >, calculated by diagonal_element
    modes::M # Store the modes of addr
end

function starting_address(h::MolecularHamiltonian)
    return h.starting_address
end

function operator_column(h::MolecularHamiltonian{T,A,D}, a::A)::MolecularHamiltonianOperatorColumn where {T<:Number,A<:FermiFS2C,D}
    modes = modes_extract(a)

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
    return MolecularHamiltonianOffDiagonalsIterator(column)
end

function random_offdiagonal(column::MolecularHamiltonianOperatorColumn)
    ods = offdiagonals(column)
    r = rand(collect(ods))
    return r[1], 1 / length(ods), r[2]
end

# function random_offdiagonal(column::MolecularHamiltonianOperatorColumn)
#     return rand(column.ods), 1 / length(column.ods)
# end

function one_electron_integral(int1::Array{T,2}, occ_modes::Tuple{AbstractVector{FermiFSIndex},AbstractVector{FermiFSIndex}}) where {T<:Number}
    one_elec_int = zero(T)
    for occ_mode in occ_modes
        for i in occ_mode
            one_elec_int += int1[i.mode, i.mode]
        end
    end
    one_elec_int
end

function two_electron_integral(int2::Array{T,4}, occ_modes::Tuple{AbstractVector{FermiFSIndex},AbstractVector{FermiFSIndex}})::T where {T<:Number}
    two_elec_int = zero(T)

    sum_alpha_alpha = zero(T)
    for i in occ_modes[1]
        for j in occ_modes[1]
            if i.mode ≠ j.mode
                sum_alpha_alpha += int2[i.mode, j.mode, i.mode, j.mode] - int2[i.mode, j.mode, j.mode, i.mode]
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
                sum_beta_beta += int2[i.mode, j.mode, i.mode, j.mode] - int2[i.mode, j.mode, j.mode, i.mode]
            end
        end
    end
    two_elec_int += 0.5 * sum_beta_beta

    two_elec_int
end

@inline function flip_spin_chan(chan::Int)::Int
    if chan == 1
        return 2
    else
        chan == 2
        return 1
    end
end

struct MolecularHamiltonianOffDiagonalsIterator{T,A<:FermiFS2C,D,OA,OB,UA,UB,TI<:FermiFSIndex}
    address::A
    op::MolecularHamiltonian{T,A,D}
    modes::Modes{OA,OB,UA,UB,TI}
end

function MolecularHamiltonianOffDiagonalsIterator(
    addr::A, op::MolecularHamiltonian{T,A,D}, m::Modes{OA,OB,UA,UB,TI}
) where {T,A,D,OA,OB,UA,UB,TI}
    return MolecularHamiltonianOffDiagonalsIterator{T,A,D,OA,OB,UA,UB,TI}(addr, op, m)
end

function MolecularHamiltonianOffDiagonalsIterator(c::MolecularHamiltonianOperatorColumn)
    return MolecularHamiltonianOffDiagonalsIterator(c.address, c.op, c.modes)
end

function Base.eltype(
    ::MolecularHamiltonianOffDiagonalsIterator{T,A}
) where {T,A}
    return Tuple{A,T}
end

struct MolecularHamiltonianOffDiagonalsIteratorState
    n_excited::Tuple{Int,Int}
    from::Tuple{Int,Int}
    to::Tuple{Int,Int}
end

function is_void_state(s::MolecularHamiltonianOffDiagonalsIteratorState)
    if s.from == (0, 0) && s.to == (0, 0)
        return true
    end
    return false
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

function MolecularHamiltonianOffDiagonalsIteratorState(na::Int, nb::Int)
    if (na, nb) == (0, 1) || (na, nb) == (1, 0)
        return MolecularHamiltonianOffDiagonalsIteratorState((na, nb), (1, 0), (1, 0))
    elseif (na, nb) == (0, 2) || (na, nb) == (2, 0)
        return MolecularHamiltonianOffDiagonalsIteratorState((na, nb), (1, 2), (1, 2))
    elseif (na, nb) == (1, 1)
        return MolecularHamiltonianOffDiagonalsIteratorState((na, nb), (1, 1), (1, 1))
    else
        return MolecularHamiltonianOffDiagonalsIteratorState((na, nb), (0, 0), (0, 0))
    end
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator)
    initial = MolecularHamiltonianOffDiagonalsIteratorState(0, 1)
    return iterate(iter, initial)
end

function Base.iterate(
    iter::MolecularHamiltonianOffDiagonalsIterator,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    na, nb = state.n_excited
    if (na, nb) == (0, 1)
        if is_void_state(state)
            return iterate(iter, MolecularHamiltonianOffDiagonalsIteratorState(1, 0))
        end
        r = one_electron_excitation(2, iter.address, iter.op, iter.modes, state)
        nstate = one_electron_excitation_generator(iter, state)
        return r, nstate
    elseif (na, nb) == (1, 0)
        if is_void_state(state)
            return iterate(iter, MolecularHamiltonianOffDiagonalsIteratorState(0, 2))
        end
        r = one_electron_excitation(1, iter.address, iter.op, iter.modes, state)
        nstate = one_electron_excitation_generator(iter, state)
        return r, nstate
    elseif (na, nb) == (0, 2)
        if is_void_state(state)
            return iterate(iter, MolecularHamiltonianOffDiagonalsIteratorState(2, 0))
        end
        r = two_electron_excitation(2, iter.address, iter.op, iter.modes, state)
        nstate = two_electron_excitation_generator(iter, state)
        return r, nstate
    elseif (na, nb) == (2, 0)
        if is_void_state(state)
            return iterate(iter, MolecularHamiltonianOffDiagonalsIteratorState(1, 1))
        end
        r = two_electron_excitation(1, iter.address, iter.op, iter.modes, state)
        nstate = two_electron_excitation_generator(iter, state)
        return r, nstate
    elseif (na, nb) == (1, 1)
        if is_void_state(state)
            return nothing
        end
        r = one_one_electron_excitation(iter.address, iter.op, iter.modes, state)
        nstate = one_one_electron_excitation_generator(iter, state)
        return r, nstate
    end
end

function Base.length(iter::MolecularHamiltonianOffDiagonalsIterator)
    return length(iter.modes.occupied[1]) * length(iter.modes.unoccupied[1]) +
           length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2]) +
           binomial(length(iter.modes.unoccupied[1]), 2) +
           binomial(length(iter.modes.unoccupied[2]), 2) +
           (length(iter.modes.occupied[1]) * length(iter.modes.unoccupied[1]) *
            length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2]))
end

function two_electron_excitation_generator(
    iter::MolecularHamiltonianOffDiagonalsIterator,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    na, nb = state.n_excited
    chan = (na == 2) ? 1 : 2

    ii, ij = state.from[1], state.from[2]
    ik, il = state.to[1], state.to[2]

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
    op::MolecularHamiltonian{T,A,D},
    m::Modes,
    state::MolecularHamiltonianOffDiagonalsIteratorState
) where {T,A<:FermiFS2C,D}
    fixed_chan = flip_spin_chan(chan)

    ii, ij = state.from[1], state.from[2]
    ik, il = state.to[1], state.to[2]

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
    return naddr, two_body
end

function one_electron_excitation(
    chan::Int, addr::A, op::MolecularHamiltonian{T,A,D}, m::Modes,
    state::MolecularHamiltonianOffDiagonalsIteratorState
) where {T,A,D}
    # `addr` corresponds to the `mode`
    fixed_chan = flip_spin_chan(chan)
    ii = state.from[1]
    ij = state.to[1]

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
    for k in m.occupied[flip_spin_chan(chan)]
        two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode]
        # print("+ < $(j.mode), $(k.mode) | $(i.mode), $(k.mode) >")
    end
    interaction = sign * (one_body + two_body)

    if chan == 1
        naddr = FermiFS2C(new_address, addr.components[fixed_chan])
    elseif chan == 2
        naddr = FermiFS2C(addr.components[fixed_chan], new_address)
    end
    return naddr, interaction
end

function one_electron_excitation_generator(
    iter::MolecularHamiltonianOffDiagonalsIterator,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    na, nb = state.n_excited
    chan = (na == 1) ? 1 : 2

    ii = state.from[1]
    ij = state.to[1]

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

function one_one_electron_excitation_generator(
    iter::MolecularHamiltonianOffDiagonalsIterator,
    state::MolecularHamiltonianOffDiagonalsIteratorState
)
    ii, ik = state.from[1], state.to[1]
    ij, il = state.from[2], state.to[2]

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
    addr::A, op::MolecularHamiltonian{T,A,D}, m::Modes,
    state::MolecularHamiltonianOffDiagonalsIteratorState
) where {T,A,D}
    ii, ij = state.from
    ik, il = state.to

    i, k = m.occupied[1][ii], m.unoccupied[1][ik]
    j, l = m.occupied[2][ij], m.unoccupied[2][il]
    new_address_alpha, sign_alpha = excitation(addr.components[1], (k,), (i,))
    new_address_beta, sign_beta = excitation(addr.components[2], (l,), (j,))
    interaction = (sign_alpha * sign_beta) * op.fcidump.int2[k.mode, l.mode, i.mode, j.mode]

    return FermiFS2C(new_address_alpha, new_address_beta), interaction
end