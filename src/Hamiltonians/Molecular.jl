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
    return Iterators.flatten(get_offdiagonal_iterators(column.address, column.op, column.modes))
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

struct ModeIndex{N}
    index::NTuple{N,Int64}
end

function ModeIndex{0}()
    return ModeIndex{0}(())
end

function ModeIndex{1}()
    return ModeIndex{}((1,))
end

function ModeIndex{2}()
    return ModeIndex{}((1, 2))
end

struct MolecularHamiltonianOffDiagonalsIterator{NA,NB,T,A<:FermiFS2C,D,OA,OB,UA,UB,TI<:FermiFSIndex}
    address::A
    op::MolecularHamiltonian{T,A,D}
    modes::Modes{OA,OB,UA,UB,TI}
end

function MolecularHamiltonianOffDiagonalsIterator(na::Int, nb::Int, addr::A, op::MolecularHamiltonian{T,A,D}, m::Modes{OA,OB,UA,UB,TI}) where {T,A,D,OA,OB,UA,UB,TI}
    return MolecularHamiltonianOffDiagonalsIterator{na,nb,T,A,D,OA,OB,UA,UB,TI}(addr, op, m)
end

function num_excitation(::MolecularHamiltonianOffDiagonalsIterator{NA,NB}) where {NA,NB}
    return (NA, NB)
end

struct MolecularHamiltonianOffDiagonalsIteratorState{NA,NB}
    alpha_from::ModeIndex{NA}
    alpha_to::ModeIndex{NA}
    beta_from::ModeIndex{NB}
    beta_to::ModeIndex{NB}
end

function MolecularHamiltonianOffDiagonalsIteratorState(na::Int, nb::Int)
    return MolecularHamiltonianOffDiagonalsIteratorState(ModeIndex{na}(), ModeIndex{na}(), ModeIndex{nb}(), ModeIndex{nb}())
end

function get_offdiagonal_iterators(addr::FermiFS2C, op::MolecularHamiltonian, m::Modes)
    return (MolecularHamiltonianOffDiagonalsIterator(0, 1, addr, op, m),
        MolecularHamiltonianOffDiagonalsIterator(0, 2, addr, op, m),
        MolecularHamiltonianOffDiagonalsIterator(1, 0, addr, op, m),
        MolecularHamiltonianOffDiagonalsIterator(1, 1, addr, op, m),
        MolecularHamiltonianOffDiagonalsIterator(2, 0, addr, op, m))
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{0,2})
    initial = MolecularHamiltonianOffDiagonalsIteratorState{0,2}(ModeIndex{0}(), ModeIndex{0}(), ModeIndex{2}(), ModeIndex{2}())
    return iterate(iter, initial)
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{0,2}, state::MolecularHamiltonianOffDiagonalsIteratorState{0,2})
    chan = 2

    addr, interaction, next_from, next_to = two_electron_excitation_state(chan, iter.address, iter.op, iter.modes, state.beta_from, state.beta_to)
    if isnothing(addr)
        return nothing
    else
        nstate = MolecularHamiltonianOffDiagonalsIteratorState{0,2}(state.alpha_from, state.alpha_to, next_from, next_to)
        naddr = FermiFS2C(iter.address.components[1], addr)
        return (naddr, interaction), nstate
    end
end

function Base.length(iter::MolecularHamiltonianOffDiagonalsIterator{0,2})
    return binomial(length(iter.modes.unoccupied[2]), 2)
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{2,0})
    initial = MolecularHamiltonianOffDiagonalsIteratorState{2,0}(ModeIndex{2}(), ModeIndex{2}(), ModeIndex{0}(), ModeIndex{0}())
    return iterate(iter, initial)
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{2,0}, state::MolecularHamiltonianOffDiagonalsIteratorState{2,0})
    chan = 1
    addr, interaction, next_from, next_to = two_electron_excitation_state(chan, iter.address, iter.op, iter.modes, state.alpha_from, state.alpha_to)
    if isnothing(addr)
        return nothing
    else
        nstate = MolecularHamiltonianOffDiagonalsIteratorState{2,0}(next_from, next_to, state.beta_from, state.beta_to)
        naddr = FermiFS2C(addr, iter.address.components[2])
        return (naddr, interaction), nstate
    end
end

function Base.length(iter::MolecularHamiltonianOffDiagonalsIterator{2,0})
    return binomial(length(iter.modes.unoccupied[1]), 2)
end

function two_electron_excitation_state(chan::Int, addr::A, op::MolecularHamiltonian{T,A,D}, m::Modes, from::ModeIndex{2}, to::ModeIndex{2}) where {T,A,D}
    ii, ij = from.index[1], from.index[2]
    ik, il = to.index[1], to.index[2]

    while ii <= length(m.occupied[chan])
        while ij <= length(m.occupied[chan])
            while ik <= length(m.unoccupied[chan])
                while il <= length(m.unoccupied[chan])
                    i, j = m.occupied[chan][ii], m.occupied[chan][ij]
                    k, l = m.unoccupied[chan][ik], m.unoccupied[chan][il]
                    naddr, interaction = excitation(addr.components[chan], (k, l), (j, i))
                    two_body = interaction * (op.fcidump.int2[k.mode, l.mode, i.mode, j.mode] - op.fcidump.int2[k.mode, l.mode, j.mode, i.mode])
                    il += 1
                    next_from = ModeIndex((ii, ij))
                    next_to = ModeIndex((ik, il))
                    return naddr, two_body, next_from, next_to
                end
                ik += 1
                il = ik + 1
            end
            ij += 1
            ik = 1
            il = ik + 1
        end
        ii += 1
        ij = ii + 1
        ik = 1
        il = ik + 1
    end
    return nothing, zero(T), ModeIndex{2}(), ModeIndex{2}()
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{0,1})
    initial = MolecularHamiltonianOffDiagonalsIteratorState{0,1}(ModeIndex{0}(), ModeIndex{0}(), ModeIndex{1}(), ModeIndex{1}())
    return iterate(iter, initial)
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{0,1}, state::MolecularHamiltonianOffDiagonalsIteratorState{0,1})
    chan = 2
    addr, interaction, next_from, next_to = one_electron_excitation_state(chan, iter.address, iter.op, iter.modes, state.beta_from, state.beta_to)
    if isnothing(addr)
        return nothing
    else
        nstate = MolecularHamiltonianOffDiagonalsIteratorState{0,1}(state.alpha_from, state.alpha_to, next_from, next_to)
        naddr = FermiFS2C(iter.address.components[1], addr)
        return (naddr, interaction), nstate
    end
end

function Base.length(iter::MolecularHamiltonianOffDiagonalsIterator{0,1})
    return length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2])
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{1,0})
    initial = MolecularHamiltonianOffDiagonalsIteratorState{1,0}(ModeIndex{1}(), ModeIndex{1}(), ModeIndex{0}(), ModeIndex{0}())
    return iterate(iter, initial)
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{1,0}, state::MolecularHamiltonianOffDiagonalsIteratorState{1,0})
    chan = 1
    addr, interaction, next_from, next_to = one_electron_excitation_state(chan, iter.address, iter.op, iter.modes, state.alpha_from, state.alpha_to)
    if isnothing(addr)
        return nothing
    else
        nstate = MolecularHamiltonianOffDiagonalsIteratorState{1,0}(next_from, next_to, state.beta_from, state.beta_to)
        naddr = FermiFS2C(addr, iter.address.components[2])
        return (naddr, interaction), nstate
    end
end

function Base.length(iter::MolecularHamiltonianOffDiagonalsIterator{1,0})
    return length(iter.modes.occupied[1]) * length(iter.modes.unoccupied[1])
end

function one_electron_excitation_state(chan::Int, addr::A, op::MolecularHamiltonian{T,A,D}, m::Modes, from::ModeIndex{1}, to::ModeIndex{1}) where {T,A,D}
    # `addr` corresponds to the `mode`
    ii = from.index[1]
    ij = to.index[1]

    while ii <= length(m.occupied[chan])
        while ij <= length(m.unoccupied[chan])
            i = m.occupied[chan][ii]
            j = m.unoccupied[chan][ij]
            new_address, sign = excitation(addr.components[chan], (j,), (i,))
            one_body = op.fcidump.int1[j.mode, i.mode]
            two_body = zero(T)
            for k in m.occupied[chan]
                if k.mode ≠ i.mode
                    two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode] - op.fcidump.int2[j.mode, k.mode, k.mode, i.mode]
                    # print("+ <$(j.mode), $(k.mode) ||  $(i.mode), $(k.mode)>")
                end
            end
            for k in m.occupied[flip_spin_chan(chan)]
                two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode]
                # print("+ < $(j.mode), $(k.mode) | $(i.mode), $(k.mode) >")
            end
            interaction = sign * (one_body + two_body)
            ij += 1
            next_from = ModeIndex((ii,))
            next_to = ModeIndex((ij,))
            return new_address, interaction, next_from, next_to
        end
        ii += 1
        ij = 1
    end
    return nothing, zero(T), ModeIndex{1}(), ModeIndex{1}()
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{1,1})
    initial = MolecularHamiltonianOffDiagonalsIteratorState{1,1}(ModeIndex{1}(), ModeIndex{1}(), ModeIndex{1}(), ModeIndex{1}())
    return iterate(iter, initial)
end

function Base.iterate(iter::MolecularHamiltonianOffDiagonalsIterator{1,1}, state::MolecularHamiltonianOffDiagonalsIteratorState{1,1})
    ii, ik = state.alpha_from.index[1], state.alpha_to.index[1]
    ij, il = state.beta_from.index[1], state.beta_to.index[1]

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
        return nothing
    end
    i, k = iter.modes.occupied[1][ii], iter.modes.unoccupied[1][ik]
    j, l = iter.modes.occupied[2][ij], iter.modes.unoccupied[2][il]
    new_address_alpha, sign_alpha = excitation(iter.address.components[1], (k,), (i,))
    new_address_beta, sign_beta = excitation(iter.address.components[2], (l,), (j,))
    interaction = (sign_alpha * sign_beta) * iter.op.fcidump.int2[k.mode, l.mode, i.mode, j.mode]

    il += 1
    nstate = MolecularHamiltonianOffDiagonalsIteratorState{1,1}(ModeIndex((ii,)), ModeIndex((ik,)), ModeIndex((ij,)), ModeIndex((il,)))
    return (FermiFS2C(new_address_alpha, new_address_beta), interaction), nstate
end

function Base.length(iter::MolecularHamiltonianOffDiagonalsIterator{1,1})
    return length(iter.modes.occupied[1]) * length(iter.modes.unoccupied[1]) * length(iter.modes.occupied[2]) * length(iter.modes.unoccupied[2])
end