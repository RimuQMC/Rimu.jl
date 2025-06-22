"""
    MolecularMolecularHamiltonian(fcidump::String)
    
    MolecularMolecularHamiltonian(fd::ElemCo.FciDumps.FDump)

Implements a Molecular Hamiltonian with ElemCo.jl.
"""
struct MolecularHamiltonian{T,A<:FermiFS2C,D<:FDump} <: AbstractHamiltonian{T}
    starting_address::A
    fcidump::D
end

function MolecularHamiltonian(fcidump::String)
    fd = read_fcidump(fcidump, Val(4))
    MolecularHamiltonian(fd)
end

function MolecularHamiltonian(fd::QFDump)
    n_orb = headvar(fd, "NORB")
    n_elec = headvar(fd, "NELEC")
    ms2 = headvar(fd, "MS2")

    n_alpha_elec = (n_elec + ms2) ÷ 2
    n_beta_elec = (n_elec - ms2) ÷ 2

    starting_address = FermiFS2C(near_uniform(FermiFS{n_alpha_elec,n_orb}), near_uniform(FermiFS{n_beta_elec,n_orb}))

    val_type = typeof(fd.int0)
    addr_type = typeof(starting_address)
    fd_type = typeof(fd)
    MolecularHamiltonian{val_type,addr_type,fd_type}(starting_address, fd)
end

struct Modes{OA,OB,UA,UB,T<:FermiFSIndex}
    occupied::Tuple{OccupiedModeMap{OA,T},OccupiedModeMap{OB,T}}
    unoccupied::Tuple{OccupiedModeMap{UA,T},OccupiedModeMap{UB,T}}
end

function modes_extract(addr::FermiFS2C)
    occupied_modes = (OccupiedModeMap(addr.components[1]), OccupiedModeMap(addr.components[2]))
    unoccupied_modes = (UnoccupiedModeMap(addr.components[1]), UnoccupiedModeMap(addr.components[2]))
    Modes(occupied_modes, unoccupied_modes)
end

struct MolecularHamiltonianOperatorColumn{A<:FermiFS2C,T,O<:MolecularHamiltonian{T,A},OD,M<:Modes} <: AbstractOperatorColumn{A,T,O}
    addr::A # Contains address Psi_i, provided by operator_column
    op::O   # Represent Hamiltonian itself, provided by operator_column
    diag::T # < Psi_i | Psi_i >, calculated by diagonal_element
    ods::OD # Vector [ < Psi_j | Psi_i > ], calculated by offdiagonals
    modes::M # Store the modes of addr
end

function starting_address(h::MolecularHamiltonian)
    return h.starting_address
end

function operator_column(h::MolecularHamiltonian{T,A,D}, a::FermiFS2C)::MolecularHamiltonianOperatorColumn where {T<:Number,A,D}
    modes = modes_extract(a)

    diag = zero(T)
    diag_one_elec_int = one_electron_integral(h.fcidump.int1, modes.occupied)
    diag_two_elec_int = two_electron_integral(h.fcidump.int2, modes.occupied)
    diag = h.fcidump.int0 + diag_one_elec_int + diag_two_elec_int

    ods = init_offdiagonals(a, h, modes)

    MolecularHamiltonianOperatorColumn{typeof(a),T,typeof(h),typeof(ods),typeof(modes)}(a, h, diag, ods, modes)
end

function diagonal_element(column::MolecularHamiltonianOperatorColumn{A,T,O,OD}) where {A<:FermiFS2C,T,O,OD}
    return column.diag
end

function num_offdiagonals(column::MolecularHamiltonianOperatorColumn)
    return length(column.ods)
end

function ref_num_offdiagonals(column::MolecularHamiltonianOperatorColumn)
    n_orb = num_modes(column.addr.components[1])
    n_alpha_elec = num_occupied_modes(column.addr.components[1])
    n_beta_elec = num_occupied_modes(column.addr.components[2])

    n_alpha_hole = n_orb - n_alpha_elec
    n_beta_hole = n_orb - n_beta_elec

    # One-electron excitation
    n_one_electron_excitation = n_alpha_elec * n_alpha_hole + n_beta_elec * n_beta_hole

    # Two-electron excitation
    n_two_electron_excitation =
        binomial(n_alpha_elec, 2) * binomial(n_alpha_hole, 2) +
        binomial(n_beta_elec, 2) * binomial(n_beta_hole, 2) +
        (n_alpha_elec * n_alpha_hole) * (n_beta_elec * n_beta_hole)

    n_one_electron_excitation + n_two_electron_excitation
end

function offdiagonals(column::MolecularHamiltonianOperatorColumn)
    return column.ods
end

function init_offdiagonals(addr::FermiFS2C, op::MolecularHamiltonian{T,A,D}, modes::Modes) where {T,A,D}
    states = Tuple{FermiFS2C,T}[]
    alpha_one_electron = one_electron_excitation_state(1, addr, op, modes)
    beta_one_electron = one_electron_excitation_state(2, addr, op, modes)

    alpha_two_electron = two_electron_excitation_state(1, addr, op, modes)
    beta_two_electron = two_electron_excitation_state(2, addr, op, modes)

    for i in alpha_one_electron
        push!(states, (FermiFS2C(i[1], addr.components[2]), i[2]))
    end

    for i in beta_one_electron
        push!(states, (FermiFS2C(addr.components[1], i[1]), i[2]))
    end

    for i in alpha_two_electron
        push!(states, (FermiFS2C(i[1], addr.components[2]), i[2]))
    end

    for i in beta_two_electron
        push!(states, (FermiFS2C(addr.components[1], i[1]), i[2]))
    end

    for i in alpha_one_electron
        for j in beta_one_electron
            push!(states, (FermiFS2C(i[1], j[1]), i[2] * j[2]))
        end
    end

    states
end

function random_offdiagonal(column::MolecularHamiltonianOperatorColumn)
    return rand(column.ods)
end

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


function one_electron_excitation_state(chan::Int, addr::FermiFS2C, op::MolecularHamiltonian{T,A,D}, modes::Modes) where {T,A,D}
    new_addresses = Tuple{FermiFS,T}[]
    for i in modes.occupied[chan]
        for j in modes.unoccupied[chan]
            new_address, interaction = excitation(addr.components[chan], (j,), (i,))
            one_body = op.fcidump.int1[j.mode, i.mode]
            two_body = zero(T)
            for k in OccupiedModeMap(new_address)
                two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode] - op.fcidump.int2[j.mode, k.mode, k.mode, i.mode]
            end
            interaction *= one_body + two_body
            push!(new_addresses, (new_address, interaction))
        end
    end
    new_addresses
end

function two_electron_excitation_state(chan::Int, addr::FermiFS2C, op::MolecularHamiltonian{T,A,D}, modes::Modes) where {T,A,D}
    new_addresses = Tuple{FermiFS,T}[]
    for i in modes.occupied[chan]
        for j in modes.occupied[chan]
            for a in modes.unoccupied[chan]
                for b in modes.unoccupied[chan]
                    if i.mode < j.mode && a.mode < b.mode
                        new_address, interaction = excitation(addr.components[chan], (a, b), (i, j))
                        two_body = op.fcidump.int2[a.mode, b.mode, i.mode, j.mode] - op.fcidump.int2[a.mode, b.mode, j.mode, i.mode]
                        interaction *= two_body
                        push!(new_addresses, (new_address, interaction))
                    end
                end
            end
        end
    end
    new_addresses
end

const ModeIndex = @NamedTuple{ch::Int, orb::Int}

struct ModeTransition
    old::ModeIndex
    new::ModeIndex
end

mutable struct MolecularHamiltonianOffDiagonalsIterState
    occ_ind::Int
    unocc_ind::Int
    spin_ch::Int
end

struct MolecularHamiltonianOffDiagonals
    modes::Modes{FermiFSIndex}
    transition::Vector{ModeTransition}
end

function Base.iterate(mhod::MolecularHamiltonianOffDiagonals, state::MolecularHamiltonianOffDiagonalsIterState=MolecularHamiltonianOffDiagonalsIterState(1, 1, 1))
    # Only includes one-electron exctitation
    if state.spin_ch == 3
        return nothing
    end
    cur_trans = ModeTransition(
        (state.spin_ch, mhod.modes.occupied[state.spin_ch][state.occ_ind].mode),
        (state.spin_ch, mhod.modes.unoccupied[state.spin_ch][state.unocc_ind].mode)
    )
    println(cur_trans)
    if state.unocc_ind < length(mhod.modes.unoccupied[state.spin_ch])
        state.unocc_ind += 1
    else
        state.unocc_ind = 1
        if state.occ_ind < length(mhod.modes.occupied[state.spin_ch])
            state.occ_ind += 1
        else
            state.occ_ind = 1
            if state.spin_ch <= 2
                state.spin_ch += 1
            end
        end
    end
    return MolecularHamiltonianOffDiagonals(mhod.modes, [cur_trans]), state
end