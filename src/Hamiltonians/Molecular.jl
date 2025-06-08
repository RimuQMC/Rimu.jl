import ElemCo.FciDumps: FDump, QFDump, read_fcidump, headvar

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
    n_alpha_elec = n_elec ÷ 2
    starting_address = FermiFS2C(near_uniform(FermiFS{n_alpha_elec,n_orb}), near_uniform(FermiFS{n_alpha_elec,n_orb}))

    MolecularHamiltonian{Float64,FermiFS2C,QFDump}(starting_address, fd)
end

struct MolecularHamiltonianOperatorColumn{A<:FermiFS2C,T,O<:MolecularHamiltonian,OD} <: AbstractOperatorColumn{A,T,O}
    addr::A # Contains address Psi_i, provided by operator_column
    op::O   # Represent Hamiltonian itself, provided by operator_column
    diag::T # < Psi_i | Psi_i >, calculated by diagonal_element
    ods::OD # Vector [ < Psi_j | Psi_i > ], calculated by offdiagonals
end

function starting_address(h::MolecularHamiltonian)
    return h.starting_address
end

function operator_column(h::MolecularHamiltonian, a::FermiFS2C)::MolecularHamiltonianOperatorColumn
    diag = 0.0::Float64
    ods = Vector{Float64}[]
    MolecularHamiltonianOperatorColumn(a, h, diag, ods)
end

function diagonal_element(column::MolecularHamiltonianOperatorColumn{A,T,O,OD}) where {A<:FermiFS2C,T,O,OD}
    one_elec_int = one_electron_integral(column.op.fcidump.int1, column.addr)
    two_elec_int = two_electron_integral(column.op.fcidump.int2, column.addr)
    one_elec_int + two_elec_int
end

# For testing purpose
function diagonal_element(h::MolecularHamiltonian, a::FermiFS2C)
    one_elec_int = one_electron_integral(h.fcidump.int1, a)
    two_elec_int = two_electron_integral(h.fcidump.int2, a)
    one_elec_int + two_elec_int
end

function num_offdiagonals(column::MolecularHamiltonianOperatorColumn)::Int
    n_orb = headvar(column.op.fcidump, "NORB")
    n_elec = headvar(column.op.fcidump, "NELEC")
    binomial(2 * n_orb, n_elec) - 1
end

function offdiagonals(column::MolecularHamiltonianOperatorColumn)

end

function random_offdiagonal(column::MolecularHamiltonianOperatorColumn)

end

function one_electron_integral(int1::Array{T,2}, addr::FermiFS2C) where {T<:Number}
    one_elec_int = 0.0::T
    n_comp = num_components(addr)
    for c = 1:n_comp
        occ_map = OccupiedModeMap(addr.components[c])
        for occ in occ_map
            one_elec_int += int1[occ.mode, occ.mode]
        end
    end
    one_elec_int
end

function two_electron_integral(int2::Array{T,4}, addr::FermiFS2C)::T where {T<:Number}
    two_elec_int = 0.0::T
    occ_map_c1 = OccupiedModeMap(addr.components[1])
    occ_map_c2 = OccupiedModeMap(addr.components[2])

    sum_alpha_alpha = 0.0::T
    for i in occ_map_c1
        for j in occ_map_c1
            if i.mode ≠ j.mode
                sum_alpha_alpha += int2[i.mode, j.mode, i.mode, j.mode] - int2[i.mode, j.mode, j.mode, i.mode]
            end
        end
    end
    two_elec_int += 0.5 * sum_alpha_alpha

    sum_alpha_beta = 0.0::T
    for i in occ_map_c1
        for j in occ_map_c2
            sum_alpha_beta += int2[i.mode, j.mode, i.mode, j.mode]
        end
    end
    two_elec_int += sum_alpha_beta

    sum_beta_beta = 0.0::T
    for i in occ_map_c2
        for j in occ_map_c2
            if i.mode ≠ j.mode
                sum_beta_beta += int2[i.mode, j.mode, i.mode, j.mode] - int2[i.mode, j.mode, j.mode, i.mode]
            end
        end
    end
    two_elec_int += 0.5 * sum_beta_beta

    two_elec_int
end

function one_electron_integral(int1::Array{T,2}, addr1::FermiFS2C, addr2::FermiFS2C) where {T<:Number}

end

function two_electron_integral(int1::Array{T,2}, addr1::FermiFS2C, addr2::FermiFS2C) where {T<:Number}

end