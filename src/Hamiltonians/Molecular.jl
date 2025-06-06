import ElemCo.FciDumps: FDump, QFDump, read_fcidump

"""
    MolecularMolecularHamiltonian(fcidump::String)
    
    MolecularMolecularHamiltonian(fd::ElemCo.FciDumps.FDump)

Implements a Molecular Hamiltonian with ElemCo.jl.
"""
struct MolecularHamiltonian{T,A<:Union{FermiFS,FermiFS2C},D<:FDump} <: AbstractHamiltonian{T}
    starting_address::A
    fcidump::D
end

function MolecularHamiltonian(fcidump::String)
    fd = read_fcidump(fcidump, Val(4))
    MolecularHamiltonian(fd)
end

function MolecularHamiltonian(fd::QFDump)
    n_orb = fd.head["NORB"][1]
    n_elec = fd.head["NELEC"][1]
    if fd.uhf
        n_alpha_elec = n_elec ÷ 2
        starting_address = FermiFS2C(near_uniform(FermiFS{n_alpha_elec,n_orb}), near_uniform(FermiFS{n_alpha_elec,n_orb}))
        MolecularHamiltonian{Float64,FermiFS2C,QFDump}(starting_address, fd)
    else
        starting_address = near_uniform(FermiFS{n_elec,n_orb})
        MolecularHamiltonian{Float64,FermiFS,QFDump}(starting_address, fd)
    end
end

struct MolecularHamiltonianOperatorColumn{A<:Union{FermiFS,FermiFS2C},T,O<:MolecularHamiltonian,OD} <: AbstractOperatorColumn{A,T,O}
    addr::A # Contains address Psi_i, provided by operator_column
    op::O   # Represent Hamiltonian itself, provided by operator_column
    diag::T # < Psi_i | Psi_i >, calculated by diagonal_element
    ods::OD # Vector [ < Psi_j | Psi_i > ], calculated by offdiagonals
end

function starting_address(h::MolecularHamiltonian)
    return h.starting_address
end

function operator_column(h::MolecularHamiltonian, a::AbstractFockAddress)::MolecularHamiltonianOperatorColumn

end

function diagonal_element(column::MolecularHamiltonianOperatorColumn{A<:FermiFS,T,O,OD})

end

function diagonal_element(column::MolecularHamiltonianOperatorColumn{A<:FermiFS2C,T,O,OD})
    
end

function num_offdiagonals(column::MolecularHamiltonianOperatorColumn)

end

function offdiagonals(column::MolecularHamiltonianOperatorColumn)

end

function random_offdiagonal(column::MolecularHamiltonianOperatorColumn)

end