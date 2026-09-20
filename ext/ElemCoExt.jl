module ElemCoExt

import Rimu: MolecularHamiltonian, FermiFS, FermiFS2C, near_uniform, num_modes_check_equal
import ElemCo.FciDumps: FDump, QFDump, read_fcidump, headvar

function MolecularHamiltonian(
    fd::QFDump;
    starting_address::Union{Nothing,FermiFS2C}=nothing,
    specifier::String="",
)
    if isempty(fd)
        throw(ArgumentError("invalid input FCIDUMP file"))
    end
    n_orb = headvar(fd, "NORB", Int)
    n_elec = headvar(fd, "NELEC", Int)
    ms2 = headvar(fd, "MS2", Int)

    if isnothing(n_orb) || isnothing(n_elec) || isnothing(ms2)
        throw(
            ArgumentError(
                "input FCIDUMP file must have `NORB`, `NELEC`, and `MS2` defined in header"
            ),
        )
    end

    n_alpha_elec = (n_elec + ms2) ÷ 2
    n_beta_elec = (n_elec - ms2) ÷ 2

    if starting_address === nothing
        starting_address = FermiFS2C(
            near_uniform(FermiFS{n_alpha_elec,n_orb}),
            near_uniform(FermiFS{n_beta_elec,n_orb}),
        )
    end

    if num_modes_check_equal(starting_address) != n_orb
        throw(
            ArgumentError(
                "starting_address must have the same number of orbital as the FCIDUMP."
            ),
        )
    end

    val_type = typeof(fd.int0)
    addr_type = typeof(starting_address)
    fd_type = typeof(fd)
    return MolecularHamiltonian{val_type,addr_type,fd_type}(specifier, starting_address, fd)
end

end
