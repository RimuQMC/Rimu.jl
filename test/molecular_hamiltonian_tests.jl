using Rimu
using ElemCo
using Test
using Rimu.InterfaceTests: test_observable_interface, test_operator_interface,
    test_hamiltonian_interface, test_hamiltonian_structure


@testset "MolecularHamitonian" begin
    molecules = ("h2", "h2o")
    error = 1e-10
    @testset "Basic interface functionality" begin
        fcidump = string(@__DIR__, "/examples/h2.FCIDUMP")
        h = MolecularHamiltonian(fcidump)
        test_hamiltonian_interface(h)
        test_hamiltonian_structure(h)
    end

    @testset "Hartree-Fock ground state energy" begin
        for molecule in molecules
            fcidump = string(@__DIR__, "/examples/$(molecule).FCIDUMP")
            ref_hf_ground_energy = @bohf
            h = MolecularHamiltonian(fcidump)
            a = starting_address(h)
            c = operator_column(h, a)
            @test diagonal_element(c) ≈ ref_hf_ground_energy atol = error
        end
    end

    @testset "Off-diagonal states and energies" begin
        h2_fci_matrix = [
            -1.117008268 0 0 0.1809125853;
            0 -0.3437220243 0.1809125853 0;
            0 0.1809125853 -0.3437220243 0;
            0.1809125853 0 0 0.4746278719
        ]
        fcidump = string(@__DIR__, "/examples/h2.FCIDUMP")
        h = MolecularHamiltonian(fcidump)
        bsr = BasisSetRepresentation(h, sort=true)
        @test bsr.sparse_matrix ≈ h2_fci_matrix atol = error
        p = ExactDiagonalizationProblem(h)
        s = solve(p)
        @test s.values[1] ≈ -1.137312593210905 atol = error
    end
end