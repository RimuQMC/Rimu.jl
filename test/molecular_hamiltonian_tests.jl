using Rimu
using ElemCo
using Test
using Rimu.InterfaceTests: test_observable_interface, test_operator_interface,
    test_hamiltonian_interface, test_hamiltonian_structure

@testset "MolecularHamitonian" begin
    fcidump = string(@__DIR__, "/examples/h2o.FCIDUMP")
    ref_hf_ground_energy = @bohf
    h = MolecularHamiltonian(fcidump)
    test_hamiltonian_interface(h)
    a = starting_address(h)
    c = operator_column(h, a)
    @test diagonal_element(c) ≈ ref_hf_ground_energy
end