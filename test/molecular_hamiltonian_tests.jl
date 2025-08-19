using Rimu
using ElemCo
using Test
using Rimu.InterfaceTests: test_observable_interface, test_operator_interface,
    test_hamiltonian_interface, test_hamiltonian_structure


@testset "MolecularHamitonian" begin
    molecules = ("h2", "h2o")
    error = 1e-10
    @testset "Basic interface functionality" begin
        fcidump = joinpath(@__DIR__, "examples/h2.FCIDUMP")
        h = MolecularHamiltonian(fcidump)
        test_hamiltonian_interface(h)
        test_hamiltonian_structure(h)
    end

    @testset "User provided starting_address" begin
        fcidump = joinpath(@__DIR__, "examples/h2.FCIDUMP")

        @testset "Ill FermiFC2C" begin
            ill_addr = FermiFS2C(near_uniform(FermiFS{0,1}), near_uniform(FermiFS{0,1}))
            @test_throws ArgumentError MolecularHamiltonian(fcidump, ill_addr)
        end
        @testset "Normal FermiFC2C" begin
            normal_addr = FermiFS2C(near_uniform(FermiFS{1,2}), near_uniform(FermiFS{1,2}))
            h = MolecularHamiltonian(fcidump, normal_addr)
            @test starting_address(h) == normal_addr
        end
    end

    @testset "Hartree-Fock ground state energy" begin
        for molecule in molecules
            fcidump = joinpath(@__DIR__, "examples/$(molecule).FCIDUMP")
            ref_hf_ground_energy = @bohf
            h = MolecularHamiltonian(fcidump)
            a = starting_address(h)
            c = operator_column(h, a)
            @test diagonal_element(c) ≈ ref_hf_ground_energy atol = error
        end
    end

    @testset "Off-diagonal states and energies for H2 Molecule" begin
        h2_fci_matrix = [
            -1.117008268 0 0 0.1809125853;
            0 -0.3437220243 0.1809125853 0;
            0 0.1809125853 -0.3437220243 0;
            0.1809125853 0 0 0.4746278719
        ]
        fcidump = joinpath(@__DIR__, "examples/h2.FCIDUMP")
        h = MolecularHamiltonian(fcidump)
        bsr = BasisSetRepresentation(h, sort=true)
        @test bsr.sparse_matrix ≈ h2_fci_matrix atol = error
        p = ExactDiagonalizationProblem(h)
        s = solve(p)
        @test s.values[1] ≈ -1.137312593210905 atol = error
    end

    @testset "Off-diagonal term generation" begin
        excitation_type_map = ((0, 1), (1, 0), (0, 2), (2, 0), (1, 1))
        @testset "Corner cases: no enough electrons" begin
            fcidump = joinpath(@__DIR__, "examples/h2.FCIDUMP")
            h = MolecularHamiltonian(fcidump)
            a = starting_address(h)
            c = operator_column(h, a)
            ods = offdiagonals(c)
            excitation_type_indices = (3, 4)
            for t in excitation_type_indices
                ref_next_state = Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                    (1, 1), (0, 0), (0, 0)
                )
                _, next_state = iterate(ods,
                    Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                        excitation_type_map[t]...
                    ))
                @test ref_next_state == next_state
            end
        end

        @testset "Corner cases: boundary" begin
            fcidump = joinpath(@__DIR__, "examples/h2_dual.FCIDUMP")
            h = MolecularHamiltonian(fcidump)
            a = starting_address(h)
            c = operator_column(h, a)
            ods = offdiagonals(c)

            boundary_states = (
                Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                    excitation_type_map[1],
                    (length(c.modes.occupied[2]), 0),
                    (length(c.modes.unoccupied[2]), 0)
                ),
                Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                    excitation_type_map[2],
                    (length(c.modes.occupied[1]), 0),
                    (length(c.modes.unoccupied[1]), 0)
                ),
                Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                    excitation_type_map[3],
                    (length(c.modes.occupied[2]) - 1, length(c.modes.occupied[2])),
                    (length(c.modes.unoccupied[2]) - 1, length(c.modes.unoccupied[2]))
                ),
                Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                    excitation_type_map[4],
                    (length(c.modes.occupied[1]) - 1, length(c.modes.occupied[1])),
                    (length(c.modes.unoccupied[1]) - 1, length(c.modes.unoccupied[1]))
                ),
                Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                    excitation_type_map[5],
                    (length(c.modes.occupied[1]), length(c.modes.occupied[2])),
                    (length(c.modes.unoccupied[1]), length(c.modes.unoccupied[2]))
                ),
            )

            for (i, t) in enumerate(excitation_type_map)
                _, next_state = iterate(ods, boundary_states[i])
                ref_next_state = Rimu.Hamiltonians.MolecularHamiltonianOffDiagonalsIteratorState(
                    excitation_type_map[i], (0, 0), (0, 0)
                )
                @test next_state == ref_next_state
            end
        end
    end

    @testset "Off-diagonal matrix terms for dual H2 Molecules" begin
        struct ModeTransition
            from::Int64
            to::Int64
        end

        function ref_num_offdiagonals(column::MolecularHamiltonianOperatorColumn)
            n_orb = num_modes(column.address.components[1])
            n_alpha_elec = num_particles(column.address.components[1])
            n_beta_elec = num_particles(column.address.components[2])

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

        function init_offdiagonals(addr::A, op::MolecularHamiltonian{T,A,D}, modes::Rimu.Hamiltonians.Modes) where {T,A<:Rimu.FermiFS2C,D}
            states = Tuple{A,T}[]
            alpha_one_electron = one_electron_excitation_state(1, addr, op, modes)
            beta_one_electron = one_electron_excitation_state(2, addr, op, modes)

            alpha_two_electron = two_electron_excitation_state(1, addr, op, modes)
            beta_two_electron = two_electron_excitation_state(2, addr, op, modes)

            for i in beta_one_electron
                push!(states, (FermiFS2C(addr.components[1], i[1]), i[2]))
            end

            for i in alpha_one_electron
                push!(states, (FermiFS2C(i[1], addr.components[2]), i[2]))
            end

            for i in beta_two_electron
                push!(states, (FermiFS2C(addr.components[1], i[1]), i[2]))
            end

            for i in alpha_two_electron
                push!(states, (FermiFS2C(i[1], addr.components[2]), i[2]))
            end

            for i in alpha_one_electron
                for j in beta_one_electron
                    if (i[4] * j[4] > 0)
                        # alpha_to, beta_to, alpha_from, beta_from
                        interaction = op.fcidump.int2[i[3].to, j[3].to, i[3].from, j[3].from]
                    else
                        interaction = -op.fcidump.int2[i[3].to, j[3].to, i[3].from, j[3].from]
                    end
                    push!(states, (FermiFS2C(i[1], j[1]), interaction))
                end
            end

            states
        end

        function one_electron_excitation_state(chan::Int, addr::FermiFS2C, op::MolecularHamiltonian{T,A,D}, modes::Rimu.Hamiltonians.Modes) where {T,A,D}
            # `addr` corresponds to the `mode`
            new_addresses = Tuple{FermiFS,T,ModeTransition,T}[]
            for i in modes.occupied[chan]
                for j in modes.unoccupied[chan]
                    new_address, sign = excitation(addr.components[chan], (j,), (i,))
                    # print(interaction, " ")
                    one_body = op.fcidump.int1[j.mode, i.mode]
                    two_body = zero(T)
                    for k in modes.occupied[chan]
                        if k.mode ≠ i.mode
                            two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode] - op.fcidump.int2[j.mode, k.mode, k.mode, i.mode]
                            # print("+ <$(j.mode), $(k.mode) ||  $(i.mode), $(k.mode)>")
                        end
                    end
                    for k in modes.occupied[Rimu.Hamiltonians.flip_spin_chan(chan)]
                        two_body += op.fcidump.int2[j.mode, k.mode, i.mode, k.mode]
                        # print("+ < $(j.mode), $(k.mode) | $(i.mode), $(k.mode) >")
                    end
                    interaction = sign * (one_body + two_body)
                    push!(new_addresses, (new_address, interaction, ModeTransition(i.mode, j.mode), sign))
                    # println()
                end
            end
            new_addresses
        end

        function two_electron_excitation_state(chan::Int, addr::FermiFS2C, op::MolecularHamiltonian{T,A,D}, modes::Rimu.Hamiltonians.Modes) where {T,A,D}
            new_addresses = Tuple{FermiFS,T}[]
            for i in modes.occupied[chan]
                for j in modes.occupied[chan]
                    for a in modes.unoccupied[chan]
                        for b in modes.unoccupied[chan]
                            if i.mode < j.mode && a.mode < b.mode
                                tmp_address, interaction1 = excitation(addr.components[chan], (a,), (i,))
                                new_address, interaction2 = excitation(tmp_address, (b,), (j,))
                                two_body = op.fcidump.int2[a.mode, b.mode, i.mode, j.mode] - op.fcidump.int2[a.mode, b.mode, j.mode, i.mode]
                                if interaction1 * interaction2 > 0
                                    push!(new_addresses, (new_address, two_body))
                                else
                                    push!(new_addresses, (new_address, -two_body))
                                end
                            end
                        end
                    end
                end
            end
            new_addresses
        end

        fcidump = joinpath(@__DIR__, "examples/h2_dual.FCIDUMP")
        h = MolecularHamiltonian(fcidump)
        a = starting_address(h)
        modes = Rimu.Hamiltonians.modes_extract(a)
        ref_ods = init_offdiagonals(a, h, modes)
        c = operator_column(h, a)
        ods = offdiagonals(c)

        @testset "Off-diagonal term number test" begin
            @test ref_num_offdiagonals(c) == length(ods)
        end

        @testset "Off-diagonal term test" begin
            @test collect(ods) == ref_ods
        end
    end
end