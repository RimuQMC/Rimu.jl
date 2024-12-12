using Rimu
using Rimu: SingleState
using Rimu.Interfaces: apply_operator!
using Test

"""
    apply_operator_wrap!(r::SingleState)

Returning the vectors is tracked as an allocation. This wrapper takes care of that.
"""
function apply_operator_wrap!(r::SingleState)
    transition = Rimu.FirstOrderTransitionOperator(r.shift_parameters, r.hamiltonian)
    apply_operator!(r.wm, r.pv, r.v, transition)
    return nothing
end

@testset "Allocations" begin
    # The purpose of these tests is to find type instabilities that might appear as the
    # Julia compiler changes. If allocations suddenly increase by a lot, there is a good
    # chance that dynamic dispatch is happening somewhere in the code.
    b1 = near_uniform(BoseFS{10,10})
    b2 = near_uniform(BoseFS{50,50})
    b3 = near_uniform(BoseFS{100,100})

    f1 = near_uniform(FermiFS{9,10})
    f2 = near_uniform(FermiFS{24,50})
    f3 = near_uniform(FermiFS{49,100})

    for H in (
        HubbardReal1D(b1),
        HubbardReal1D(b2),
        HubbardReal1D(b3),

        HubbardMom1D(b1),
        HubbardMom1D(b2),
        HubbardMom1D(b3),

        ExtendedHubbardReal1D(b1),
        ExtendedHubbardReal1D(b2),
        ExtendedHubbardReal1D(b3),

        HubbardRealSpace(b1),
        HubbardRealSpace(b2),
        HubbardRealSpace(b3),

        HubbardRealSpace(f1),
        HubbardRealSpace(f2),
        HubbardRealSpace(f3),

        HubbardRealSpace(CompositeFS(f1, f1)),
        HubbardRealSpace(CompositeFS(f2, f2)),
        HubbardRealSpace(CompositeFS(f3, f3)),

        HubbardReal1DEP(b1; u=0.5, v_ho=0.5),
        HubbardReal1DEP(b2; u=0.5, v_ho=0.5),
        HubbardReal1DEP(b3; u=0.5, v_ho=0.5),

        HubbardMom1DEP(b1),
        HubbardMom1DEP(b2),
        HubbardMom1DEP(b3),

        HubbardMom1DEP(CompositeFS(f1, f1)),
        HubbardMom1DEP(CompositeFS(f2, f2)),
        HubbardMom1DEP(CompositeFS(f3, f3)),

        Transcorrelated1D(CompositeFS(f1, f1)),
        Transcorrelated1D(CompositeFS(f2, f2)),
        Transcorrelated1D(CompositeFS(f3, f3)),
        )
        addr = starting_address(H)
        for dv_type in (DVec, InitiatorDVec)
            for style in (
                IsDeterministic(),
                IsStochasticInteger(),
                IsStochasticWithThreshold(),
                IsDynamicSemistochastic(),
                )
                hamname = string(
                    nameof(typeof(H)), "(", num_modes(addr), ")/", nameof(typeof(style))
                )
                @testset "Allocations for $(hamname)" begin
                    dτ = if num_modes(addr) == 10
                        1e-4
                    elseif num_modes(addr) == 50
                        1e-5
                    else
                        1e-6
                    end

                    dv = dv_type(addr => 1.0, style=IsDynamicSemistochastic())
                    sizehint!(dv, 500_000)

                    # Warmup for lomc!
                    _, st = lomc!(
                        H, dv;
                        params=RunTillLastStep(;shift=float(diagonal_element(H, addr)), dτ),
                        maxlength=10_000,
                        laststep=1,
                    )

                    r = only(only(st.spectral_states).single_states)

                    # Warmup for step!
                    apply_operator_wrap!(r)
                    apply_operator_wrap!(r)
                    apply_operator_wrap!(r)
                    apply_operator_wrap!(r)
                    apply_operator_wrap!(r)

                    allocs_step = @allocated apply_operator_wrap!(r)
                    @test allocs_step ≤ 512

                    dv = dv_type(addr => 1.0, style=IsDynamicSemistochastic())
                    allocs_full = @allocated lomc!(
                        H, dv; dτ, laststep=200, maxlength=10_000
                    )
                    @test allocs_full ≤ 1e8 # 100MiB

                    # Print out the results to make it easier to find problems.
                    print(rpad(hamname, 50))
                    print(": per step ", allocs_step, ", full ", allocs_full/(1024^2), "M\n")
                end
            end
        end
    end
end
