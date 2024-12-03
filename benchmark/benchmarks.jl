using Rimu
using KrylovKit
using BenchmarkTools

const SUITE = @benchmarkset "Rimu" begin
    @benchmarkset "Exact" begin
        @benchmarkset "Diagonalization" begin
            @case "2D Hubbard" begin
                M = 16
                addr = FermiFS2C(M, 1 => 1, 2 => 1, 1 => -1, 2 => -1)
                ham = HubbardRealSpace(addr; geometry=PeriodicBoundaries(4, 4))
                dv = PDVec(addr => 1.0)
                eigsolve(ham, dv, 1, :SR; tol=1e-9)
            end seconds=30

            @case "Bose-Hubbard in momentum space" begin
                M = N = 10
                addr = BoseFS(M, M ÷ 2 => N)
                ham = HubbardMom1D(addr; u=6.0)
                dv = PDVec(addr => 1.0)
                eigsolve(ham, dv, 1, :SR; tol=1e-9)
            end seconds=40
        end

        @benchmarkset "Multiplication" begin
            @case "Momentum space" begin
                # dimension is 189225
                M = 20
                addr = BoseFS(M, M÷2 => 10)
                ham = HubbardMom1D(addr; u=6.0)
                dv1 = PDVec(addr => 1.0)
                dv2 = zerovector(dv1)
                mul!(dv2, ham, dv1)
                mul!(dv1, ham, dv2)
                mul!(dv2, ham, dv1)
                mul!(dv1, ham, dv2)
                mul!(dv2, ham, dv1)
            end seconds=10

            @case "Transcorrelated" begin
                # dimension is 189225
                M = 30
                addr = FermiFS2C(M, M÷2-1 => 1, M => 1, M÷2 => -1, M÷2+1 => -1)
                ham = Transcorrelated1D(addr)
                dv1 = PDVec(addr => 1.0)
                dv2 = zerovector(dv1)
                mul!(dv2, ham, dv1)
                mul!(dv1, ham, dv2)
                mul!(dv2, ham, dv1)
                mul!(dv1, ham, dv2)
            end seconds=10
        end
    end

    @benchmarkset "FCIQMC" begin
        @case "(10, 20) Mom space with projected energy and initiator" begin
            addr = BoseFS(20, 10 => 10)
            ham = HubbardMom1D(addr, u=1.0)
            dv = PDVec(addr => 1.0; style=IsDynamicSemistochastic(), initiator=true)
            post_step = ProjectedEnergy(ham, dv)
            s_strat = DoubleLogUpdate(target_walkers=40_000)

            lomc!(ham, dv; s_strat, post_step, dτ=1e-4, laststep=8000)
        end seconds=150

        @case "(50, 50) Real space" begin
            addr = near_uniform(BoseFS{50,50})
            ham = HubbardReal1D(addr, u=6.0)
            dv = PDVec(addr => 1.0; style=IsDynamicSemistochastic())
            s_strat = DoubleLogUpdate(target_walkers=50_000)

            lomc!(ham, dv; s_strat, dτ=1e-4, laststep=1000)
        end seconds=150
    end
end
