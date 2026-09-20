using Rimu
using Test
using Rimu.StochasticStyles
using Random: seed!

using Rimu.StochasticStyles: RandomFromCumsumIterator, ColumnStats, update_column_stats!

@testset "ColumnStats" begin
    address = BoseFS(1,2,3)
    cs = ColumnStats{ComplexF64, typeof(address), Float64}()
    @test length(cs) == 0
    @test cs.column_sum[] == 0 + 0im
    @test eval(Meta.parse(repr(cs))) == cs
    @test hash(cs) == hash(ColumnStats{ComplexF64,typeof(address),Float64}())
end

@testset "IsWalkerDiffusion" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    h = HubbardRealSpace(address; t=1, w=-2)
    column = h * address
    iwd = IsWalkerDiffusion(h; splitting_threshold=0.1, projection_threshold=0.2,
        target_threshold=0.3, rel_spawning_threshold=0.4, abs_spawning_threshold=0.5
    )
    @test isa(iwd, IsWalkerDiffusion{Float64,typeof(address),Float64})
    @test iwd.splitting_threshold == 0.1
    @test iwd.projection_threshold == 0.2
    @test iwd.target_threshold == 0.3
    @test iwd.rel_spawning_threshold == 0.4
    @test iwd.abs_spawning_threshold == 0.5
    @test length(iwd.column_stats) == 0
    @test eval(Meta.parse(repr(iwd))) == iwd
    @test hash(iwd) == hash(IsWalkerDiffusion{Float64,typeof(address),Float64}(
        0.1, 0.2, 0.3, 0.4, 0.5, ColumnStats{Float64,typeof(address),Float64}()
    ))
end

@testset "update_column_stats!" begin
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    h = HubbardRealSpace(address; t=1, w=-2)
    column = h * address
    iwd = IsWalkerDiffusion(h)
    stats = iwd.column_stats
    @test length(stats) == 0
    @test update_column_stats!(iwd, column) isa IsWalkerDiffusion
    @test length(stats) == num_offdiagonals(column) + 1
    @test stats.column_length[] == length(stats) == length(stats.mod_cumsum) ==
        length(stats.addresses) == length(stats.values)
    @test stats.column_sum[] ≈ sum(last.(collect(column)))
    @test stats.mod_cumsum[end] == sum(abs.(last.(collect(column))))
    @test stats == update_column_stats!(ColumnStats{Float64,typeof(address),Float64}(), column)
end

# function vcat_samples(rfci::RandomFromCumsumIterator, reps)
#     v = mapreduce(vcat, 1:reps) do i
#         collect(rfci)
#     end
#     return v
# end
function accumulate_samples(rfci::RandomFromCumsumIterator, reps)
    v = zeros(length(rfci.cumsum))
    vsquares = zeros(length(rfci.cumsum))
    for rep in 1:reps
        for i in rfci
            v[i] += rfci.weight
            vsquares[i] += rfci.weight^2
        end
    end
    return v ./ reps
end

@testset "RandomFromCumsumIterator" begin
    seed!(1234)
    r = rand(10)
    cs = cumsum(r)
    rci1 = RandomFromCumsumIterator(1, cs)
    @test length(collect(rci1)) == length(rci1) == 1
    means_independent = accumulate_samples(rci1, 1000)
    rci50 = RandomFromCumsumIterator(50, cs)
    @test length(collect(rci50)) == length(rci50) == 50
    means_correlated = accumulate_samples(rci50, 20)

    # using StatsPlots: scatter
    # scatter(means_independent; label="independent", title="RandomFromCumsumIterator test",
    #     xlabel="Index", ylabel="Mean value", legend=:topright)
    # scatter!(means_correlated; label="correlated")
    # scatter!(r; label="original")

    @test isapprox(means_independent, r; atol=0.3)
    @test isapprox(means_correlated, r; atol=0.1) # correlated samples have lower variance
end

@testset "apply_column!" begin
    seed!(1234)
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    h = HubbardRealSpace(address; t=1, w=-2)
    column = h * address
    iwd = IsWalkerDiffusion(h; projection_threshold=0.2)
    stats = iwd.column_stats
    @test length(stats) == 0

    val = 2.0 # triggers exact step
    w = empty!(DVec(column))
    step_stat = apply_column!(iwd, w, column, val)
    @test step_stat[7] == 1 # :exact_steps
    @test w == val * DVec(column)

    val = 1.0 # triggers inexact step, but still exact with 8 walkers
    w = empty!(DVec(column))
    step_stat = apply_column!(iwd, w, column, val)
    @test step_stat[8] == 1 # :inexact_steps
    @test step_stat[5] == 8 # :walkers
    @test w == val * DVec(column)

    val = 0.0 # triggers stochastic projection to zero
    w = empty!(DVec(column))
    step_stat = apply_column!(iwd, w, column, val)
    @test step_stat[4] == 1 # :deaths
    @test length(w) == 0

    val = 0.3 # stochastic spawning
    w = empty!(DVec(column))
    step_stat = apply_column!(iwd, w, column, val)
    @test step_stat[8] == 1 # :inexact_steps
    @test step_stat[5] == 3 # :walkers
    @test 1 ≤ length(w) ≤ 3
end

@testset "projector Monte Carlo with walker diffusion" begin
    seed!(1234)
    address = FermiFS(1, 1, 1, 1, 0, 0, 0, 0)
    h = HubbardRealSpace(address; t=1, w=-2)
    iwd = IsWalkerDiffusion(h)
    pmp = ProjectorMonteCarloProblem(h; style=iwd)
    res = solve(pmp)

    @test res.success
end
