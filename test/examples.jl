using Test
using Rimu

# Note: Tests for MPI scripts are in `mpi_runtests.jl`

# Test specific scripts
@testset "Ex1-BHM" begin
    include("../scripts/Ex1-BHM.jl")
    dfr = load_df("fciqmcdata.arrow")
    qmcdata = last(dfr,steps_measure)
    (qmcShift,qmcShiftErr) = mean_and_se(qmcdata.shift)
    @test qmcShift ≈ -4.171133393316872 rtol=0.01

    # clean up
    rm("fciqmcdata.arrow", force=true)
end

@testset "Ex2-G2" begin
    include("../scripts/Ex2-G2.jl")
    r = rayleigh_replica_estimator(df; op_name = "Op1", skip=steps_equilibrate)
    @test r.f ≈ 0.23371704332410984 rtol=0.01
end
