using Test
using Rimu
using Rimu: GlobalStepAction, OperatorOverlaps, StrictPairIter, SingleState,
                SpectralState

@testset "StrictPairIter" begin
    spi = StrictPairIter(4)
    pairs = collect(spi)
    @test pairs == [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    @test length(spi) == 6
end
