using Rimu
using StaticArrays
using Test

using Rimu: replace_keys, delete_and_warn_if_present, clean_and_warn_if_others_present,
    split_keys

@testset "helpers" begin
    @testset "walkernumber" begin
        v = [1, 2, 3]
        @test walkernumber(v) == norm(v, 1)
        dvc = DVec(:a => 2 - 5im)
        @test StochasticStyle(dvc) isa StochasticStyles.IsStochastic2Pop
        @test walkernumber(dvc) == 2.0 + 5.0im
        dvi = DVec(:a => Complex{Int32}(2 - 5im))
        @test StochasticStyle(dvi) isa StochasticStyles.IsStochastic2Pop
        dvr = DVec(i => randn() for i in 1:100; capacity=100)
        @test walkernumber(dvr) ≈ norm(dvr, 1)
    end
    @testset "MultiScalar" begin
        a = Rimu.MultiScalar(1, 1.0, SVector(1))
        @test a[1] ≡ 1
        @test a[2] ≡ 1.0
        @test a[3] ≡ SVector(1)
        @test length(a) == 3
        @test collect(a) == [1, 1.0, SVector(1)]
        b = Rimu.MultiScalar(SVector(2, 3.0, SVector(4)))
        for op in (+, min, max)
            c = op(a, b)
            @test op(a[1], b[1]) == c[1]
            @test op(a[2], b[2]) == c[2]
            @test op(a[2], b[2]) == c[2]
        end
        @test_throws MethodError a + Rimu.MultiScalar(1, 1, 1)
    end

    @testset "keyword helpers" begin
        nt = (; a=1, b=2, c=3, d=4)
        nt2 = replace_keys(nt, (:a => :x, :b => :y, :u => :v))
        @test nt2 == (c=3, d=4, x=1, y=2)
        nt3 = @test_logs((:warn, "The keyword(s) \"a\", \"b\" are unused and will be ignored."),
            delete_and_warn_if_present(nt, (:a, :b, :u)))
        @test nt3 == (; c=3, d=4)
        nt4 = @test_logs((:warn, "The keyword(s) \"c\", \"d\" are unused and will be ignored."),
            clean_and_warn_if_others_present(nt, (:a, :b, :u)))
        @test nt4 == (; a=1, b=2)

        split, rest = split_keys(nt, :a, :b, :e)
        @test split_keys(nt, :a) == split_keys(nt, (:a,))
        @test split == (; a=1, b=2)
        @test rest == (; c=3, d=4)

        @test split_keys((;), :a, :b, :c) == split_keys((), :a, :b, :c) == ((;), (;))
    end
    @testset "index_apply" begin
        t = (1, 1.0, 2 + 3im)
        @test Rimu.Hamiltonians.index_apply(isreal, t, 1)
        @test Rimu.Hamiltonians.index_apply(isreal, t, 2)
        @test Rimu.Hamiltonians.index_apply(!isreal, t, 3)

        @test Rimu.Hamiltonians.index_apply(+, t, 1, 0.2im) == 1 + 0.2im
        @test Rimu.Hamiltonians.index_apply(+, t, 2, 0.2im) == 1 + 0.2im
        @test Rimu.Hamiltonians.index_apply(+, t, 3, 0.2im) == 2 + 3.2im
    end
end
