using Test
using Rimu
using Arrow
using DataFrames
using Tables

const tmpdir = mktempdir()

@testset "save_df, load_df" begin
    file = joinpath(tmpdir, "tmp.arrow")
    rm(file; force=true)

    df = DataFrame(a=[1, 2, 3], b=Complex{Float64}[1, 2, 3+im], d=rand(Complex{Int}, 3))
    RimuIO.save_df(file, df)
    df2 = RimuIO.load_df(file)
    @test df == df2

    rm(file)

    # test compression
    r = 10_005
    df2 = DataFrame(a = collect(1:r), b = rand(1:30,r))
    RimuIO.save_df(file, df2)
    compressed = filesize(file)
    rm(file)
    RimuIO.save_df(file, df2, compress=nothing)
    uncompressed = filesize(file)
    rm(file)
    @test compressed < uncompressed
end

@testset "Addresses" begin
    for addr in (
        BitString{10}(0b1100110011),
        SortedParticleList((1, 0, 1, 0, 0, 2, 3)),
        near_uniform(BoseFS{10, 10}),
        BoseFS(101, 5 => 10),
        FermiFS(1,1,1,0,0,0),
        FermiFS2C(near_uniform(FermiFS{50,100}), FermiFS(100, 1 => 1)),
        CompositeFS(near_uniform(BoseFS{8,9}), near_uniform(BoseFS{1,9})),
        OccupationNumberFS(1,2,3,4,5),
        CompositeFS(
            BoseFS(1,1,1,1,1),
            FermiFS(1,0,0,0,0),
            OccupationNumberFS(5,4,3,2,1),
            BoseFS(1,1,0,0,0),
            FermiFS(1,1,1,0,0),
        ),
    )
        @testset "$(typeof(addr))" begin
            @testset "ArrowTypes interface" begin
                arrow_name = ArrowTypes.arrowname(typeof(addr))
                ArrowType = ArrowTypes.ArrowType(typeof(addr))
                serialized = ArrowTypes.toarrow(addr)
                @test typeof(serialized) ≡ ArrowType
                meta = ArrowTypes.arrowmetadata(typeof(addr))

                # This takes care of some weirdness with how Arrow handles things.
                if addr isa CompositeFS
                    T = NamedTuple{ntuple(Symbol, num_components(addr)), typeof(serialized)}
                    JuliaType = ArrowTypes.JuliaType(Val(arrow_name), T, meta)
                    result = ArrowTypes.fromarrow(JuliaType, serialized...)
                else
                    T = typeof(serialized)
                    JuliaType = ArrowTypes.JuliaType(Val(arrow_name), T, meta)
                    result = ArrowTypes.fromarrow(JuliaType, serialized)
                end

                @test result ≡ addr
            end
            @testset "saving and loading" begin
                file = joinpath(tmpdir, "tmp-addr.arrow")
                RimuIO.save_df(file, DataFrame(addr = [addr]))
                result = only(RimuIO.load_df(file).addr)
                @test result ≡ addr
                rm(file; force=true)
            end
        end
    end
end

@testset "(P)DVec Tables.jl interface" begin
    for T in (PDVec, DVec)
        dvec = T(zip(1:100, 0.5:0.5:50))
        tbl = Tables.table(dvec)
        @test Tables.istable(tbl)
        @test Tables.rowaccess(tbl)
        @test !Tables.columnaccess(tbl)
        @test Tables.schema(tbl) == Tables.Schema((:key, :value), (Int, Float64))
        @test length(Tables.rows(tbl)) == length(dvec)

        rows = Tables.rows(tbl)
        @test [row.value for row in rows] == [dvec[row.key] for row in rows]
        @test sum(row.value for row in Tables.rows(tbl)) == sum(values(dvec))
    end
end

@testset "save_state, load_state" begin
    file = joinpath(tmpdir, "tmp-dvec.arrow")
    rm(file; force=true)

    @testset "vectors" begin
        ham = HubbardReal1D(BoseFS(1,1,1))
        @testset "errors" begin
            df1 = DataFrame(key=[1,2,3], value=[1,2,3], error=[0,0,0])
            save_df(file, df1)
            @test_throws ArgumentError load_state(file)

            df2 = DataFrame(error=[0,0,0])
            save_df(file, df2)
            @test_throws ArgumentError load_state(file)

            rm(file)
        end

        @testset "save DVec" begin
            dvec = ham * DVec([BoseFS(1,1,1) => 1.0, BoseFS(2,1,0) => π])
            save_state(file, dvec)
            output, _ = load_state(file)
            @test output == dvec
            rm(file)
        end

        @testset "save PDVec" begin
            pdvec = ham * PDVec([BoseFS(1,1,1) => 1.0, BoseFS(0,3,0) => ℯ])
            save_state(file, pdvec)
            output, _ = load_state(file)
            @test output == pdvec

            @test load_state(PDVec, file)[1] isa PDVec
            @test load_state(PDVec, file)[1] == pdvec
            @test load_state(DVec, file)[1] isa DVec
            @test load_state(DVec, file)[1] == pdvec
            rm(file)
        end

        @testset "save empty vector" begin
            dvec = DVec{Int,Int}()
            save_state(file, dvec)
            @test isempty(load_state(file)[1])
            pdvec = PDVec{Int,Int}()
            save_state(file, pdvec)
            @test isempty(load_state(file)[1])
            rm(file)
        end
    end

    @testset "metadata" begin
        dvec = DVec(BoseFS(1,1,1,1) => 1.0)
        save_state(
            file, dvec;
            int=1, float=2.3, complex=1.2 + 3im, string="a string", bool=true
        )
        _, meta = load_state(file)

        @test meta.int === 1
        @test meta.float === 2.3
        @test meta.complex === 1.2 + 3im
        @test meta.bool === true
        @test meta.string === "a string"
        @test meta.RIMU_PACKAGE_VERSION == Rimu.PACKAGE_VERSION
        rm(file)
    end

end
