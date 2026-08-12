using Rimu
using Test
using DataFrames
using KrylovKit
using Suppressor
using Setfield
import Tables

@testset "Reporting" begin
    address = BoseFS((1, 2, 1, 1))
    H = HubbardReal1D(address; u=2)
    dv = PDVec(address => 1, style=IsDeterministic())

    @testset "ReportDFAndInfo" begin
        reporting_strategy = ReportDFAndInfo(
            reporting_interval=5, info_interval=10, io=devnull, writeinfo=true
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100
        )
        df = DataFrame(solve(p))
        @test size(df, 1) == 20
        @test metadata(df, "Rimu.PACKAGE_VERSION") == string(pkgversion(Rimu))

        out = @capture_out begin
            reporting_strategy = ReportDFAndInfo(
                reporting_interval=5, info_interval=10, io=stdout, writeinfo=true
            )
            p = ProjectorMonteCarloProblem(
                H;
                start_at=dv, reporting_strategy, last_step=100
            )
            df = DataFrame(solve(p))
        end
        @test length(split(out, '\n')) == 3 # (last line is empty)
    end
    @testset "ReportToFile" begin
        # Clean up.
        rm("test-report.arrow"; force=true)
        rm("test-report-1.arrow"; force=true)
        rm("test-report-2.arrow"; force=true)
        rm("test-report-3.arrow"; force=true)
        rm("test-report-nc.arrow"; force=true)
        rm("test-report-lz4.arrow"; force=true)

        reporting_strategy = ReportToFile(
            filename="test-report.arrow", io=devnull, save_if=false
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100
        )
        df = DataFrame(solve(p))
        @test !isfile("test-report.arrow")
        @test Rimu._isopen(reporting_strategy) == false

        reporting_strategy = ReportToFile(filename="test-report.arrow", io=devnull)
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100, metadata=(; u=6.0)
        )
        df = DataFrame(solve(p))
        @test isempty(df)
        @test Rimu._isopen(reporting_strategy) == false
        df1 = RimuIO.load_df("test-report.arrow")
        @test metadata(df1, "u") == "6.0" # custom metadata is saved
        @test metadata(df1, "filename") == "test-report.arrow" # filename in metadata

        reporting_strategy = ReportToFile(
            filename="test-report.arrow", io=devnull, chunk_size=5
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100
        )
        df = DataFrame(solve(p))
        @test isempty(df)
        @test Rimu._isopen(reporting_strategy) == false
        df2 = RimuIO.load_df("test-report-1.arrow")

        reporting_strategy = ReportToFile(
            filename="test-report.arrow", io=devnull, return_df=true
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100
        )
        df3 = DataFrame(solve(p))
        @test !isempty(df3)
        @test Rimu._isopen(reporting_strategy) == false
        df4 = RimuIO.load_df("test-report-2.arrow")

        @test df1.shift ≈ df2.shift
        @test df2.norm ≈ df3.norm
        @test df3 == df4

        # ReportToFile with skipping interval
        df5 = df1[10:10:100, :]
        reporting_strategy = ReportToFile(
            filename="test-report.arrow", reporting_interval=10, io=devnull, chunk_size=10
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100
        )
        df = DataFrame(solve(p))
        @test isempty(df)
        df6 = RimuIO.load_df("test-report-3.arrow")

        @test df6.shift ≈ df5.shift
        @test df6.norm ≈ df5.norm

        # ReportToFile with compression
        @test_throws ArgumentError ReportToFile(compress=false)

        reporting_strategy = ReportToFile(
            filename="test-report-nc.arrow", io=devnull, return_df=true,
            compress=nothing
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100
        )
        df7 = DataFrame(solve(p))
        @test !isempty(df7)
        @test Rimu._isopen(reporting_strategy) == false
        @test df7 == RimuIO.load_df("test-report-nc.arrow")


        reporting_strategy = ReportToFile(
            filename="test-report-lz4.arrow", io=devnull, return_df=true,
            compress=:lz4
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, reporting_strategy, last_step=100
        )
        df8 = DataFrame(solve(p))
        @test !isempty(df8)
        @test Rimu._isopen(reporting_strategy) == false
        @test df8 == RimuIO.load_df("test-report-lz4.arrow")

        @test filesize("test-report-lz4.arrow") < filesize("test-report-nc.arrow")
        @test filesize("test-report.arrow") < filesize("test-report-lz4.arrow")
        # The default compression `:zstd` produces the smallest files.

        # Clean up.
        rm("test-report.arrow"; force=true)
        rm("test-report-1.arrow"; force=true)
        rm("test-report-2.arrow"; force=true)
        rm("test-report-3.arrow"; force=true)
        rm("test-report-nc.arrow"; force=true)
        rm("test-report-lz4.arrow"; force=true)
    end
    @testset "Report" begin
        rp = Rimu.Report()
        Rimu.report!(rp, :b, 4)
        Rimu.report!(rp, :b, 6)
        Rimu.metadata!(rp, :a, 1)
        @test metadata(rp, "a") == "1"
        @test sprint(show, rp) == "Report:\n  b => [4, 6]\n metadata:\n  a => 1"

        # Tables integration
        NamedTuple(first(Tables.rows(rp))) == (b=4,)
        Tables.schema(rp) isa Tables.Schema
    end
end


@testset "Post step" begin
    address = BoseFS((0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0))
    H = HubbardMom1D(address; u=4)
    dv = DVec(address => 1)

    @testset "Projector, ProjectedEnergy" begin
        mpi_seed!(1337)

        post_step_strategy = (
            Projector(p1=NormProjector()),
            Projector(p2=copy(dv)),
            ProjectedEnergy(H, dv),
            ProjectedEnergy(H, dv, vproj=:vproj2, hproj=:hproj2),
            ProjectedEnergy(H, UniformProjector(), vproj=:vproj3, hproj=:hproj3),
        )
        p = ProjectorMonteCarloProblem(
            H;
            start_at=dv, post_step_strategy
        )
        df = DataFrame(solve(p))
        @test df.vproj == df.vproj2 == df.p2
        @test df.norm ≈ df.p1
        @test df.norm ≥ df.vproj3

        @test_throws ArgumentError solve(ProjectorMonteCarloProblem(
            H; start_at=dv, post_step_strategy=(Projector(a=dv), Projector(a=dv))
        ))
        @test_throws ArgumentError Projector(a=dv, b=dv)
        @test_throws ArgumentError Projector()
    end

    @testset "SignCoherence" begin
        mpi_seed!(1337)

        ref = eigsolve(H, dv, 1, :SR; issymmetric=true)[2][1]
        post_step_strategy = (SignCoherence(ref), SignCoherence(dv * -1, name=:single_coherence))
        df = solve(ProjectorMonteCarloProblem(H; start_at=dv, post_step_strategy)).df
        @test df.coherence[1] == 1.0
        @test all(-1.0 .≤ df.coherence .≤ 1.0)
        @test all(in.(df.single_coherence, Ref((-1, 0, 1))))

        # test type stability of `coherence` with complex vectors
        v1 = rand(ComplexF64, 10) .- (0.5 + 0.5im)
        v2 = rand(ComplexF64, 10) .- (0.5 + 0.5im)
        @inferred dot(v1, SignCorrelator{Float64}(), v2)

        cdv = DVec(address => 1 + im; style=StochasticStyles.IsStochastic2Pop())
        shift = float(valtype(cdv))(diagonal_element(H*address)) # need complex shift
        df = solve(ProjectorMonteCarloProblem(H; start_at=cdv, shift, post_step_strategy)).df
        @test df.coherence isa Vector{ComplexF64}
    end

    @testset "WalkerLoneliness" begin
        mpi_seed!(1337)

        post_step_strategy = WalkerLoneliness()
        df = solve(ProjectorMonteCarloProblem(H; start_at=dv, post_step_strategy)).df
        @test df.loneliness[1] == 1
        @test all(1 .≥ df.loneliness .≥ 0)

        cdv = DVec(address => 1 + im; style=StochasticStyles.IsStochastic2Pop())
        shift = float(valtype(cdv))(diagonal_element(H*address)) # need complex shift
        df = solve(ProjectorMonteCarloProblem(H; start_at=cdv, shift, post_step_strategy)).df
        @test df.loneliness isa Vector{ComplexF64}
    end

    @testset "Timer" begin
        post_step_strategy = Rimu.Timer()
        time_before = time()
        df = solve(ProjectorMonteCarloProblem(H; start_at=dv, post_step_strategy)).df
        time_after = time()
        @test df.time[1] > time_before
        @test df.time[end] < time_after
        @test issorted(df.time)
    end

    @testset "SingleParticleDensity" begin
        post_step_strategy = (
            SingleParticleDensity(save_every=2),
        )
        sim = solve(ProjectorMonteCarloProblem(H; start_at=dv, post_step_strategy))
        df = sim.df
        st = sim.state
        @test all(==(ntuple(_ -> 0, num_modes(address))), df.single_particle_density[1:2:end])
        @test all(≈(3), sum.(df.single_particle_density[2:2:end]))

        @test df.single_particle_density[end] == single_particle_density(
            st.spectral_states[1].single_states[1].v
        )

        for address in (
            CompositeFS(BoseFS((1, 2, 3)), FermiFS((0, 1, 0))),
        )
            @test single_particle_density(address) == (1, 3, 3)
            @test single_particle_density(address; component=1) == (1, 2, 3)
            @test single_particle_density(address; component=2) == (0, 1, 0)
            @test single_particle_density(DVec(address => 1); component=0) == (1, 3, 3)
            @test single_particle_density(DVec(address => 2); component=1) == (1, 2, 3)
            @test single_particle_density(DVec(address => 3); component=2) == (0, 1, 0)
        end
    end
end
