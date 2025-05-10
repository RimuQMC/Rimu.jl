# copied from https://github.com/JuliaLang/julia/blob/master/test/netload/memtest.jl#L3
# This relies on internals of the Julia runtime, so it may break in future versions.

struct RUsage
    ru_utime_sec::Clong         #  user CPU time used
    ru_utime_usec::Clong        #  user CPU time used
    ru_stime_sec::Clong         #  system CPU time used
    ru_stime_usec::Clong        #  system CPU time used
    ru_maxrss::Clong            #  maximum resident set size
    ru_ixrss::Clong             #  integral shared memory size
    ru_idrss::Clong             #  integral unshared data size
    ru_isrss::Clong             #  integral unshared stack size
    ru_minflt::Clong            #  page reclaims (soft page faults)
    ru_majflt::Clong            #  page faults (hard page faults)
    ru_nswap::Clong             #  swaps
    ru_inblock::Clong           #  block input operations
    ru_oublock::Clong           #  block output operations
    ru_msgsnd::Clong            #  IPC messages sent
    ru_msgrcv::Clong            #  IPC messages received
    ru_nsignals::Clong          #  signals received
    ru_nvcsw::Clong             #  voluntary context switches
    ru_nivcsw::Clong            #  involuntary context switches
end


"""
    Rimu.get_vmsize()

Get the maximum resident set size (VM size) of the current process in bytes.

## Example
```julia
julia> Rimu.get_vmsize()
1214218240

julia> Rimu.get_vmsize()|> Base.format_bytes
"1.133 GiB"
```
"""
function get_vmsize()
    ru = Vector{RUsage}(undef, 1)
    ccall(:getrusage, Cint, (Cint, Ptr{Cvoid}), 0, ru)
    return ru[1].ru_maxrss
end
