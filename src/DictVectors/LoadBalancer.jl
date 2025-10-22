using Rimu
using Rimu.DictVectors: Communicator, PDWorkingMemory, localpart
using Rimu.DictVectors:
    remote_segments, local_segments,
    first_column, dict_add!, fastrange_hash

using MPI
using Statistics
using Random
import Rimu.DictVectors: mpi_rank, mpi_size, mpi_comm, synchronize_remote!, copy_to_local!, remote_segments,local_segments, first_column, dict_add!, fastrange_hash

# Convenience macro for root-only printing
macro mpi_root(expr)
    return quote
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            $(esc(expr))
        end
    end
end

struct LoadBalancer
    variance_threshold::Float64
    check_frequency::Int
end

LoadBalancer(; variance_threshold=1.5, check_frequency=100) = LoadBalancer(variance_threshold, check_frequency)

```@docs
Major implementation of the custom communicator with integrated load balancing.
This majorly deals with the implementation of the custom communicator
For this LoadBalancing.
```
struct LoadBalancedCommunicator{K,V} <: Communicator
    mpi_comm::MPI.Comm
    mpi_rank::Int
    mpi_size::Int
    load_balancer::LoadBalancer
    step::Base.RefValue{Int}
end

function LoadBalancedCommunicator{K,V}(; mpi_comm=MPI.COMM_WORLD, variance_threshold=1.5, check_frequency=100) where {K,V}
    mpi_rank = MPI.Comm_rank(mpi_comm)
    mpi_size = MPI.Comm_size(mpi_comm)
    balancer = LoadBalancer(variance_threshold, check_frequency)
    return LoadBalancedCommunicator{K,V}(mpi_comm, mpi_rank, mpi_size, balancer, Ref(0))
end

mpi_rank(comm::LoadBalancedCommunicator) = comm.mpi_rank
mpi_size(comm::LoadBalancedCommunicator) = comm.mpi_size
mpi_comm(comm::LoadBalancedCommunicator) = comm.mpi_comm

```@docs
    synchronize_remote!(comm::LoadBalancedCommunicator, w::PDWorkingMemory)

Performs the primary synchronization step, exchanging walkers (key-value pairs)
that were "spawned" on the local rank but belong on a remote rank.

This function is the core of the parallel "spawning" logic. It:
1.  Increments the communicator's step counter.
2.  Gathers all walkers from `remote_segments(w)` destined for other ranks.
3.  Uses a non-blocking `MPI.Alltoall` pattern (exchanging counts first, then data)
    to send these local "remote" walkers to their correct owner rank.
4.  Receives walkers from all other ranks that belong on the *current* rank.
5.  **Mutates** `w` by adding the received walkers into the appropriate
    `local_segments(w)` using `dict_add!`.
6.  Checks if the current `step` triggers a load-balancing check (based on
    `comm.load_balancer.check_frequency`) and calls `perform_load_balancing!`
    if it does.

This function is "bang" (`!`) because it modifies `w` in-place.

# Arguments
- `comm::LoadBalancedCommunicator`: The communicator wrapper. Its internal `step`
  counter is incremented, and its load balancer settings are checked.
- `w::PDWorkingMemory`: The working memory data structure. Its `local_segments`
  **will be modified** to include newly received walkers.

# Returns
- `(), ()`: Returns empty tuples, as the function's purpose is its side effects.
```
function synchronize_remote!(comm::LoadBalancedCommunicator{K,V}, w::PDWorkingMemory{K,V,W,S}) where {K,V,W,S}
    comm.step[] += 1
    step = comm.step[]
    rank = mpi_rank(comm)
    size = mpi_size(comm)
    comm_mpi = mpi_comm(comm)

    if size == 1
        return (), ()
    end

    # Original synchronization logic
    send_data_per_rank = [Vector{Pair{K,W}}() for _ in 0:(size - 1)]

    for i in 0:(size - 1)
        if i == rank
            continue
        end
        for segment_dict in remote_segments(w, i)
            append!(send_data_per_rank[i + 1], [k => W(v) for (k, v) in segment_dict])
        end
    end

    send_counts = [length(data) for data in send_data_per_rank]
    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm_mpi)

    recv_buffers = [Vector{Pair{K,W}}(undef, recv_counts[i + 1]) for i in 0:(size - 1)]
    send_requests = MPI.Request[]
    recv_requests = MPI.Request[]

    for i in 0:(size - 1)
        if i == rank
            continue
        end
        if !isempty(send_data_per_rank[i + 1])
            req = MPI.Isend(send_data_per_rank[i + 1], comm_mpi; dest=i, tag=0)
            push!(send_requests, req)
        end
        if !isempty(recv_buffers[i + 1])
            req = MPI.Irecv!(recv_buffers[i + 1], comm_mpi; source=i, tag=0)
            push!(recv_requests, req)
        end
    end

    MPI.Waitall(send_requests)
    MPI.Waitall(recv_requests)

    # Collect segment data before load balancing
    local_lengths_before = [length(seg) for seg in local_segments(w)]

    for i in 0:(size - 1)
        if i == rank
            continue
        end
        for (seg_idx, local_seg) in enumerate(local_segments(w))
            segment_data = [k => v for (k, v) in recv_buffers[i + 1]
                             if Rimu.DictVectors.target_segment(comm, k, length(local_segments(w)))[1] == seg_idx]
            dict_add!(local_seg, segment_data)
        end
    end

    # Collect segment data after synchronization for load balancing check
    local_lengths_after = [length(seg) for seg in local_segments(w)]

    # Print segment lengths for debugging
    if rank == 0 && step > 0 && step % 50 == 0
        total_before = sum(local_lengths_before)
        total_after = sum(local_lengths_after)
        println("Step $step - Local segments before sync: $local_lengths_before (total: $total_before)")
        println("Step $step - Local segments after sync: $local_lengths_after (total: $total_after)")
    end

    # Perform load balancing check if needed
    if step > 0 && step % comm.load_balancer.check_frequency == 0
        perform_load_balancing!(comm, w, step)
    end

    return (), ()
end

```@docs
    copy_to_local!(
        comm::LoadBalancedCommunicator,
        w::PDWorkingMemory,
        pdvec
    ) -> Rimu.DictVectors.localparttype(pdvec)

Copy and redistribute a parallel dictionary vector `pdvec` into the `PDWorkingMemory` `w`,
partitioning the data according to the hashing function defined in `comm`.

This function is a core component of the load-balancing data exchange. It performs an
all-to-all communication (`MPI.Allgatherv`) to gather the complete vector `pdvec` onto
every MPI rank. Each rank then iterates through the complete dataset and repartitions it:

a)All local and remote segments in `w` are emptied.
b)An `MPI.Allgatherv` collects all key-value pairs from all ranks.
c)Each rank processes all pairs:
    A global segment index and `target_rank` are computed for each key using
      `fastrange_hash`.
    If the `target_rank` matches the current rank, the pair is added to the
      appropriate **local segment** in `w`.
    If the `target_rank` is different, the pair is added to the
      remote segment in `w` corresponding to that `target_rank`. These remote
      segments act as send-buffers for a subsequent `exchange_vectors!` call.

If `mpi_size(comm) == 1`, this function is a no-op and returns `pdvec`.

# Arguments
 `comm`: The `LoadBalancedCommunicator` managing MPI communication and hashing.
 `w`: The `PDWorkingMemory` to be populated. **This argument is mutated.**
 `pdvec`: The parallel vector (e.g., a `DVec`) containing the data to be
    redistributed.

# Returns
 The `first_column(w)` (the local part of the working memory), which is now
    populated with the redistributed local data.

```
function copy_to_local!(comm::LoadBalancedCommunicator{K,V}, w::PDWorkingMemory{K,V,W,S}, pdvec) where {K,V,W,S}
    mpi_size(comm) == 1 && return pdvec

    rank = mpi_rank(comm)
    size = mpi_size(comm)
    comm_mpi = mpi_comm(comm)

    # Clear local segments
    local_segs = local_segments(w)
    n_segments = length(local_segs)
    for local_seg in local_segs
        empty!(local_seg)
    end

    # Clear all remote segments
    for i in 0:(size - 1)
        if i == rank
            continue
        end
        for remote_seg in remote_segments(w, i)
            empty!(remote_seg)
        end
    end

    # Collect all pairs from all ranks
    local_pairs = collect(pairs(localpart(pdvec)))
    n_local = length(local_pairs)
    counts = MPI.Allgather(n_local, comm_mpi)
    all_pairs = MPI.Allgatherv(local_pairs, Cint.(counts), comm_mpi)

    # Distribute received data into local/remote segments
    total_segments = Rimu.DictVectors.total_num_segments(comm, n_segments)
    offset = 0
    for i in 0:(size - 1)
        rank_pairs = view(all_pairs, (offset + 1):(offset + counts[i + 1]))
        for (k, v) in rank_pairs
            # Compute global segment index
            global_seg = fastrange_hash(k, total_segments)
            target_rank = div(global_seg - 1, n_segments)
            seg_idx = mod(global_seg - 1, n_segments) + 1
            if target_rank == rank
                # Add to local segment
                local_segs[seg_idx][k] = get(local_segs[seg_idx], k, W(0.0)) + W(v)
            else
                # Add to specific remote segment for that rank
                remote_segs = remote_segments(w, target_rank)
                remote_segs[seg_idx][k] = get(remote_segs[seg_idx], k, W(0.0)) + W(v)
            end
        end
        offset += counts[i + 1]
    end

    return first_column(w)
end

```@docs
    perform_load_balancing!(
        comm::LoadBalancedCommunicator,
        w::PDWorkingMemory,
        step::Int
    )

Check the current load imbalance across MPI ranks and, if it exceeds a
predefined threshold, perform a load balancing operation by migrating segments
of the `PDWorkingMemory` `w`.

This function is intended to be called at each step of a simulation. It
executes the following logic:

1.  Gathers the total number of "walkers" (sum of `length` of all local segments)
    from every rank using `MPI.Allgather`.
2.  Computes the standard deviation (`variance`) of these walker counts on rank 0
    and broadcasts the result to all ranks.
3.  Compares the `variance` to `comm.load_balancer.variance_threshold`.
4.  If the imbalance is too high (variance > threshold), it calls
    `decide_migrations` to determine which segments should be moved.
5.  If a migration plan is returned, it calls `perform_segment_migration!` to
    execute the data transfer, mutating `w` in the process.
6.  If balancing was performed, rank 0 prints a log message with the current
    step and variance.

If `mpi_size(comm) == 1`, this function is a no-op.

# Arguments
`comm`: The `LoadBalancedCommunicator` managing the simulation. It contains
    the MPI communicator and the load balancer configuration.
`w`: The `PDWorkingMemory` holding the distributed data (walkers).
    **This argument is mutated** if balancing is triggered.
`step`: The current simulation step number, used for logging purposes.

```
function perform_load_balancing!(comm::LoadBalancedCommunicator, w::PDWorkingMemory, step::Int)
    # Skip if single rank
    mpi_size(comm) == 1 && return

    # Collect load metrics
    local_segs = local_segments(w)
    local_walkers = sum(length(seg) for seg in local_segs)
    all_walker_counts = MPI.Allgather(local_walkers, mpi_comm(comm))

    variance = 0.0
    if mpi_rank(comm) == 0
        variance = isempty(all_walker_counts) ? 0.0 : std(all_walker_counts)
    end
    variance = MPI.Bcast(variance, 0, mpi_comm(comm))

    # Check if balancing is needed
    if variance > comm.load_balancer.variance_threshold
        migration_commands = decide_migrations(comm, all_walker_counts, comm.load_balancer.variance_threshold)
        if !isnothing(migration_commands)
            my_sends, my_receives = migration_commands
            perform_segment_migration!(comm, w, my_sends, my_receives)

            if mpi_rank(comm) == 0
                println("Load balancing triggered at step $step, variance ≈ $(round(variance, digits=3))")
                # Print final lengths after balancing
                final_lengths = [length(seg) for seg in local_segs]
                final_total = sum(final_lengths)
                println("After balancing - Local segments: $final_lengths (total: $final_total)")
            end
        end
    end
end

```@docs
    decide_migrations(comm::LoadBalancedCommunicator, all_walker_counts, variance_threshold=2.0)

Calculates and distributes a migration plan to balance the walker load across MPI ranks.

The calculation is performed only on the root rank (rank 0). If the standard
deviation of walker counts (`all_walker_counts`) exceeds the `variance_threshold`,
a migration plan is generated to move "walkers" from overloaded to underloaded ranks.

The balancing strategy is greedy:
1.  Ranks are classified as "overloaded" (> 100.5% of mean) or "underloaded" (< 99.5% of mean).
2.  Each overloaded rank iterates through underloaded ranks, starting with the one
    with the largest deficit (mean - current_load).
3.  Walkers are transferred to fill deficits until the overloaded rank has no more
    excess walkers (walkers above the mean).

The complete `migration_plan` (a `Dict{Int, Vector{Tuple{Int, Int}}}` mapping 
`sender_rank => [(receiver_rank, amount), ...]`) is then distributed to all ranks 
using `distribute_migration_commands`.

# Arguments
- `comm::LoadBalancedCommunicator`: The MPI communicator wrapper.
- `all_walker_counts`: A vector containing the walker count for each MPI rank. 
  The index `i` is assumed to correspond to rank `i-1`.
- `variance_threshold=2.0`: The **standard deviation** threshold. Balancing is
  only triggered if `std(all_walker_counts) > variance_threshold`.

# Returns
- The output of `distribute_migration_commands(comm, migration_plan)`, which is
  expected to be the set of migration commands (sends/receives) for the *local* rank.
"""
```
function decide_migrations(comm::LoadBalancedCommunicator, all_walker_counts, variance_threshold=2.0)
    migration_plan = Dict{Int, Vector{Tuple{Int, Int}}}()

    if mpi_rank(comm) == 0
        mean_load = mean(all_walker_counts)
        current_std=std(all_walker_counts)
        println("DEBUG: mean_load = $mean_load")
        println("DEBUG: std = $current_std")

        if std(all_walker_counts) > variance_threshold
            #mean_load = mean(all_walker_counts)
            overloaded = findall(c -> c > mean_load * 1.005, all_walker_counts) .- 1
            underloaded = findall(c -> c < mean_load * 0.995, all_walker_counts) .- 1

             # If no underloaded ranks but we have overloaded, use below-mean ranks
            if isempty(underloaded) && !isempty(overloaded)
                underloaded = findall(c -> c < mean_load, all_walker_counts) .- 1
            end
            deficits = Dict(r => mean_load - all_walker_counts[r+1] for r in underloaded)

            for from_rank in overloaded
                excess = all_walker_counts[from_rank+1] - mean_load
                sorted_receivers = sort(collect(keys(deficits)); by=k->deficits[k], rev=true)
                for to_rank in sorted_receivers
                    if excess <= 0
                        break
                    end
                    if get(deficits, to_rank, 0.0) > 0
                        transfer_amount = min(excess, deficits[to_rank])
                        if transfer_amount > 0
                            plan_list = get!(migration_plan, from_rank, [])
                            push!(plan_list, (to_rank, Int(round(transfer_amount))))
                            excess -= transfer_amount
                            deficits[to_rank] -= transfer_amount
                        end
                    end
                end
            end
        end
    end

    return distribute_migration_commands(comm, migration_plan)
end
```@docs
    distribute_migration_commands(comm::LoadBalancedCommunicator, migration_plan)

Distributes a global migration plan from the root rank (0) to all other ranks.

This function operates in a controller-worker pattern:
-   **Rank 0 (Controller): Reads the complete `migration_plan`. It iterates
    through all other ranks, calculates their specific send/receive tasks,
    and sends these tasks using a 3-step MPI process.
-   **Ranks > 0 (Workers): Receive their specific tasks from rank 0.

The MPI communication protocol for each worker rank (`target_rank`) is:
1.  Tag 100: Rank 0 sends `[send_count, recv_count]` (a `Vector{Int}` of size 2).
    The worker receives this to know how much more data to expect.
2.  Tag 101: If `send_count > 0`, rank 0 sends the flattened data for all
    `send` operations. The worker receives and reconstructs its `my_sends` list.
3.  Tag 102: If `recv_count > 0`, rank 0 sends the flattened data for all
    `receive` operations. The worker receives and reconstructs its `my_receives` list.

Rank 0 computes its own `my_sends` and `my_receives` lists locally from the plan
without using MPI.

# Arguments
- `comm::LoadBalancedCommunicator`: The MPI communicator wrapper.
- `migration_plan`: A `Dict{Int, Vector{Tuple{Int, Int}}}`. This dictionary is
  expected to be fully populated *only on rank 0*. It maps a `sender_rank` to
  a list of `(receiver_rank, amount)` tuples.

# Returns
- `my_sends::Vector{Tuple{Int, Int}}`: A list of `(to_rank, amount)` tuples
  specifying where the *current* rank should send walkers.
- `my_receives::Vector{Tuple{Int, Int}}`: A list of `(from_rank, amount)` tuples
  specifying from where the *current* rank should expect to receive walkers.
```
function distribute_migration_commands(comm::LoadBalancedCommunicator, migration_plan)
    rank = mpi_rank(comm)
    comm_mpi = mpi_comm(comm)
    my_sends = Tuple{Int,Int}[]
    my_receives = Tuple{Int,Int}[]

    if rank == 0
        for target_rank in 1:(mpi_size(comm)-1)
            sends = get(migration_plan, target_rank, Tuple{Int,Int}[])

            receives = Tuple{Int,Int}[]
            for (from_rank, transfers) in migration_plan
                for (to_rank, amount) in transfers
                    if to_rank == target_rank
                        push!(receives, (from_rank, amount))
                    end
                end
            end

            send_count = length(sends)
            recv_count = length(receives)
            MPI.Send([send_count, recv_count], comm_mpi, dest=target_rank, tag=100)

            if send_count > 0
                send_data = vcat([[s[1], s[2]] for s in sends]...)
                MPI.Send(send_data, comm_mpi, dest=target_rank, tag=101)
            end
            if recv_count > 0
                recv_data = vcat([[r[1], r[2]] for r in receives]...)
                MPI.Send(recv_data, comm_mpi, dest=target_rank, tag=102)
            end
        end

        my_sends = get(migration_plan, 0, Tuple{Int,Int}[])
        for (from_rank, transfers) in migration_plan
            for (to_rank, amount) in transfers
                if to_rank == 0
                    push!(my_receives, (from_rank, amount))
                end
            end
        end
    else
        counts = Vector{Int}(undef, 2)
        MPI.Recv!(counts, comm_mpi; source=0, tag=100)
        send_count, recv_count = counts[1], counts[2]

        if send_count > 0
            send_data = Vector{Int}(undef, 2 * send_count)
            MPI.Recv!(send_data, comm_mpi; source=0, tag=101)
            for i in 1:2:(2 * send_count)
                push!(my_sends, (send_data[i], send_data[i+1]))
            end
        end

        if recv_count > 0
            recv_data = Vector{Int}(undef, 2 * recv_count)
            MPI.Recv!(recv_data, comm_mpi; source=0, tag=102)
            for i in 1:2:(2 * recv_count)
                push!(my_receives, (recv_data[i], recv_data[i+1]))
            end
        end
    end  
    return my_sends, my_receives
end
```@docs
    perform_segment_migration!(comm::LoadBalancedCommunicator, w::PDWorkingMemory, my_sends, my_receives)

Executes the migration plan by transferring `(Key, Weight)` pairs between ranks.

This function mutates the `PDWorkingMemory` `w` by:
1.  Deleting walkers (pairs) from local segments that are selected to be sent.
2.  Adding received walkers (pairs) to the appropriate local segments.

The migration follows a three-step process:

1.  Prepare Send Data: Iterates `my_sends`. For each `(to_rank, amount)` entry,
    it greedily selects `amount` pairs from the *start* of the `local_segments`.
    These selected pairs are **removed** from `w` and stored in a send buffer.
2.  Exchange Counts: Uses `MPI.Alltoall` to inform all other ranks exactly
    how many `Pair{K,W}` elements will be sent to them. It receives back a
    corresponding list of counts to expect from all other ranks.
3.  Exchange Data: Uses non-blocking `MPI.Isend` and `MPI.Irecv!` to perform
    the `Alltoallv` (all-to-all variable) transfer of the actual `Pair{K,W}` data.
    `MPI.Waitall` is called to ensure all transfers are complete.
4.  Integrate Received Data: Iterates through all received buffers. Each
    received `(k, v)` pair is integrated into `w` by re-hashing the key `k` with
    `fastrange_hash` to find its correct local segment. The weight `v` is then
    added to any existing weight for that key in that segment.

# Arguments
- `comm::LoadBalancedCommunicator`: The MPI communicator wrapper.
- `w::PDWorkingMemory`: The local working memory, which **will be modified in-place**.
- `my_sends::Vector{Tuple{Int, Int}}`: A list of `(to_rank, amount)` tuples
  specifying where the *current* rank must send walkers.
- `my_receives::Vector{Tuple{Int, Int}}`: A list of `(from_rank, amount)` tuples.
  (Note: This argument is used for the debug print on rank 0 but not for the
  core data transfer logic, which is driven by `MPI.Alltoall`).

# Returns
- `nothing` (the function modifies `w` in-place).
```
function perform_segment_migration!(comm::LoadBalancedCommunicator, w::PDWorkingMemory{K,V,W,S}, my_sends, my_receives) where {K,V,W,S}
    rank = mpi_rank(comm)
    comm_mpi = mpi_comm(comm)

    if rank == 0
        println("Migration plan - my_sends: $my_sends, my_receives: $my_receives")
    end

    # Get local segments
    local_segs = local_segments(w)

    # Prepare data to send
    send_data_per_rank = Dict{Int, Vector{Pair{K,W}}}()

    for (to_rank, amount) in my_sends
        pairs_to_send = Vector{Pair{K,W}}()
        remaining = amount

        # Select pairs from local segments
        for seg in local_segs
            if remaining <= 0
                break
            end

            seg_pairs = collect(pairs(seg))
            pairs_to_take = min(remaining, length(seg_pairs))

            if pairs_to_take > 0
                selected_pairs = seg_pairs[1:pairs_to_take]
                append!(pairs_to_send, selected_pairs)

                # Remove from local segment
                for (k, _) in selected_pairs
                    delete!(seg, k)
                end

                remaining -= pairs_to_take
            end
        end

        if !isempty(pairs_to_send)
            send_data_per_rank[to_rank] = pairs_to_send
        end
    end

    # Exchange data counts
    send_counts = zeros(Int, mpi_size(comm))
    for (to_rank, data) in send_data_per_rank
        send_counts[to_rank + 1] = length(data)
    end

    recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm_mpi)

    # Post receives and sends
    requests = MPI.Request[]
    recv_buffers = Dict{Int, Vector{Pair{K,W}}}()

    for from_rank in 0:(mpi_size(comm) - 1)
        if rank != from_rank && recv_counts[from_rank + 1] > 0
            buffer = Vector{Pair{K,W}}(undef, recv_counts[from_rank + 1])
            req = MPI.Irecv!(buffer, comm_mpi; source=from_rank, tag=3)
            push!(requests, req)
            recv_buffers[from_rank] = buffer
        end
    end

    for (to_rank, data) in send_data_per_rank
        req = MPI.Isend(data, comm_mpi; dest=to_rank, tag=3)
        push!(requests, req)
    end

    MPI.Waitall(requests)

    # Integrate received data into appropriate local segments
    total_segments = length(local_segs)
    for (_, buffer) in recv_buffers
        for (k, v) in buffer
            # Determine which local segment this should go to
            seg_idx = ((fastrange_hash(k, total_segments) - 1) % total_segments) + 1
            local_segs[seg_idx][k] = get(local_segs[seg_idx], k, W(0.0)) + v
        end
    end
end