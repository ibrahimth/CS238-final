using DataFrames
using DataStructures

small_file = "./small.csv"
med_file = "./medium.csv"
large_file = "./large.csv"

function find_small_policy(data)
    actions = [1,2,3,4]
    discount = 0.95
    #state = sub2ind((10,10), x, y)
    R, T, sps = count_R_T_sps(data, d_o_c=3.0)
    Us = valueIteration(data, actions, R, T, discount,sps)
    println(Us)
    policy = valuesToPolicy(Us, actions, discount, R, T, sps)
    println("Policy:", policy)
    writecsv("./small.policy", policy)
end

function prioruninformed_T()
    T = DefaultDict(Any, Float64, 0) #sp, s, a
    nT_sasp = DefaultDict(Any, Int64, 0) #sp, s, a
    nT_sa = DefaultDict(Any, Int64, 0) #s, a
    sps = DefaultDict(Any, Vector{Int}, Vector{Int})
    return T, nT_sasp, nT_sa, sps
end

function priorDeterministic_T(size; degree_of_certainty=1.0) #degree of certainty = 0 is completely uninformed
    T = DefaultDict(Any, Float64, 0) #sp, s, a
    nT_sasp = DefaultDict(Any, Int64, 0) #sp, s, a
    nT_sa = DefaultDict(Any, Int64, 0) #s, a
    sps = DefaultDict(Any, Vector{Int}, Vector{Int})
    if size == "small"
        for row in 1:10
            for col in 1:10
                state = sub2ind((10,10), row, col)
                action = 1#left = 1
                if col>1
                    sp = sub2ind((10,10), row, col-1)
                    T[(sp,state,action)]=1.0
                    nT_sasp[(sp,state,action)]=degree_of_certainty
                    nT_sa[(state,action)]=degree_of_certainty
                    push!(sps[(state,action)], sp)
                end
                action += 1#right = 2
                if col<10
                    sp = sub2ind((10,10), row, col+1)
                    nT_sasp[(sp,state,action)]=degree_of_certainty
                    nT_sa[(state,action)]=degree_of_certainty
                    push!(sps[(state,action)], sp)
                end
                action += 1#   up = 3
                if row>1
                    sp = sub2ind((10,10), row-1, col)
                    T[(sp,state,action)]=1.0
                    nT_sasp[(sp,state,action)]=degree_of_certainty
                    nT_sa[(state,action)]=degree_of_certainty
                    push!(sps[(state,action)], sp)
                end
                action += 1#down = 4
                if row<10
                    sp = sub2ind((10,10), row+1, col)
                    T[(sp,state,action)]=1.0
                    nT_sasp[(sp,state,action)]=degree_of_certainty
                    nT_sa[(state,action)]=degree_of_certainty
                    push!(sps[(state,action)], sp)
                end
            end
        end
    else:
        println("unsuported deterministic prior size")
    end
    return T, nT_sasp, nT_sa, sps
end

function count_R_T_sps(data; med=false, d_o_c=1.0)
    R = DefaultDict(Any, Float64, 0)
    nR = DefaultDict(Any, Int64, 0)
    if length(unique(collect(data[:s]))) == 100
        println("small")
        T, nT_sasp, nT_sa, sps = priorDeterministic_T("small", degree_of_certainty=d_o_c)
    elseif med
        println("med")
        T, nT_sasp, nT_sa, sps = priorDeterministicDynamic_T("med", degree_of_certainty=d_o_c)
    else
        T, nT_sasp, nT_sa, sps = prioruninformed_T()
    end
    for row in eachrow(data)
        state = row[:s]
        action = row[:a]
        sp = row[:sp]
        r = row[:r]
        #update expected rewards
        n = nR[(state, action)] + 1
        nR[(state, action)] = n
        prev_tot_r = (n-1) * R[(state, action)]
        R[(state, action)] = float(prev_tot_r + r) / n

        #do the same for transitions probs
        nT_sasp[(sp, state, action)] += 1
        nT_sa[(state,action)] += 1
        #record this sp as a possible sp for state and action
        if nT_sasp[(sp, state, action)] == 1
            push!(sps[(state,action)], sp)
        end
        #must update all Tprobs
        for sp in sps[(state,action)]
            T[(sp, state, action)] = float(nT_sasp[(sp, state, action)]) / nT_sa[(state,action)]
        end
    end
    return R, T, sps
end

function valueIteration(df, actions, Reward, T, discount, sps) #returns values
    U = DefaultDict(Any, Float64, 0)
    newU = DefaultDict(Any, Float64, 0)
    states = unique(collect(df[:s]))
    println(length(states))
    k = 0
    numIters = 10000
    converged = false
    converge_tol = 0
    while k < numIters && !converged
        converged = true
        if k % 100 == 0
            println("iteration: ", k)
        end
        U = deepcopy(newU)
        for state ∈ states
            max_U = typemin(Float64)
            for a in actions
                state_U = Reward[(state,a)]
                to_add = 0
                sum_t_prob = 0
                for sp in sps[(state, a)]
                    tprob = T[(sp, state, a)]
                    sum_t_prob += tprob
                    util = U[sp]
                    to_add += discount * (tprob * util)
                end
                #println(sum_t_prob)
                to_add /= sum_t_prob#length(sps[(state, a)])
                state_U += to_add
                if state_U > max_U
                    max_U = state_U
                end
            end
            if converged && abs(U[state] - max_U) > converge_tol
                converged = false
            end
            newU[state] = max_U
        end
        k += 1
    end
    println("took ", k, " iterations to converge")
    return U
end

function valuesToPolicy(Us, actions, discount, R, T, sps)
    max_state = 0
    for state in keys(Us)
        max_state = max(state, max_state)
    end
    policy = zeros(Int64, max_state)
    for state in keys(Us)
        state_max = typemin(Float64)
        for action in actions
            this_R = R[(state, action)]
            for sp in sps[(state, action)]
                this_R += discount * T[(sp, state, action)]*Us[sp]
            end
            if this_R > state_max
                policy[state] = action
                state_max = this_R
            end
        end
    end
    return policy
end

function valuesToPolicy_list(Us, actions, discount, R, T, sps)
    policy = zeros(Int64, length(Us))
    for state in 1:length(Us)
        state_max = typemin(Float64)
        for action in actions
            this_R = R[(state, action)]
            for sp in sps[(state, action)]
                this_R += discount * T[(sp, state, action)]*Us[sp]
            end
            if this_R > state_max
                policy[state] = action
                state_max = this_R
            end
        end
    end
    return policy
end

function interpolate_v(values, n, state_range, actions; weights=[abs(i-state_range) for i in 1:state_range*2+1]) #weights must be of len(staterange*2)
    for state in 1:n
        start = max(state-state_range, 1)
        fin = min(state+state_range, n)
        for action in actions
            #values keys are state, action
            neighbor_values = zeros(Float64, fin-start)
            these_weights = zeros(Float64, fin-start)
            weight_start_offset = state_range-(state-start)
            for i in 1:(fin-start)
                neighbor_values[i] = values[(start+i-1,action)]
                these_weights[i] = weights[i+weight_start_offset]
            end
            values[(state, action)] = dot(neighbor_values, these_weights)
        end
    end
    return values
end

function interpolate_v_valued(values, n, state_range, actions)
    newQ = deepcopy(values)

    for state in 1:n
        if state % 1000 == 0
            println("interpolated state:", state)
        end
        for action in actions
            new_val = float(values[state, action])
            weight_total = 1.0
            num_found = 0
            for i in 1:n
                w = 1.0/i
                first_val = 0
                if state-i >= 1
                    first_val = values[state-i,action]
                end
                second_val = 0
                if state+i <= n
                    second_val = values[state+i,action]
                end
                new_val += (first_val + second_val) * w
                if first_val != 0
                    weight_total += w
                    num_found += 1
                end
                if second_val != 0
                    weight_total += w
                    num_found += 1
                end
                if num_found >= state_range
                    break
                end
            end
            newQ[state,action] = new_val / weight_total
        end
    end
    return newQ
end


function interpolate_p(policy, n)
    for state in 1:n
        if policy[state] == 0
            prev_state = state-1
            prev_action = 0
            while prev_state > 0 && prev_action == 0
                prev_action = policy[prev_state]
                prev_state -= 1
            end
            next_state = state+1
            next_action = 0
            while next_state <= n && next_action == 0
                next_action = policy[next_state]
                next_state += 1
            end
            diff_L = abs(state - prev_state)
            diff_R = abs(next_state - state)
            if prev_action != 0 && next_action != 0
                action = (float(diff_L * prev_action) + float(diff_R * next_action)) / (diff_L + diff_R)
                policy[state] = Int(round(action))
            else
                policy[state] = Int(max(prev_action, next_action)) #one of them will not be 0
            end
        end
    end
    return [convert(Int64, policy[state]) for state in 1:n]
end

function find_nexts(Q, nstates, actions, begin_i) #this finds the next closest state with assigned value for each of the actions
    nexts = Dict()
    actions_to_find = deepcopy(actions)
    for state in begin_i:nstates
        actions_found = []
        for action in actions_to_find
            if Q[state, action] != 0
                nexts[action] = state
                push!(actions_found, action)
            end
        end
        if length(actions_found) == 0
           continue
        end
        for a in actions_found
            deleteat!(actions_to_find, findin(actions_to_find, a))
        end
        if length(actions_to_find) == 0
            return nexts
        end
    end
    nexts
end

function find_next(Q, nstates, action, start) #this finds the next closest state with assigned value for one action
    for state in start:nstates
        if Q[state, action] != 0
            return state
        end
    end
    return nstates
end

#DOESNT ACTUALLY FIND, replaces 0s with nearest neighbor
function find_nearest_neighbor(Q, states, actions, n)
    newQ = deepcopy(Q)
    left = find_nexts(Q, n, actions, 1) #returns first state with each of the actions
    right = deepcopy(left)
    for state in 1:n
        if state % 10000 == 0
            println("nearest neighbor for state: ", state)
        end
        for action in actions
            if Q[state,action] == 0.0
                if state >= right[action]
                    left[action] = deepcopy(right[action])
                    right[action] = find_next(Q, n, action, right[action]+1)
                end
                state_left = left[action]
                state_right = right[action]
                if abs(state-state_left) <= abs(state-state_right)
                    newQ[state,action] = Q[state_left, action]
                else
                    newQ[state,action] = Q[state_right, action]
                    left[action] = deepcopy(state_right)
                    right[action] = find_next(Q, n, action, state_right+1)
                end
            end
        end
    end
    newQ
end

function find_nearest_nonzero(policy, nstates, current_state; defalt=4)
    for i in 1:nstates
        left = current_state-i
        right = current_state+i
        if left > 0 && policy[left] != 0
            return policy[left]
        elseif right <= nstates && policy[right] != 0
            return policy[right]
        end
    end
    return default
end

function initializeQN(data, n, a)
    #Q = DefaultDict(Any, Float64, 0) #s, a
    Q = spzeros(n, a)
    #N = DefaultDict(Any, Float64, 0) #s, a
    N = spzeros(n, a)
    return Q, N
end

function sarsalambda(data, λ, γ, α, states, actions; batch=10000)
    Q, N = initializeQN(data, length(states), length(actions))
    t = 0
    num_iters = 100
    prev_mean_Q::Float64 = 0.0
    num_rows::Int = nrow(data)
    this_mean = 0.0
    tolerance = 1e-9
    while t < num_iters
        updated_states = []
        i = 1
        println("Iteration: ", t, " Q_mean: ", prev_mean_Q)
        for row in eachrow(data)
            state = row[:s]
            action = row[:a]
            sp = row[:sp]
            r = float(row[:r][1])
            N[state,action] += 1.0
            δ = r - Q[state, action]
            push!(updated_states, (state, action))
            next_s = 0
            if i < num_rows
                next_s = data[i+1,:]
            end
            if next_s == sp
                next_a = data[i+1,:a]
                δ += γ * Q[sp, next_a]
                Q[state,action] += α*δ*N[state,action]
            else
                #Q[state,action] += α*δ*N[state,action]
                for (state, action) in updated_states
                    #(state, action) = updated_states[end-j+1] #left over from when i was doing something stupid
                    Q[state,action] += α*δ*N[state,action]
                    N[state,action] = γ * λ * N[state,action]
                end
                updated_states = []
            end
            i+=1
        end
        t += 1
        this_mean::Float64 = mean(Q)
        if abs(this_mean - prev_mean_Q) < tolerance
            break
        end
        prev_mean_Q = this_mean
    end
    Q
end

function sarsa(data, γ::Float64, α::Float64, states, actions; batch::Int=10000)
    Q, N = initializeQN(data, length(states), length(actions))
    t::Int = 0
    num_iters::Int = 12000
    prev_mean_Q::Float64 = 0.0
    num_rows::Int = nrow(data)
    this_mean = 0.0
    tolerance = 1e-30
    while t < num_iters
        i = 1
        println("Iteration: ", t, " Q_mean: ", prev_mean_Q)
        for row in eachrow(data)
            next_s = 0
            qnext = 0.0
            if i < num_rows
                next_s = data[i+1,:s]::Int
            end
            if next_s == row[:sp]::Int
                qnext = Q[next_s, data[i+1,:a]]::Float64
            end
            Q[row[:s]::Int,row[:a]::Int] = Q[row[:s]::Int,row[:a]::Int] + α*(row[:r]::Int64+γ*qnext - Q[row[:s]::Int,row[:a]::Int])
            i+=1
        end
        t += 1
        this_mean::Float64 = mean(Q)
        if abs(this_mean - prev_mean_Q) < tolerance
            break
        end
        prev_mean_Q = this_mean
    end
    Q
end

function qlearning(data, γ::Float64, α::Float64, states, actions; batch::Int=10000)
    Q, N = initializeQN(data, length(states), length(actions))
    t::Int = 0
    num_iters::Int = 12000
    prev_mean_Q::Float64 = 0.0
    num_rows::Int = nrow(data)
    this_mean = 0.0
    tolerance = 1e-30
    while t < num_iters
        println("Iteration: ", t, " Q_mean: ", prev_mean_Q)
        for row in eachrow(data)
            Q[row[:s]::Int,row[:a]::Int] += α*(row[:r]::Int64 + γ*maximum(Q[row[:sp]::Int,:])::Float64 - Q[row[:s]::Int,row[:a]::Int])
        end
        t += 1
        this_mean::Float64 = mean(Q)
        if abs(this_mean - prev_mean_Q) < tolerance
            break
        end
        prev_mean_Q = this_mean
    end
    Q
end


function find_policy_from_values(Q, states, actions, n; default=4)
    policy = zeros(Int64, n)  #DOING POLICY INTERPOLATION
    for state in states
        if state % 100000 == 0
            println("policy for state: ", state)
        end
        max_val = typemin(Float64)
        max_action = 0
        for action in actions
            val = Q[state,action]
            if val == 0
                continue
            end
            if val > max_val
                max_val = val
                max_action = action
            end
        end
        policy[state] = max_action
    end
    orig = deepcopy(policy)
    for state in states
        if policy[state] == 0
            policy[state] = find_nearest_nonzero(orig, n, state; defalt=default)
        end
    end
    policy
end

function find_med_sars2(data; state_range=10, batch_size=1000)
    n = 50000
    states = collect(1:n)
    actions = collect(1:7)
    discount = 1
    λ = 0.99
    learning_rate = 0.3
    Q = sarsalambda(data, λ, discount, learning_rate, states, actions, batch=batch_size)
    if state_range > 1
        Q = interpolate_v_valued(Q, n, state_range, actions)
    end
    policy = find_policy_from_values(Q, states, actions, n; default=4)
    writecsv("./medium_sarsalambda.policy", policy)
end


function find_med_sarsa(data; state_range = 1)
    n = 50000
    states = collect(1:n)
    actions = collect(1:7)
    learning_rate = 0.33
    discount = 1.0
    Q = sarsa(data, discount, learning_rate, states, actions)
    if state_range > 1
    #    Q = find_nearest_neighbor(Q, states, actions, n)
    #else
        Q = interpolate_v_valued(Q, n, state_range, actions)
    end
    policy = find_policy_from_values(Q, states, actions, n; default=4)
    writecsv("./medium_sarsa.policy", policy)
end

function find_med_q(data; state_range = 1)
    n = 50000
    states = collect(1:n)
    actions = collect(1:7)
    learning_rate = 0.33
    discount = 1.0
    Q = qlearning(data, discount, learning_rate, states, actions)
    if state_range > 1
        Q = interpolate_v_valued(Q, n, state_range, actions)
    end
    policy = find_policy_from_values(Q, states, actions, n; default=4)
    writecsv("./medium_q.policy", policy)
end

function find_large_sars(data; state_range=4)
    discount = 0.99
    n = 1757600
    states = collect(1:n)
    actions = collect(1:8)
    λ = 0.95
    learning_rate = 0.33
    Q = sarsalambda(data, λ, discount, learning_rate, states, actions)
    if state_range > 1
        Q = interpolate_v_valued(Q, n, state_range, actions)
    end
    policy = find_policy_from_values(Q, states, actions, n; default=4)
    writecsv("./large_sarsalambda.policy", policy)
end

function find_large_sarsa(data; state_range = 1)
    discount = 0.99
    n = 1757600
    states = collect(1:n)
    actions = collect(1:8)
    learning_rate = 0.33
    Q = sarsa(data, discount, learning_rate, states, actions)
    if state_range > 1
        Q = interpolate_v_valued(Q, n, state_range, actions)
    end
    policy = find_policy_from_values(Q, states, actions, n; default=4)
    writecsv("./large_sarsa.policy", policy)
end

function find_large_q(data; state_range = 1)
    discount = 0.99
    n = 1757600
    states = collect(1:n)
    actions = collect(1:8)
    learning_rate = 0.33
    Q = qlearning(data, discount, learning_rate, states, actions)
    if state_range > 1
        Q = interpolate_v_valued(Q, n, state_range, actions)
    end
    policy = find_policy_from_values(Q, states, actions, n; default=4)
    writecsv("./large_q.policy", policy)
end

function main()
    size = ""
    while true
        println("pick size (small, med, large):")
        size = readline(STDIN)[1:end-(length("\n"))]
        println(size)
        if size in ["small", "med", "large"]
            break
        elseif size == "q"
            break
        end
        println("invalid size: ", size)
        println("q to quit")
    end

    if size == "small"
        data = readtable(small_file)
        return find_small_policy(data)
    elseif size == "med"
        data = readtable(med_file)
        while true
            println("pick type: sarsa_nearest, sarsa_range_x_d, sarsalambda_x_d, qlearn, where x is range, d is number of digits in x. d must be in single digits")
            choice = readline(STDIN)[1:end-(length("\n"))]
            if contains(choice,"sarsa_nearest")
                return find_med_sarsa(data, state_range=1)
            elseif contains(choice,"sarsalambda")
                num_digits = parse(Int, choice[end])
                range = parse(Int,choice[end-(1+num_digits):end-2])
                println("range=", range)
                return find_med_sars2(data, state_range = range)
            elseif contains(choice,"sarsa_range")
                num_digits = parse(Int, choice[end])
                range = parse(Int,choice[end-(1+num_digits):end-2])
                return find_med_sarsa(data, state_range=range)
            elseif contains(choice, "qlearn")
                return find_med_q(data, state_range=1)
            end
            println("invalid choice: ", choice)
            println("q to quit")
        end
    elseif size == "large"
        data = readtable(large_file)
        while true
            println("pick type: sarsa_nearest, sarsa_range_x_d, sarsalambda_x_d, qlearn, where x is range, d is number of digits in x. d must be in single digits")
            choice = readline(STDIN)[1:end-(length("\n"))]
            if contains(choice,"sarsa_nearest")
                return find_large_sarsa(data, state_range=1)
            elseif contains(choice,"sarsalambda")
                num_digits = parse(Int, choice[end])
                range = parse(Int,choice[end-(1+num_digits):end-2])
                println("range=", range)
                return find_large_sars(data, state_range=range)
            elseif contains(choice,"sarsa_range")
                println("unsupported")
                continue
                num_digits = parse(Int, choice[end])
                range = parse(Int,choice[end-(1+num_digits):end-2])
                return find_large_sarsa(data, state_range=range)
            elseif contains(choice, "qlearn")
                return find_large_q(data, state_range=1)
            end
            println("invalid choice: ", choice)
            println("q to quit")
        end
        find_large_sarsa(df_large)
    end
end
main()
#data = readtable(med_file)
#find_med_sarsa(data, state_range=1)
