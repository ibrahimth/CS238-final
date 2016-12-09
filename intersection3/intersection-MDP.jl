using POMDPs
using Distributions
using DataFrames

include("adaptedproject2code.jl")
include("functions.jl")
function buildTransisitinDict(df)
  Transitions = Dict{Tuple,Array{Int64,1}}()
  for i in 1:size(df)[1]
    println(i)
    if haskey(Transitions, (df[i,:s],df[i,:a]))
      push!(Transitions[(df[i,:s],df[i,:a])], df[i, :sp] , df[i , :r])

    else
      Transitions[(df[i,:s],df[i,:a])] = [df[i, :r] , df[i , :sp]]
    end
  end
  for key in keys(Transitions)
    A = reshape(Transitions[key], 2, convert(Int64, round(length(Transitions[key])/2)))
    for sp in unique(A[2,:])
      p = length(find(x -> x==sp, A[2,:]))/length(A[2,:])
      #println(p)
    end
  end
  return Transitions
end

#sarsp_df = readtable("SARSP.csv")
#policy_name = "final_q.policy"
sarsp_df = readtable("SARSP_wPs.csv")
policy_name = "final_q_wPs.policy"
#sleep(2)

function update_policy(policy, sub_dims, d, s, h, r, p1, p2)
    new_s = sub2ind(sub_dims, d, s, h, r, p1, p2)
    policy[new_s] = 2
end

function new_nearest_neighbor(orig_policy, sub_dims)
    policy = deepcopy(orig_policy) # to make sure no domino effect
    for s in 1:length(policy)
        if orig_policy[s] == 2
            state = ind2sub(sub_dims, s)
            dist_d, speed_d, h_d, r_d, p1_d, p2_d = state
            for new_d in dist_d+1:sub_dims[1]
                update_policy(policy, sub_dims, new_d, speed_d, h_d, r_d, p1_d, p2_d)
                for new_v in 1:speed_d-1
                    update_policy(policy, sub_dims, new_d, new_v, h_d, r_d, p1_d, p2_d)
                end
                for new_h in h_d+1:sub_dims[3]
                    update_policy(policy, sub_dims, new_d, speed_d, new_h, r_d, p1_d, p2_d)
                end
                for new_r in r_d+1:sub_dims[4]
                    update_policy(policy, sub_dims, new_d, speed_d, h_d, new_r, p1_d, p2_d)
                end
            end
            for new_v in 1:speed_d-1
                update_policy(policy, sub_dims, dist_d, new_v, h_d, r_d, p1_d, p2_d)
                for new_h in h_d+1:sub_dims[3]
                    update_policy(policy, sub_dims, dist_d, new_v, new_h, r_d, p1_d, p2_d)
                end
                for new_r in r_d+1:sub_dims[4]
                    update_policy(policy, sub_dims, dist_d, new_v, h_d, new_r, p1_d, p2_d)
                end
            end
            for new_h in h_d+1:sub_dims[3]
                update_policy(policy, sub_dims, dist_d, speed_d, new_h, r_d, p1_d, p2_d)
            end
            for new_r in r_d+1:sub_dims[4]
                update_policy(policy, sub_dims, dist_d, speed_d, h_d, new_r, p1_d, p2_d)
            end


            for p1 in 1:p1_d
                for p2 in 1:p2_d
                    update_policy(policy, sub_dims, dist_d, speed_d, h_d, r_d, p1, p2)
                    continue #was too lazy to comment out the rest...
                    for new_d in dist_d+1:sub_dims[1]
                        update_policy(policy, sub_dims, new_d, speed_d, h_d, r_d, p1, p2)
                        for new_v in 1:speed_d-1
                            update_policy(policy, sub_dims, new_d, new_v, h_d, r_d, p1, p2)
                        end
                        for new_h in h_d+1:sub_dims[3]
                            update_policy(policy, sub_dims, new_d, speed_d, new_h, r_d, p1, p2)
                        end
                        for new_r in r_d+1:sub_dims[4]
                            update_policy(policy, sub_dims, new_d, speed_d, h_d, new_r, p1, p2)
                        end
                    end
                    for new_v in 1:speed_d-1
                        update_policy(policy, sub_dims, dist_d, new_v, h_d, r_d, p1, p2)
                        for new_h in h_d+1:sub_dims[3]
                            update_policy(policy, sub_dims, dist_d, new_v, new_h, r_d, p1, p2)
                        end
                        for new_r in r_d+1:sub_dims[4]
                            update_policy(policy, sub_dims, dist_d, new_v, h_d, new_r, p1, p2)
                        end
                    end    
                end
            end
        end
    end
    return policy
end

function final_proj_q(data; state_range = 1)
    n, sub_dims = convertDiscreteState(nothing, dimens=true)
    states = collect(1:n)
    actions = [0,1]
    learning_rate = 0.1
    discount = 1.0
    Q = qlearning(data, discount, learning_rate, states, actions, 1000)
    new_actions = [1,2]
    policy = find_policy_from_values(Q, states, new_actions, n)
    policy = new_nearest_neighbor(policy, sub_dims)
    policy -= 1
    Q_full = full(Q)
    #for i = 1:size(Q_full)[1]
    #  println(Q_full[i,:], ind2sub((5,5,5,5),i))
    #end
    writecsv(policy_name, policy)
end

final_proj_q(sarsp_df)
