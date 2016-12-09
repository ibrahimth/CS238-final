using DataFrames

include("./functions.jl")

sarsp_df = readtable("SARSP.csv")
policy_name = "final_q_rand.policy"
if "tti" in ARGS
    policy_name = "final_q_tti.policy"
end


function update_policy(policy, sub_dims, d, s, h, r, p1, p2)
    new_s = sub2ind(sub_dims, d, s, h, r, p1, p2)
    policy[new_s] = 1
end

function update_for_all_h_r_p1_p2(policy, d, v, sub_dims)
    for h in 1:sub_dims[3]
        for r in 1:sub_dims[4]
            for p1 in 1:sub_dims[5]
                for p2 in 1:sub_dims[6]
                    update_policy(policy, sub_dims, d,v,h,r,p1,p2)
                end
            end
        end
    end
    return policy
end

function add_TTI(policy, sub_dims) 
    v = 1
    for d in 2:sub_dims[1]
        policy = update_for_all_h_r_p1_p2(policy, d,v, sub_dims)
    end
    v = 2
    for d in 3:sub_dims[1]
        policy = update_for_all_h_r_p1_p2(policy, d,v, sub_dims)
    end
    for v in 3:5
        for d in 4:sub_dims[1]
            policy = update_for_all_h_r_p1_p2(policy, d,v, sub_dims)
        end
    end
    return policy
end

nstates, sub_dims = convertDiscreteState(nothing, dimens=true)

if policy_name == "final_q_rand.policy"
    policy = ones(Int64, nstates)
else
    policy = zeros(Int64, nstates)
    add_TTI(policy, sub_dims)
end
writecsv(policy_name, policy)
