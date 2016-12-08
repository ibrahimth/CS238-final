using DataFrames

include("./functions.jl")

sarsp_df = readtable("SARSP.csv")
policy_name = "final_q_rand.policy"
#policy_name = "final_q_tti.policy"

nstates = convertDiscreteState(nothing)

if policy_name == "final_q_rand.policy"
    policy = ones(Int64, nstates)
    writecsv(policy_name, policy)
end

