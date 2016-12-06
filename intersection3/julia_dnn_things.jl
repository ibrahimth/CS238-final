using DataFrames
using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport intention_pred_sumo as intent

include("./functions.jl")

function convert_intents(intent_preds)
    new_intent_preds = DataFrame(p1 = Float64[], p2 = Float64[])
    for i in 1:length(intent_preds)
        entry = intent_preds[i]
        this_intents = [convert(Float64,entry[1]), convert(Float64,entry[2])]
        push!(new_intent_preds, this_intents)
        #new_intent_preds = [new_intent_preds; this_intents]
    end
    new_intent_preds
end

function convert_sim_states_and_features_to_sarsp()
    all_states = readtable("simulated_states.csv")
    all_features = readtable("simulated_corresponding_features.csv")
    prev_sarsp = readtable("SARSP.csv") #to get actions and rewards
    intent_predictions = convert_intents(intent.johngetDNNbelief(Array(all_features)))
    println(size(intent_predictions))
    
    println(size(prev_sarsp))
    println(size(all_states))
    all_states =hcat(all_states, intent_predictions)
    println(size(all_states))
    new_sarsps = DataFrame(s = Int64[], a = Int64[], r = Int64[], sp = Int64[])
    prev_a = 1
    n = size(intent_predictions)[1]
    for i in 1:n
        sarsp_i = DataFrame(s = Int64[], a = Int64[], r = Int64[], sp = Int64[])
        cur_s = all_states[i,:]
        s, sub_dims = convertDiscreteState(cur_s)
        sarsp_i = [s[1], prev_sarsp[i,:a], prev_sarsp[i,:r], prev_sarsp[i,:sp]]
        if prev_a == 0
            new_sarsps[i-1,:sp] = s[1]
        end
        push!(new_sarsps, sarsp_i)
        prev_a = prev_sarsp[i,:a]
    end
    println(size(new_sarsps))
    writetable("SARSP_wPs.csv",new_sarsps)
end

convert_sim_states_and_features_to_sarsp()
