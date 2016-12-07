using POMDPs
using Distributions
using DataFrames

include("adaptedproject2code.jl")
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

sarsp_df = readtable("SARSP_wPs.csv")
policy_name = "final_q_wPs.policy"
#sleep(2)


function final_proj_q(data; state_range = 1)
    n = maximum(data[:s])
    states = collect(1:n)
    actions = [0,1]
    learning_rate = 0.1
    discount = 1.0
    Q = qlearning(data, discount, learning_rate, states, actions, 1000)
    new_actions = [1,2]
    policy = find_policy_from_values(Q, states, new_actions, n; default=20) #includes nearest neighbor
    policy -= 1
    Q_full = full(Q)
    #for i = 1:size(Q_full)[1]
    #  println(Q_full[i,:], ind2sub((5,5,5,5),i))
    #end
    writecsv(policy_name, policy)
end

final_proj_q(sarsp_df)
