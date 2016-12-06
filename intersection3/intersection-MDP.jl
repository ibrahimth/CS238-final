using POMDPs
using Distributions
using DataFrames


sarsp_df = readtable("SARSP.csv")

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
      println(p)
    end
  end
  return Transitions
end

T = buildTransisitinDict(sarsp_df)
