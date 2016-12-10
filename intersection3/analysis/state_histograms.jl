using DataFrames
using Discretizers
using Distributions
using PyPlot


states = readtable("simulated_states.csv")

for i = 1:4
  data = states[i]
  data = data[find(x -> x < 800, data)]
  n = 4
  if i <= 2
    n = 5
  end
  bins = binedges(DiscretizeUniformCount(n),data)
  println(bins)
  h = PyPlot.plt[:hist](data,50)
  sleep(10)
end


#data = readcsv("analysis/rewards_results_of_working2.policy")
# data = convert(Array{Float64,1},reshape(data,length(data)[1]))
# data = data[find(x -> x > -9000,data)]
# PyPlot.plt[:hist](data,40)
# writecsv("rewards_working2_noColisions.dat",data)
