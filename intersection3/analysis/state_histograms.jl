using DataFrames
using Discretizers
using Distributions
using PyPlot


states = readtable("simulated_states.csv")

for i = 1:4
  data = states[i]
  data = data[find(x -> x < 800,data)]
  n = 4
  if i <= 2
    n = 5
  end
  bins = binedges(DiscretizeUniformWidth(n),data)
  println(bins)
  #h = PyPlot.plt[:hist](data,50)
end
