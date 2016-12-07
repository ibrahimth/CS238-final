using PyCall
using LightGraphs
using DataFrames


#unshift!(PyVector(pyimport("sys")["path"]), "")

@pyimport traci
@pyimport sumolib.net as sumonet
#@pyimport intention_pred_sumo as intent

include("./functions.jl")

net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()

n_tracked_cars = 2
timestep = 0.1


sarsp_df = DataFrame(s = Int64[], a = Int64[], r = Int64[], sp = Int64[])

#classifier = intent.loadDNNonly()
all_states = DataFrame(dist=Float64[], speed = Float64[], headway = Float64[], rearway=Float64[], p1 = Float64[], p2 = Float64[])
all_features = DataFrame(vid=Any[], fid=Float64[], vel_x=Float64[], vel_y=Float64[], Ax=Float64[], Ay=Float64[], yaw=Float64[], numberOfLanesToMedian=Float64[], numberOfLanesToCurb=Float64[], headway=Float64[], dist=Float64[], nextmove=Float64[])
start_sarsp_at = 0
for i = 1:300

  println(i/10, "%")

  randomizeRoutes()
  initSimulation(gui = false)
  collision = false
  reward = 0
  step = 0
  last_step = 0
  go = 175 + round(rand()*10)
  oncoming_cars = Array(String,0)
  end_dists = Array(Float64,n_tracked_cars,2)
  df = DataFrame(dist = Float64[], speed = Float64[], headway = Float64[])
  v_dict = Dict() #to keep track of prev velocities to calculate acceleration
  i_dict = Dict() #to keep track of prev intentions to avoid recalculating every time step
  while true
    step += 1
    traci.simulationStep();
    pos_ego = traci.vehicle[:getPosition]("ego1")
    if step > 130 && step < go
      vehciles, dists, dists_sort, states, features = getSimulationInfo(step, n_tracked_cars, v_dict, i_dict)
      if !isempty(states[1])
        all_states = [all_states; states[1,:]] #only care about the first one
        all_features = [all_features;features[1,:]] #especially for this one
      end
    end

    if step < go
      traci.vehicle[:slowDown]("ego1", 0, 100);
    elseif step == go
      traci.vehicle[:slowDown]("ego1", 20, 5000);
      oncoming_cars = features[:, :vid]
    else
      traci.vehicle[:slowDown]("ego1", 20, 5000);
      collision = checkForcollisions()
      if collision
        break
      end
    end
    dist = traci.simulation[:getDistance2D](pos_ego[1], pos_ego[2], pos_i[1], pos_i[2])
    if dist > 25
     i = 0
     #finds the final distance for the cars
     for vehicle in oncoming_cars
       i += 1
       pos = traci.vehicle[:getPosition](vehicle)
       end_dists[i,1] = dist = traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2])
     end
     last_step = step
     break
    end
  end
  traci.close()

  reward = calculateReward(end_dists, last_step, collision, oncoming_cars)
  #println(size(all_states))
  for j = start_sarsp_at:length(all_states[1])
      s, sub_dims = convertDiscreteStateNoP(all_states[j,:])
      push!(sarsp_df, [s[1], 0, -1, 0])
      if j > 1
        sarsp_df[end-1,:sp] = s[1]
      end
  end
  #println(size(sarsp_df))


  sarsp_df[end,:a] = 1
  try
    sarsp_df[end,:r] = round(reward)
  catch
    sarsp_df[end,:r] = 0
  end
  sarsp_df[end,:sp] = 0
  start_sarsp_at = length(all_states[1])+1

end

writetable("SARSP.csv",sarsp_df)
writetable("simulated_states.csv", all_states)
writetable("simulated_corresponding_features.csv",all_features)
