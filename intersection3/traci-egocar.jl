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

TTI_policy = false
if "tti" in ARGS
    println("Doing with tti")
    TTI_policy = true
    #Was gonna try to do online Q learning for exploration, but nah
    #N = Array(convertDiscreteState(nothing))
    #Acts = [0,1]
    #Q = spzeros(N, 2)
    #N = spzeros(N, 2)
end

sarsp_df = DataFrame(s = Int64[], a = Int64[], r = Int64[], sp = Int64[])

#classifier = intent.loadDNNonly()
all_states = DataFrame(dist=Float64[], speed = Float64[], headway = Float64[], rearway=Float64[], p1 = Float64[], p2 = Float64[])
all_features = DataFrame(vid=Any[], fid=Float64[], vel_x=Float64[], vel_y=Float64[], Ax=Float64[], Ay=Float64[], yaw=Float64[], numberOfLanesToMedian=Float64[], numberOfLanesToCurb=Float64[], headway=Float64[], dist=Float64[], nextmove=Float64[])
start_sarsp_at = 1
num_sims = 5000
start_policy_at = 140
for i = 1:num_sims
  println((i*100)/num_sims, "%")

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
    policy = 0 #default
    if step > start_policy_at
        vehciles, dists, dists_sort, states, features = getSimulationInfo(step, n_tracked_cars, v_dict, i_dict)
        if !isempty(states[1])
            all_states = [all_states;states[1,:]]
            all_features = [all_features; features[1,:]]
            s, sub_dims = convertDiscreteState(states[1,:])
            if TTI_policy == true
                tti = features[1,:dist] / sqrt(features[1,:vel_x]^2 + features[1,:vel_y]^2)
                if tti > 2.0 || rand() > 0.80 #explore with 20% probability
                   policy = 1
                end
            else
                policy = Int(go <= step)
            end
        end
    end

    if policy == 0
      traci.vehicle[:slowDown]("ego1", 0, 100);
    #elseif step == go
    #  traci.vehicle[:slowDown]("ego1", 20, 5000);
    #  oncoming_cars = features[:, :vid]
    else
      traci.vehicle[:slowDown]("ego1", 20, 5000);
      oncoming_cars = features[:, :vid]
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
  println(reward)
  n = size(all_states)[1]
  for j = start_sarsp_at:n
      s, sub_dims = convertDiscreteState(all_states[j,:])
      push!(sarsp_df, [s[1], 0, -1, 0])
      if j > start_sarsp_at
        sarsp_df[end-1,:sp] = s[1]
      end
  end
  start_sarsp_at = n+1


  sarsp_df[end,:a] = 1
  sarsp_df[end,:r] = round(reward)
  #try
  #  sarsp_df[end,:r] = round(reward)
  #catch
  #  println("failed reward:", reward)
  #  sarsp_df[end,:r] = 0
  #end
  sarsp_df[end,:sp] = 0

  #might as well save every time
  writetable("SARSP.csv",sarsp_df)
  writetable("simulated_states.csv", all_states)
  writetable("simulated_corresponding_features.csv",all_features)
end

writetable("SARSP.csv",sarsp_df)
writetable("simulated_states.csv", all_states)
writetable("simulated_corresponding_features.csv",all_features)
