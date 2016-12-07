using PyCall
using LightGraphs
using DataFrames

#unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport traci
@pyimport sumolib.net as sumonet

include("./functions.jl")

net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()

n_trials = 20
n_tracked_cars = 2
timestep = 0.1
rewards = zeros(Float64, n_trials)
policy_array = readcsv("working1.policy")
policy_array = convert(Array{Int64,1},reshape(policy_array,length(policy_array)[1]))
for j = 1:n_trials
  println(j/n_trials * 100, "%")
  randomizeRoutes()
  initSimulation(gui = false)
  collision = false
  policy = 0
  reward = 0
  step = 0
  last_step = 0
  end_dists = Array(Float64,n_tracked_cars,2)
  last_tracked_cars = Array(String,0)
  v_dict = Dict() #to keep track of prev velocities to calculate acceleration
  i_dict = Dict() #to keep track of prev intentions to avoid recalculating every time step
  while true
    step += 1
    traci.simulationStep();
    pos_ego = traci.vehicle[:getPosition]("ego1")
    if step <= 130
      traci.vehicle[:slowDown]("ego1", 0, 100);
    end

    if step > 130 && policy == 0
      traci.vehicle[:slowDown]("ego1", 0, 100);
      vehciles, dists, dists_sort, states, features = getSimulationInfo(step, n_tracked_cars, v_dict, i_dict)

      if !isempty(states[1])
        s, sub_dims = convertDiscreteStateNoP(states[1,:])
        s = s[1]
      end

      policy = policy_array[s]
      reward += -1
    end

    if policy == 1
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
     for vehicle in features[:,:vid]
       i += 1
       pos = traci.vehicle[:getPosition](vehicle)
       end_dists[i,1] = dist = traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2])
     end
     last_tracked_cars = features[:,:vid]
     last_step = step
     break
    end
  end
  traci.close()

  reward += calculateReward(end_dists, last_step, collision, last_tracked_cars)

  rewards[j] = reward
  println(reward)
end
println(mean(rewards))
