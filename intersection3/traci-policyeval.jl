using PyCall
using LightGraphs
using DataFrames

#unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport traci
@pyimport sumolib.net as sumonet

include("./functions.jl")

doing_intentions = false
policy_name = "final_q.policy"
if doing_intentions
    println("Evaluating with intentions")
    @pyimport intention_pred_sumo as intent
    policy_name = "final_q_wPs.policy"
    classifier = intent.loadDNNonly()
end

net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()


n_trials = 200
n_tracked_cars = 2
timestep = 0.1
rewards = zeros(Float64, n_trials)
policy_array = readcsv(policy_name)
policy_array = convert(Array{Int64,1},reshape(policy_array,length(policy_array)[1]))
num_collisions = 0
num_reward_issues = 0
start_time = now()
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
      vehicles, dists, dists_sort, states, features = getSimulationInfo(step, n_tracked_cars, v_dict, i_dict)

      if !isempty(states[1])
        if doing_intentions
            if step % 5 == 0 || !haskey(i_dict, vehicles[1])
                array_features = ["0", "0", features[1,:vel_x], features[1,:vel_x], features[1,:Ax], 
                                  features[1,:Ay], features[1,:yaw], features[1,:numberOfLanesToMedian], 
                                  features[1,:numberOfLanesToCurb], features[1,:headway], features[1,:dist], 0]
                        #just using convert didnt work
                intents_pred = intent.johngetDNNbelief(array_features, classifier)[1]
                i_dict[vehicles[1]] = (intents_pred[1], intents_pred[2])
            end
            states[1,:p1] = i_dict[vehicles[1]][1]
            states[1,:p2] = i_dict[vehicles[1]][1]
        end
        s, sub_dims = convertDiscreteStateNoP(states[1,:])
        s = s[1]

        policy = policy_array[s]
        reward += -1
      end
    end

    if policy == 1
      traci.vehicle[:slowDown]("ego1", 20, 5000);
      collision = checkForcollisions()
      if collision
        num_collisions += 1
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

  try
      reward += calculateReward(end_dists, last_step, collision, last_tracked_cars)
  catch
      println("error with calculating reward")
      reward -= 1000
      num_reward_issues += 1
  end
  if reward == NaN
    println("NaN reward, setting to -1000")
    reward = -1000
  end
  rewards[j] = reward
  println(reward)
end
fin = now()
duration = fin - start_time
println("Took: ", duration, " to run ", n_trials, " trials, with ", num_reward_issues, " issues with the reward.")
println("Average Reward: ", mean(rewards))
println("Number of collisions: ", num_collisions)
save_file = string("results_of_",policy_name)
f = open(save_file, "w")
writedlm(f, [string(duration), n_trials, mean(rewards), num_collisions, number_reward_issues])
close(f)
