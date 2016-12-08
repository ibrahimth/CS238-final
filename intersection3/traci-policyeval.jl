using PyCall
using LightGraphs
using DataFrames

#unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport traci
@pyimport sumolib.net as sumonet

include("./functions.jl")

doing_intentions = true
doing_baseline = false   #options: false, "rand", "tti"
policy_name = "final_q.policy"
if doing_intentions
    println("Evaluating with intentions")
    @pyimport intention_pred_sumo as intent
    policy_name = "final_q_wPs.policy"
    classifier = intent.loadDNNonly()
elseif doing_baseline != false
    println("Doing baseline:", doing_baseline)
    policy_name = string("final_q_rand.policy") #tti wont use, need to make sure loads fine
end

net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()

start_policy_at = 150
n_trials = 2000
n_tracked_cars = 2
timestep = 0.1
rewards = zeros(Float64, n_trials)
policy_array = readcsv(policy_name)
policy_array = convert(Array{Int64,1},reshape(policy_array,length(policy_array)[1]))
num_collisions = 0
num_reward_issues = 0
num_sim_issues = 0
avg_wait = 0
start_time = now()
for j = 1:n_trials
  println(j/n_trials * 100, "%")
  try
    randomizeRoutes()
    initSimulation(gui = false)
  except
    print("Caught issue initializing simulation...")
    j -= 1
    num_sim_issues += 1
    continue
  end
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
    try
        traci.simulationStep();
    except
        print("Caught issue in simulation taking a step...")
        step -= 1
        num_sim_issues += 1
        continue
    end
    
    pos_ego = traci.vehicle[:getPosition]("ego1")
    if step <= start_policy_at
      traci.vehicle[:slowDown]("ego1", 0, 100);
    end

    if step > start_policy_at && policy == 0
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
        s, sub_dims = convertDiscreteState(states[1,:]) #modified so it handles no probs
        s = s[1]
        policy = 0
        try
          policy = policy_array[s]
        catch
          println("error assigning policy state:",s)
        end
        if doing_baseline == "tti"
            policy = 0
            tti = features[1,:dist] / sqrt(features[1,:vel_x]^2 + features[1,:vel_y]^2)
            if tti >= 3.0
                policy = 1
            end
        end
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
  this_wait = last_step - start_policy_at
  avg_wait += convert(Float64,(this_wait - avg_wait)) / j
  try
      reward += calculateReward(end_dists, last_step, collision, last_tracked_cars)
  catch
      println("error with calculating reward")
      reward -= 5000
      num_reward_issues += 1
  end
  if reward == NaN
      println("NaN reward, setting to -5000")
      reward = -5000
      num_reward_issues += 1
  end
  rewards[j] = reward
  println(reward)
  if j % 10 == 0 #save checkpoint
    fin = now()
    duration = fin - start_time
    save_file = string("results_of_",policy_name)
    f_rewards = open(string("rewards_", save_file), w)
    writedlm(f_rewards, rewards)
    close(f_rewards)
    
    f = open(save_file, "w")
    writedlm(f, [string(duration), j, mean(rewards), num_collisions, avg_wait, num_reward_issues, num_sim_issues])
    close(f)
  end
end
fin = now()
duration = fin - start_time
println("Took: ", duration, " to run ", n_trials, " trials, with ", num_reward_issues, " issues with the reward.")
println("Encountered ", num_sim_issues, " issues with the simulation.")
println("Average Reward: ", mean(rewards), ", Average wait of: ", avg_wait)
println("Number of collisions: ", num_collisions)
save_file = string("results_of_",policy_name)
f_rewards = open(string("rewards_", save_file), w)
writedlm(f_rewards, rewards)
close(f_rewards)
f = open(save_file, "w")
writedlm(f, [string(duration), n_trials, mean(rewards), num_collisions, avg_wait, num_reward_issues, num_sim_issues])
close(f)
