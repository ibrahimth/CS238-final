using Discretizers

function get_sorted_distances(vehicles, pos_i; ego = "ego1")
    dists = Float64[]
    for vehicle in vehicles
        if vehicle == "ego1"
            continue
        end
        pos = traci.vehicle[:getPosition](vehicle)
        push!(dists, traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2]))
    end
    dists_sort = sort(dists)
    return dists, dists_sort
end

#separate function for ease of editting, not sure if headway (getLeader) is right
function get_state(vehicleid, dist_to_inter, prev_v, intent_p; classifier=nothing, dt=0.1)
    headway = traci.vehicle[:getLeader](vehicleid,100)
    if typeof(headway) != Void
        headway = headway[2]
    else
        headway = 1000.0
    end
    features = get_features(vehicleid, pos_i, prev_v, dt=dt)
    if intent_p == nothing && false #did this for sim so inent pred not done
      if classifier != nothing
        #intention_pred = intent.johngetDNNbelief(features, classifier)[1]
      else
        #intention_pred = intent.johngetDNNbelief(features)[1]
      end
    else
      intention_pred = intent_p
    end
    return [dist_to_inter, traci.vehicle[:getSpeed](vehicleid),headway, 1000, 0.0,0.0], features #intention_pred[1], intention_pred[2]], features #rearwaydefault=100
end

function append_rearway(state)
    nrows, _ = size(state)
    for row_i in 1:nrows-1
        state[row_i,:rearway] = state[row_i+1,:headway]
    end
end

#given the list of vehicles and their sorted distances, returns first n vehicles with time to intersection > 1.7
function get_tracked_cars_state(vehicles, dists, dists_sort, v_dict, i_dict, redo_i; n=1, tti_min=1.7, classifier = nothing)
    state = DataFrame(dist=Float64[], speed = Float64[], headway = Float64[], rearway=Float64[], p1 = Float64[], p2 = Float64[])
    features_df = DataFrame(vid=Any[], fid=Float64[], vel_x=Float64[], vel_y=Float64[], Ax=Float64[], Ay=Float64[], yaw=Float64[], numberOfLanesToMedian=Float64[], numberOfLanesToCurb=Float64[], headway=Float64[], dist=Float64[], nextmove=Float64[])
    for i = 1:n_tracked_cars
        #find the next closest car
        car_to_add = ""
        this_dist = 0
        while car_to_add == "" && length(dists_sort) > 0
            next_closest = vehicles[find(x -> x == dists_sort[1], dists)][1]
            this_dist = dists_sort[1]
            tti = this_dist / traci.vehicle[:getSpeed](next_closest)
            dists_sort = deleteat!(dists_sort, 1)
            if tti > tti_min
                car_to_add = next_closest
            end
        end
        if car_to_add != ""
            prev_v = get(v_dict, car_to_add, (0,0))
            if redo_i
                intent_p = nothing
            else
                intent_p = get(i_dict, car_to_add, nothing)
            end
            new_state, features = get_state(car_to_add, this_dist, prev_v, intent_p, classifier = classifier, dt= timestep)
            #push!(new_state, i) #when doing by order, djp changed down stream to just add first state to all states
            push!(state, new_state)
            push!(features_df, features)
            i_dict[car_to_add] = (new_state[5], new_state[6])
        end
    end
    append_rearway(state)
    return state, features_df
end

function get_features(vehicle, pos_i, prev_v,; dt=0.1)
    pos = traci.vehicle[:getPosition](vehicle)
    dist = convert(Float64,traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2]))
    speed = convert(Float64,traci.vehicle[:getSpeed](vehicle))
    yaw = convert(Float64,traci.vehicle[:getAngle](vehicle))
    vel_x = convert(Float64,speed*cos(yaw*pi/180))
    vel_y = convert(Float64,speed*sign(yaw*pi/180))
    Ax = (vel_x - prev_v[1]) / dt
    Ay = (vel_y - prev_v[2]) / dt
    headway = traci.vehicle[:getLeader](vehicle,100)
    if typeof(headway) != Void
        headway = convert(Float64,headway[2])
    else
        headway = 1000.0
    end
    laneID = traci.vehicle[:getLaneID](vehicle)
    edgeID = traci.lane[:getEdgeID](laneID)
    if edgeID[1] != ':'
        edge = net[:getEdge](edgeID)
        n_lanes = edge[:getLaneNumber]()
        laneInd = traci.vehicle[:getLaneIndex](vehicle)
        numberOfLanesToMedian = convert(Float64,n_lanes - 1 - laneInd)
        numberOfLanesToCurb = convert(Float64,laneInd)
    else
        numberOfLanesToCurb = -1.0
        numberOfLanesToMedian = -1.0
    end
    #here features should be vid fid vx vy ax ay lanesMed, lanesCurb, yaw, hdwy, dist move
    #in python after reordering feautres should be lanesMed, lanesCub, Vy, Ay, Vx, Ax, yaw, hdwy, dist, fid, vid, move
    return [vehicle, 0, vel_x, vel_y, Ax, Ay, yaw, numberOfLanesToMedian, numberOfLanesToCurb, headway, dist, 0]
end


#checks to see if the ego car comes too close to other cars
function checkForcollisions()
  collision = false
  pos_ego = traci.vehicle[:getPosition]("ego1")
  for vehicle in traci.vehicle[:getIDList]()
    #:center_12_0 is the lane for turning right
    #:center_14_0 is the lane for turning left
    #:center_13_0 is the lane for going straight

    laneID = traci.vehicle[:getLaneID](vehicle)
    if laneID == ":center_13_0" ||  laneID == ":center_14_0"
      pos_o = traci.vehicle[:getPosition](vehicle)
      dist = traci.simulation[:getDistance2D](pos_o[1], pos_o[2], pos_ego[1], pos_ego[2])
      if dist < 2
        println("collision detected")
        collision = true
      end
    end
  end
  return collision
end

#reruns the simulation to look at the differences in ending conditions for the cars
function calculateReward(end_dists, last_step, collision, oncoming_cars)
  if collision
    return -10000
  end
  traci.start(["sumo", "-c" , "i3.sumocfg"])
  traci.simulationStep()
  traci.vehicle[:moveToXY]("ego1","bottom_in", 0, pos_i[1], pos_i[2] - 6, 1)
  traci.vehicle[:setSpeedMode]("ego1", 01100)
  step = 0
  go = 180
  while step < last_step
    step += 1
    traci.simulationStep();
    pos_ego = traci.vehicle[:getPosition]("ego1")
    dist = traci.simulation[:getDistance2D](pos_ego[1], pos_ego[2], pos_i[1], pos_i[2])
    traci.vehicle[:slowDown]("ego1", 0, 100);
  end
  i = 0
  for vehicle in oncoming_cars
    i += 1
    pos = traci.vehicle[:getPosition](vehicle)
    end_dists[i,2] = dist = traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2])
  end
  traci.close()
  reward = -10 * sum(abs(end_dists[:,2] - end_dists[:,1]))
  if reward == NaN
    reward = 0
  end
  return reward
end

#convets a single line of the states dataframe into a state using sub2ind. Also returns the sub_dims array which
#contains the number of possible values for each state
function convertDiscreteState(state)
  disc_d = LinearDiscretizer([0,30,45,60,75,100])
  disc_v = LinearDiscretizer([0,4,7,10,13,100])
  disc_h = LinearDiscretizer([0,10,30,50,70,100])
  disc_r = LinearDiscretizer([0,10,30,50,70,100])
  disc_p = LinearDiscretizer([0.0, 0.05, 0.25, 0.5, 0.75, 0.95])
  dist = state[:dist]
  speed = state[:speed]
  head = state[:headway]
  rear = state[:rearway]
  p1 = state[:p1]
  p2 = state[:p2]
  dist_d = encode(disc_d, round(dist))
  speed_d = encode(disc_v, round(speed))
  head_d = encode(disc_h, round(head))
  rear_d = encode(disc_r, round(rear))
  p1_d = encode(disc_p, p1)
  p2_d = encode(disc_p, p2)
  sub_dims = (nlabels(disc_d), nlabels(disc_v), nlabels(disc_h), nlabels(disc_p), nlabels(disc_p))
  #sub_dims = (nlabels(disc_d), nlabels(disc_v), nlabels(disc_h), nlabels(disc_r), nlabels(disc_p), nlabels(disc_p))
  state = sub2ind(sub_dims, dist_d, speed_d, head_d, rear_d, p1_d, p2_d)
  return state, sub_dims
end

function convertDiscreteStateNoP(state)
  disc_d = LinearDiscretizer([0,30,45,60,75,100])
  disc_v = LinearDiscretizer([0,4,7,10,13,100])
  disc_h = LinearDiscretizer([0,10,30,50,70,100])
  disc_r = LinearDiscretizer([0,10,30,50,70,100])
  dist = state[:dist]
  speed = state[:speed]
  head = state[:headway]
  rear = state[:rearway]
  dist_d = encode(disc_d, round(dist))
  speed_d = encode(disc_v, round(speed))
  head_d = encode(disc_h, round(head))
  rear_d = encode(disc_r, round(rear))
  sub_dims = (nlabels(disc_d), nlabels(disc_v), nlabels(disc_h))
  #sub_dims = (nlabels(disc_d), nlabels(disc_v), nlabels(disc_h), nlabels(disc_r), nlabels(disc_p), nlabels(disc_p))
  state = sub2ind(sub_dims, dist_d, speed_d, head_d, rear_d)
  return state, sub_dims
end
