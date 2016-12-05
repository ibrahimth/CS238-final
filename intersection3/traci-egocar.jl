using PyCall
using LightGraphs
using DataFrames

@pyimport traci
@pyimport sumolib.net as sumonet

net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()

n_tracked_cars = 2
timestep = 0.1

df_array = Array(DataFrame,0)
for i = 1:100
  try
    run(`build.bat >nul`)
  catch
    run(`./build.bat`)
  end
  traci.start(["sumo-gui", "-c" , "i3.sumocfg"])
  traci.simulationStep()
  traci.vehicle[:moveToXY]("ego1","bottom_in", 0, pos_i[1], pos_i[2] - 6, 1)
  traci.vehicle[:setSpeedMode]("ego1", 01100)
  step = 0
  last_step = 0
  go = 180
  oncoming_cars = Array(String,0)

  end_dists = Array(Float64,n_tracked_cars,2)
  df = DataFrame(dist = Float64[], speed = Float64[], headway = Float64[])
  while true

    step += 1
    traci.simulationStep();
    pos_ego = traci.vehicle[:getPosition]("ego1")
    dist = traci.simulation[:getDistance2D](pos_ego[1], pos_ego[2], pos_i[1], pos_i[2])

    if step < go
      traci.vehicle[:slowDown]("ego1", 0, 100);
    elseif step == go
      traci.vehicle[:slowDown]("ego1", 20, 5000);
      vehicles = traci.vehicle[:getIDList]()
      dists = Float64[]
      for vehicle in vehicles
        if vehicle == "ego1"
          continue
        end
        pos = traci.vehicle[:getPosition](vehicle)
        push!(dists, traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2]))
      end
      dists_sort = sort(dists)
      for i = 1:n_tracked_cars
        #find the next closest car
        car_found = false
        car_to_add = "ego1"
        while car_found == false && length(dists_sort) > 0
            next_closest = vehicles[find(x -> x == dists_sort[1], dists)][1]
            tti = dists_sort[1] / traci.vehicle[:getSpeed](next_closest) 
            dists_sort = deleteat!(dists_sort, 1)
            if tti > 1.7
                car_found = true
                car_to_add = next_closest
                println(car_to_add)
            end
        end
        if car_to_add != "ego1"
            push!(oncoming_cars, car_to_add)#vehicles[find(x -> x == dists_sort[i],dists)][1])
        end
      end
    else
      traci.vehicle[:slowDown]("ego1", 20, 5000);
      #check for colision



    end
    if dist > 25
     i = 0

     #finds the final distance for the cars
     for vehicle in oncoming_cars
       println(typeof(vehicle))
       i += 1
       pos = traci.vehicle[:getPosition](vehicle)
       end_dists[i,1] = dist = traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2])
     end
     last_step = step
     break
    end
    end
  traci.close()

  #now simulate again, but without having the car go
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
  println(reward)
end
