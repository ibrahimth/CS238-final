using PyCall
using LightGraphs
using DataFrames

@pyimport traci
@pyimport sumolib.net as sumonet

df_t  = DataFrame(VID = String[], dist = Float64[], speed = Float64[])
df_nt = DataFrame(VID = String[], dist = Float64[], speed = Float64[])

net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()
#TLS = net[:getTLSSecure](TLSID)
#print(TLS[:getEdges]())

for i = 1:100
  run(`build.bat >nul`)
  #traci.start(["sumo", "-c", "i3.sumocfg"], label="sim1")
  #traci.start(["sumo", "-c", "i3.sumocfg"], label="sim2")
  traci.start(["sumo-gui", "-c" , "i3.sumocfg"])
  #traci.switch("sim1")
  traci.simulationStep()
  traci.vehicle[:moveToXY]("ego1","bottom_in", 0, pos_i[1], pos_i[2] - 6, 1)
  traci.vehicle[:setSpeedMode]("ego1", 01100)
  #traci.switch("sim2")
  #traci.simulationStep()
  #traci.vehicle[:moveToXY]("ego1","bottom_in", 0, pos_i[1], pos_i[2] - 6, 1)
  #traci.vehicle[:setSpeedMode]("ego1", 01100)
  step = 0
  last_step = 0
  go = 180
  oncoming_cars = Array(String,0)
  while true
     step += 1
     traci.simulationStep();
     pos_ego = traci.vehicle[:getPosition]("ego1")
     dist = traci.simulation[:getDistance2D](pos_ego[1], pos_ego[2], pos_i[1], pos_i[2])
     if step < go
       traci.vehicle[:slowDown]("ego1", 0, 100);

     elseif step == go
       vehicles = traci.vehicle[:getIDList]()
       if length(vehicles) < 2
         break;
       end
       println(vehicles)
       dists = Float64[]
       for vehicle in vehicles
         if vehicle == "ego1"
           continue
         end
         pos = traci.vehicle[:getPosition](vehicle)
         push!(dists, traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2]))
       end
       dists_sort = sort(dists)
       for i = 1:2
         push!(oncoming_cars, vehicles[find(x -> x == dists_sort[i],dists)][1])
       end
       println(oncoming_cars)
       traci.vehicle[:slowDown]("ego1", 20, 5000);

     else
       traci.vehicle[:slowDown]("ego1", 20, 5000);
       for vehicle in oncoming_cars
         pos = traci.vehicle[:getPosition](vehicle)
         dist_v = traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2])
         push!(df_t, [vehicle, dist_v, traci.vehicle[:getSpeed](vehicle)])
         #println([vehicle, dist_v, traci.vehicle[:getSpeed](vehicle)])
       end
     end
     if dist > 25
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
    if step == go
      vehicles = traci.vehicle[:getIDList]()
      println(vehicles)
    elseif step > go
      for vehicle in oncoming_cars

        pos = traci.vehicle[:getPosition](vehicle)
        dist_v = traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2])
        push!(df_nt, [vehicle, dist_v, traci.vehicle[:getSpeed](vehicle)])
        #println([vehicle, dist_v, traci.vehicle[:getSpeed](vehicle)])
      end
    end
  end
 traci.close()
 #writetable("turn.csv", df_turn)
 #writetable("noturn.csv", df_noturn)
 reward = 0
 for car in unique(convert(Array,df_nt[:VID]))
   df_t_s = df_t[find(x -> x == car, df_t[:VID]), :]
   df_nt_s = df_nt[find(x -> x == car, df_nt[:VID]), :]
   diffs = abs(df_nt_s[:dist] - df_t_s[:dist])
   diff = diffs[length(diffs)]
   #println(diff)
   reward -= 10*diff
 end
 println(reward)
end
