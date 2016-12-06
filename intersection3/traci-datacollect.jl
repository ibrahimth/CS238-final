using PyCall
using LightGraphs
using DataFrames

@pyimport traci
@pyimport sumolib.net as sumonet


df = DataFrame(VID = String[], TimeStep = Float64[], vel_x = Float64[], vel_y = Float64[],
  accel_x = Float64[], accel_y = Float64[], yaw = Float64[], numberOfLanesToMedian = Int64[],
  numberOfLanesToCurb = Int64[], headway = Float64[], distanceToIntersection = Float64[],
  turn = String[])

#BUILD FOR DATA IS SET TO 4200 TIME STEPS, CHANGE SIM LENGTH THERE
try
  run(`build_for_data.bat >nul`)
catch
  run(`./build_for_data.bat`) #for max
end
net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()

sumoCmd = [ "sumo", "-c" , "i3.sumocfg"]

traci.start(sumoCmd)
traci.simulationStep();
traci.vehicle[:remove]("ego1")
step = 0
while step < 42000 #this should be end in .bat / time_step
   traci.simulationStep();
   vehicles = traci.vehicle[:getIDList]()
   for vehicle in vehicles
     pos = traci.vehicle[:getPosition](vehicle)
     dist = traci.simulation[:getDistance2D](pos[1], pos[2], pos_i[1], pos_i[2])
     if dist < 100
       speed = traci.vehicle[:getSpeed](vehicle)
       traci.vehicle[:setColor](vehicle, (255,0,0,0))
       yaw = traci.vehicle[:getAngle](vehicle)
       vel_x = speed*cos(yaw*pi/180)
       vel_y = speed*sign(yaw*pi/180)
       headway = traci.vehicle[:getLeader](vehicle,100)
       if typeof(headway) != Void
         headway = headway[2]
       else
         headway = NA
       end
       laneID = traci.vehicle[:getLaneID](vehicle)
       edgeID = traci.lane[:getEdgeID](laneID)
       if edgeID[1] != ':'
         edge = net[:getEdge](edgeID)
         n_lanes = edge[:getLaneNumber]()
         laneInd = traci.vehicle[:getLaneIndex](vehicle)
         numberOfLanesToMedian = n_lanes - 1 - laneInd
         numberOfLanesToCurb = laneInd
      else
        numberOfLanesToCurb = NA
        numberOfLanesToMedian = NA
      end
      push!(df, [vehicle, step/10, vel_x, vel_y, NA, NA, yaw, numberOfLanesToMedian, numberOfLanesToCurb, headway, dist, NA])
      ##println(vehicle, ", ", step/10, ", ", vel_x, ", ", vel_y, ", ", yaw, ", ", numberOfLanesToMedian, ", ", numberOfLanesToCurb, ", ", headway, ", ", dist)
     end
   end
   if mod(step, 50) == 0
     println(step)
   end
   step += 1
 end
traci.close()
df[1,:turn] = "fff"
df[1,:accel_x] = 0.0
df[1,:accel_y] = 0.0
writetable("car_turning_data.csv", df)
