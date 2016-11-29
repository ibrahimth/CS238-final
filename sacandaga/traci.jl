using PyCall
using LightGraphs
using DataFrames

@pyimport traci
@pyimport sumolib.net as sumonet

df = DataFrame(VID = String[], TimeStep = Float64[], vel_x = Float64[], vel_y = Float64[],
  yaw = Float64[], numberOfLanesToMedian = Int64[],
  numberOfLanesToCurb = Int64[], headway = Float64[], distanceToIntersection = Float64[])

net = sumonet.readNet("osm.net.xml")

pos_I = (667.06, 3708.88)

TLSID = "41898831"


#TLS = net[:getTLSSecure](TLSID)
#print(TLS[:getEdges]())

sumoBinary = "/path/to/sumo-gui"
sumoCmd = ["sumo-gui", "-c", "osm.sumocfg"]

traci.start(sumoCmd)
step = 0
while step < 10000
   traci.simulationStep()
   vehicles = traci.vehicle[:getIDList]()
   for vehicle in vehicles
     pos = traci.vehicle[:getPosition](vehicle)
     dist = traci.simulation[:getDistance2D](pos[1], pos[2], pos_I[1], pos_I[2])
     if dist < 100
       speed = traci.vehicle[:getSpeed](vehicle)
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
      push!(df, [vehicle, step/10, vel_x, vel_y, yaw, numberOfLanesToMedian, numberOfLanesToCurb, headway, dist])
      println(vehicle, ", ", step/10, ", ", vel_x, ", ", vel_y, ", ", yaw, ", ", numberOfLanesToMedian, ", ", numberOfLanesToCurb, ", ", headway, ", ", dist)
     end
   end
   println(step)
   step += 1
 end
traci.close()
