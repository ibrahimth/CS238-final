using PyCall
using LightGraphs
using DataFrames

@pyimport traci
@pyimport sumolib.net as sumonet

net = sumonet.readNet("i3.net.xml")
pos_i = net[:getNode]("center")[:getCoord]()
#TLS = net[:getTLSSecure](TLSID)
#print(TLS[:getEdges]())

sumoCmd = [ "sumo-gui", "-c" , "i3.sumocfg"]

traci.start(sumoCmd)
traci.simulationStep();
traci.vehicle[:moveToXY]("ego1","bottom_in", 0, pos_i[1], pos_i[2] - 6, 1)
step = 0
while step < 10000
   step += 1
   traci.simulationStep();
   pos_ego = traci.vehicle[:getPosition]("ego1")
   dist = traci.simulation[:getDistance2D](pos_ego[1], pos_ego[2], pos_i[1], pos_i[2])
   println(dist)
   if dist < 11
     traci.vehicle[:slowDown]("ego1", 0, 100);
   end

  #  if mod(step,100) > 50
  #    traci.vehicle[:slowDown]("ego1", 10, 1000);
  #  else
  #    traci.vehicle[:slowDown]("ego1", 0, 1000);
  #  end

 end
traci.close()

writetable("car_turning_data.csv", df)
