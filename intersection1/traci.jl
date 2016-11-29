using PyCall

@pyimport traci
@pyimport sumolib.net as sumonet

net = sumonet.readNet("osm.net.xml")
TLSID = "cluster_191230877_301049333_301049388_65518534"
#TLS = net[:getTLSSecure](TLSID)
#print(TLS[:getEdges]())

sumoBinary = "/path/to/sumo-gui"
sumoCmd = ["sumo-gui", "-c", "osm.sumocfg"]

traci.start(sumoCmd)
step = 0
print(traci.trafficlights[:getRedYellowGreenState](TLSID))
while step < 10000

   traci.simulationStep()

   print(traci.trafficlights[:getPhase](TLSID))
   print(traci.trafficlights[:getRedYellowGreenState](TLSID))

   step += 1
 end
traci.close()
