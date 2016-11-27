using PyCall

@pyimport traci

sumoBinary = "/path/to/sumo-gui"
sumoCmd = ["sumo-gui", "-c", "testworld.sumo.cfg"]

traci.start(sumoCmd)
step = 0
while step < 1000
   traci.simulationStep()
   step += 1
 end
traci.close()
