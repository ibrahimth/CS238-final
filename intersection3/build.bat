netconvert --node-files=i3.nod.xml --edge-files=i3.edg.xml --output-file=i3.net.xml
jtrrouter --flow-files=i3.flows.xml --turn-ratio-files=i3.turns.xml --net-file=i3.net.xml --output-file=i3.rou.xml --begin 0 --end 1000 --random true
