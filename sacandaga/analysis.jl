using DataFrames

df = readtable("car_turning_data.csv")
df_small = df[find(x -> x == "veh23", df[:VID]), :]

yaw1 = df_small[indmin(df_small[:TimeStep]), :yaw]
yaw2 = df_small[indmax(df_small[:TimeStep]), :yaw]
println(yaw1)
println(yaw2)
dyaw = yaw2-yaw1

if dyaw > 180
  dyaw = dyaw -360
end
if dyaw < -180
  dyaw = dyaw +360
end
if abs(dyaw) < 40
  turn = "straight"
elseif dyaw < 0
  turn = "left"
elseif dyaw > 0
  turn = "right"
end
println(dyaw)
println(turn)
println("done")
