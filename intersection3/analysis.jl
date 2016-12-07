using DataFrames

df_new = DataFrame(VID = Float64[], TimeStep = Float64[], vel_x = Float64[], vel_y = Float64[],
  accel_x = Float64[], accel_y = Float64[], yaw = Float64[], numberOfLanesToMedian = Int64[],
  numberOfLanesToCurb = Int64[], headway = Float64[], distanceToIntersection = Float64[],
  turn = String[])

df = readtable("car_turning_data.csv")


delta_t = df[2,:TimeStep]- df[1,:TimeStep]
i = 0
n_cars = length(unique(df[:VID]))
for car in unique(convert(Array,df[:VID]))
  i += 1
  df_small = df[find(x -> x == car, df[:VID]), :]
  yaw1 = df_small[indmin(df_small[:TimeStep]), :yaw]
  yaw2 = df_small[indmax(df_small[:TimeStep]), :yaw]
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
  df_small[1,:accel_x] = 0
  df_small[1,:accel_y] = 0
  for t in 2:length(df_small[1])

    accel_x = (df_small[t,:vel_x] - df_small[t-1,:vel_x])/delta_t
    accel_y = (df_small[t,:vel_y] - df_small[t-1,:vel_y])/delta_t
    if t > 2
        if df_small[t-2, :VID] == df_small[t, :VID] && df_small[t, :TimeStep] - df_small[t-2, :TimeStep] == 2 * delta_t
                #make sure its the same car
            accel_x = df_small[t-1, :accel_x] + df_small[t-2, :accel_x] + accel_x
            accel_x /= 3.0
            accel_y = df_small[t-1, :accel_y] + df_small[t-2, :accel_y] + accel_y
            accel_y /= 3.0
        end
    end
    df_small[t,:accel_x] = accel_x
    df_small[t,:accel_y] = accel_y
    df_small[t,:turn] = turn

  end
  append!(df_new,df_small[2:length(df_small[1]),:])
  println(i*100/n_cars)
end
println("done")
writetable("refined_turning_data.csv", df_new)
