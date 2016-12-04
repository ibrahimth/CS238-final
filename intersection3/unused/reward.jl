using DataFrames
df_t = readtable("turn.csv")
df_nt = readtable("noturn.csv")

for car in unique(convert(Array,df_nt[:VID]))
  df_t_s = df_t[find(x -> x == car, df_t[:VID]), :]
  df_nt_s = df_nt[find(x -> x == car, df_nt[:VID]), :]
  diffs = df_nt_s[:dist] - df_t_s[:dist]
  println(diffs[length(diffs)])
end
