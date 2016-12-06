import intention_pred_sumo as ips

ips.loadDataTrainSaveDNN(csv_file="refined_turning_data.csv")
ips.testAccuracyDNN()
