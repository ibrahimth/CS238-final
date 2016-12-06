using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport intention_pred_sumo as intent
intent.testAccuracyDNN()
