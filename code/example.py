from itertools import permutations

import numpy as np
import dt_provided as dt
import dt_global as dtg
import dt_core as dtc
import dt_cv as dtcv
from anytree import Node, RenderTree

data = dt.read_data("data3.csv")
#input_feature_names = dtg.feature_names[:-1]
#print(dtc.choose_feature_split(data, input_feature_names))
#tree = dtc.learn_dt(data, input_feature_names)
#print(RenderTree(tree))
training_accuracy, validation_accuracy = dtcv.cv_post_prune([data[1:3], data[3:5]], [0.7, 0.8, 1])
print(training_accuracy)
print(validation_accuracy)
#input_feature_names = dtg.feature_names[:-1]
#tree = dtc.learn_dt(data, input_feature_names)
#dtc.post_prune(tree, 0.95)
#print(RenderTree(tree))
#for i in processed_data:
 #   print(dtc.choose_feature_split(i, ["attr1"]))