# ltfs-av
LTFS loan default predictions competition hosted by AV

GBM model:
generate_feats1.py --> does some preprocessing on given data
generate_base_feather.py --> save prepared data as feather file for faster loading

trainer_v10.py --> LGB Script with some basic feature engineerings

NN_model:
The idea here was to create three different set of features:
1. Embedding for all categorical features using w2v (word 2 vec was used to capture codependence of many categorical features)
2. Autooencoders for getting representations from bureau features
3. Perform rank transform on other continous features and min max transform on count features

generate_w2v_feats.py and generate_bureau_features.py do 1 and 2 respectively.

nn_model runs a simpe 3 layer neural net after concatenating all features


