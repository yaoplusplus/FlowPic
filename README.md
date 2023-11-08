# FlowPic: A Generic Representation for Encrypted Traffic Classification and Applications Identification

# JointFeature

1. 运行 utils.py-make_joint_features,得到数据集
2. 运行EIMTC-main.py分割数据集
3. 最后修改config.py， 将feature_method设置为JointFeature，模型设置为PureClassifier, 运行main.py

# Joint

1. 运行 EIMTc-data_preprocess.py-make_joint_dataset,得到数据集
2. 同上(模型是FlowPicNet_adaptive,注意设置好in_features)

# FlowPic

# MyFlowPic