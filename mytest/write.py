import numpy as np
import os

# 定义外参矩阵
extrinsics = np.array([
    [500.,   0., 512.],
    [  0., 500., 288.],
    [  0.,   0.,   1.]
], dtype=np.float32)

# 创建目标文件夹（如果不存在）
save_path = "/home/tione/notebook/TrajectoryCrafter/data/mytest"
os.makedirs(save_path, exist_ok=True)

# 保存为 .npy 文件
np.save(os.path.join(save_path, "intrinsics.npy"), extrinsics)