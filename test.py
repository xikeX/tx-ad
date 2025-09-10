import pickle
import numpy as np
with open("data.pkl",'rb') as f:
    loaded_data = pickle.load(f)
    
    
with open("data_v1.pkl",'rb') as f:
    loaded_data_1 = pickle.load(f)
    

# def deep_equal(a, b):
#     """
#     递归比较两个对象是否深度相等。
#     支持类型：list, dict, int, float, str, bool, None
#     """
#     # 类型不同，直接不等
#     if type(a) != type(b):
#         return False

#     # 如果是字典
#     if isinstance(a, dict):
#         if set(a.keys()) != set(b.keys()):
#             return False
#         return all(deep_equal(a[k], b[k]) for k in a)

#     # 如果是列表或元组
#     if isinstance(a, (list, tuple,np.ndarray)):
#         if len(a) != len(b):
#             return False
#         return all(deep_equal(x, y) for x, y in zip(a, b))

#     # 如果是基本类型（int, float, str, bool, None 等）
#     try:
#         if a != b:
#             return False
#         # 特殊处理 float 的 nan
#         if isinstance(a, float) and a != a and b != b:  # nan != nan 为 True
#             return True
#         return True
#     except Exception as e:
#         print(e)
#         return False

#     return True
# deep_equal(loaded_data, loaded_data_1)
for i in range(len(loaded_data)):
    for j in range(len(loaded_data[i])):
        if loaded_data[i][j]!=loaded_data_1[i][j]:
            print(i,j)
        # for k in range(len(loaded_data[i][j])):
        #     if loaded_data[i][j][k]!=loaded_data_1[i][j][k]:
        #         print(i,j,k)