import pandas as pd
import numpy as np
PERIOD=252
arr=np.array([8,1,2,3,4,5,6,7,9])
max_index2 = np.argsort(arr)[-2]
print(max_index2)