import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from functions import *

[X, y] = Loadtxt('data.txt')
[X, mu, s] = Normalize(X)
[Theta, J_hist] = GradientDescent(X,y,0.2,400)
input = np.array([1,3,1000])
input = (input-mu)/s
#Lưu ý sửa lại x0 = 1
input[0] = 1
predict = predict(input,Theta)
print('%.2f$'%(predict))
