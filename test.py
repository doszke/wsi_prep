from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

args=  [6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
vals=  [145, 145, 145, 145, 147, 147, 149, 149, 160, 150, 151, 152, 153, 153, 154, 154, 154, 153, 153, 158, 154, 154, 154, 153, 152, 152, 152, 146, 146, 146, 145, 145]
a, b, _, _, _ = stats.linregress(args, vals)

plt.plot(np.array(args), np.array(vals), np.array(args), a * np.array(args) + b)
plt.show()