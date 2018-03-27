from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

x = [i for i in range(1, 10)] + [10, 15, 20, 25, 50, 75]
print (x)
y = [0.71246, 0.717818, 0.716757, 0.715755, 0.730048,
     0.732322, 0.736068, 0.749008, 0.747566, 0.741046,
     0.754327, 0.753425, 0.749539, 0.754177, 0.764063]

print (len(x), len(y))
plt.plot(x, y, 'bo-', label='data')

x_pick = [1, 3, 5, 10, 15, 25, 50, 75]
y_pick = [0.71246, 0.716757, 0.730048, 0.741046, 0.754327,
          0.753425, 0.749539, 0.754177, 0.764063]

for a,b in zip(x_pick, y_pick):
    plt.text(a, b, (a, float("%.3f" % b)) )

def func(x, a, b, c):
   return a * np.exp(-b * x) + c

x = np.array(x)
y = np.array(y)
popt, pcov = curve_fit(func, x, y)
plt.plot(x, func(x, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.grid()

plt.xlabel('Image quality')
plt.ylabel('Classification accuracy')
plt.legend()
plt.title('Cat vs Dog')
plt.savefig("graph.png")

