import numpy as np
import matplotlib.pyplot as plt

# plt.xticks([0,1,2])
# plt.yticks()

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75, 100]
y = [0.714343, 0.56, 0.67, 0.78, 0.8, 0.83, 0.85, 0.88, 0.89, 0.90, 0.93, 0.95, 0.97, 0.96, 0.805489]

print (len(x), len(y))

plt.plot(x, y)
plt.grid()
plt.xlabel('Image quality')
plt.ylabel('Classification accuracy')
plt.title('Cat vs Dog')
# plt.show()
plt.savefig("graph.pdf")
