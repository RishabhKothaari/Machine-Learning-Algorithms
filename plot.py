import matplotlib.pyplot as plt
import numpy as np

x1, y1 = np.loadtxt('exp-3-training-data-size-15000-.csv', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('exp-3-testing-data-size-15000-.csv', delimiter=',', unpack=True)
fig, ax = plt.subplots()
plt.plot(x1,y1, label='Training Set')
plt.plot(x2,y2, label='Testing Set')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
# plt.title('Accuracy as a function of number of epochs with 50 hidden units.')
plt.legend(loc='lower right')
plt.show()