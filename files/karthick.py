import numpy as np
from numpy import array

pixel_data = [[i for i in range(1)] for j in range(500)]
pixel_data = np.array(pixel_data)
x = 0
for i in range(500):
    pixel_data[x,0] = i
    x += 1

print(np.shape(pixel_data))
print (pixel_data)
