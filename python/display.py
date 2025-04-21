import numpy as np
import matplotlib.pyplot as plt

loaded = np.load('python/tmp.npz')
sensor_size = loaded['sensor_size']
projector_size = loaded['projector_size']
ltm = loaded['ltm'].reshape((sensor_size[0] * sensor_size[1], projector_size[0] * projector_size[1]))

p = (np.sin(np.linspace(0, 100, projector_size[0])) + 1.0) / 2.0
p = np.tile(p, (projector_size[1]))
# p = np.ones((projector_size[0] * projector_size[1]))
image = np.reshape(ltm @ p, (sensor_size[0], sensor_size[1]))

plt.axis("off")
plt.imshow(image / np.max(image), cmap='gray', vmin=0.0, vmax=1.0)
plt.show()