import numpy as np
import matplotlib.pyplot as plt

camera_size = (64, 64)
projector_size = (64, 64)

loaded = np.load('python/tmp.npz')
ltm = loaded['ltm'].reshape((camera_size[0] * camera_size[1], projector_size[0] * projector_size[1]))

print(ltm.shape)

p = (np.sin(np.linspace(0, 100, projector_size[0])) + 1.0) / 2.0
p = np.tile(p, (projector_size[1]))
# p = np.ones((projector_size[0] * projector_size[1]))
print(p.shape)
# p = np.ones((projector_width * projector_height))
image = np.reshape(ltm @ p, (camera_size[0], camera_size[1]))
print(image.shape)

plt.axis("off")
# plt.imshow(image / np.max(image))
plt.imshow(image / np.max(image), cmap='gray', vmin=0.0, vmax=1.0)
plt.show()