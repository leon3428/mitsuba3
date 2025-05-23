import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse

ltm = scipy.sparse.load_npz("ltm.npz")
sensor_size = (128, 128)
projector_size = (128, 128)

# p = (np.sin(np.linspace(0, 100, projector_size[0])) + 1.0) / 2.0
# p = np.tile(p, (projector_size[1]))
p = np.ones((projector_size[0] * projector_size[1]))
image = np.reshape(ltm @ p, (sensor_size[0], sensor_size[1]))

hdr_mapped = image ** (1.0 / 2.2)
print(np.max(hdr_mapped), np.min(hdr_mapped))

plt.axis("off")
plt.imshow(hdr_mapped, cmap='gray', vmin=0.0, vmax=1.0)
plt.show()