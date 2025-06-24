import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import h5py

with h5py.File('ltm.h5', 'r') as f:
    rows = np.array(f['rows'])
    cols = np.array(f['cols'])
    values = np.array(f['values'])

values = values.astype(np.float32) / ((1 << 16) - 1)  # Normalize values to [0, 1]
sensor_size = (1024, 1024)
projector_size = (1024, 1024)
ltm = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(
    sensor_size[0]*sensor_size[1], projector_size[0]*projector_size[1]))


# # p = (np.sin(np.linspace(0, 100, projector_size[0])) + 1.0) / 2.0
# # p = np.tile(p, (projector_size[1]))
p = np.ones((projector_size[0] * projector_size[1]))
image = np.reshape(ltm @ p, (sensor_size[0], sensor_size[1]))

hdr_mapped = image ** (1.0 / 2.2)
print(np.max(hdr_mapped), np.min(hdr_mapped))

plt.axis("off")
plt.imshow(hdr_mapped, cmap='gray', vmin=0.0, vmax=1.0)
plt.savefig('plot.png', dpi=300, bbox_inches='tight', pad_inches=0)