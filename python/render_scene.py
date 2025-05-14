import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np

mi.set_variant('cuda_mono')
scene = mi.load_file("./scenes/test.xml")
img = mi.render(scene)

hdr_mapped = img ** (1.0 / 2.2)
print(np.max(hdr_mapped), np.min(hdr_mapped))

plt.axis("off")
plt.imshow(hdr_mapped, cmap="gray", vmin=0.0, vmax=1.0)
plt.show()
