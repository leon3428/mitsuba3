import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant('scalar_rgb')

scene = mi.load_file('/home/leon3428/dev/deep-structure/scenes/test.xml')

sensor = scene.sensors()[0]
sampler = sensor.sampler()
spp = sampler.sample_count()

time = sensor.shutter_open()
if sensor.shutter_open_time() > 0.0:
    time += sampler.next_1d() * sensor.shutter_open_time()

rays = sensor.sample_ray_differential(time, )

print(sensor)
print(sampler)
print(spp)