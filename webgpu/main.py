import math
import numpy as np
from pathlib import Path
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import time


data = ND2Reader(str(Path("../data/20210811/bf_image.nd2").resolve()))

# print(image)
# print(array)

# plt.imshow(image, cmap=plt.cm.gray)
# plt.show()

# import sys
# sys.exit()


def circle(radius):
  points = list()

  x = radius
  y = 0

  while x >= y:
    fx = round(x)

    points += [
      y, fx,
      y, -fx,
      -y, fx,
      -y, -fx,

      fx, y,
      fx, -y,
      -fx, y,
      -fx, -y
    ]

    x = math.sqrt(x ** 2 - 2 * y - 1);
    y += 1

  return np.array(points)


import wgpu
import wgpu.backends.rs
import wgpu.utils


shader_source = open("./shader.wgsl").read()

device = wgpu.utils.get_default_device()
shader = device.create_shader_module(code=shader_source)

size = (512, 512)
nbytes = size[0] * size[0] * 4


buffer1 = device.create_buffer(
  size=nbytes,
  usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
)

buffer2 = device.create_buffer(
  size=nbytes,
  usage=wgpu.BufferUsage.STORAGE
)

buffer3 = device.create_buffer(
  size=nbytes,
  usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)


circle_points = circle(100)
circle_data=np.array([len(circle_points) * 0.5, *circle_points], np.int32)
print(len(circle_data))

buffer4 = device.create_buffer_with_data(
  data=circle_data,
  usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
)


bind_group_layout = device.create_bind_group_layout(entries=[
  { "binding": 0,
    "visibility": wgpu.ShaderStage.COMPUTE,
    "buffer": {
      # "type": wgpu.BufferBindingType.read_only_storage,
      "type": wgpu.BufferBindingType.storage,
    } },
  { "binding": 1,
    "visibility": wgpu.ShaderStage.COMPUTE,
    "buffer": {
      "type": wgpu.BufferBindingType.storage,
    } },
  { "binding": 2,
    "visibility": wgpu.ShaderStage.COMPUTE,
    "buffer": {
      "type": wgpu.BufferBindingType.storage,
    } },

  { "binding": 3,
    "visibility": wgpu.ShaderStage.COMPUTE,
    "buffer": {
      "type": wgpu.BufferBindingType.read_only_storage
    } }
])

bindings = [
  { "binding": 0,
    "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size } },
  { "binding": 1,
    "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size } },
  { "binding": 2,
    "resource": {"buffer": buffer3, "offset": 0, "size": buffer3.size } },
  { "binding": 3,
    "resource": {"buffer": buffer4, "offset": 0, "size": buffer4.size } }
]

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

def create_pipeline(entry_point):
  return device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shader, "entry_point": entry_point}
  )

compute_pipeline = create_pipeline("main")
compute_pipeline2 = create_pipeline("main2")
compute_pipeline3 = create_pipeline("main3")


# ---


def create_pass(command_encoder, compute_pipeline):
  compute_pass = command_encoder.begin_compute_pass()
  compute_pass.set_pipeline(compute_pipeline)
  compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
  compute_pass.dispatch(size[0], size[1], 1)
  compute_pass.end_pass()

def argmax(arr):
  return np.unravel_index(np.argmax(arr), arr.shape)

def process_image(index):
  image = data[index].astype(np.float32) / 0xffff
  array = np.frombuffer(image, np.float32)

  device.queue.write_buffer(buffer1, 0, array)

  command_encoder = device.create_command_encoder()

  create_pass(command_encoder, compute_pipeline)
  create_pass(command_encoder, compute_pipeline2)
  create_pass(command_encoder, compute_pipeline3)

  device.queue.submit([
    command_encoder.finish()
  ])

  output_view = device.queue.read_buffer(buffer3).cast("f")
  output_array = np.asarray(output_view).reshape(size)

  return argmax(output_array)
  # return output_array



# import sys
# sys.exit()

from tqdm import tqdm

print("Starting")
a = time.time_ns()
n = 1024

for i in tqdm(range(n)):
  x = process_image(i)
  # print(x)

print((time.time_ns() - a) / 1e6 / n)

# plt.imshow(x, cmap=plt.cm.gray, norm=None, vmin=0, vmax=1)
# plt.imshow(x, cmap=plt.cm.gray)
# plt.show()
