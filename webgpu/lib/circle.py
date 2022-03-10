import math
from venv import create
import numpy as np
import struct
import wgpu


def create_circle(radius):
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

    x = math.sqrt(x ** 2 - 2 * y - 1)
    y += 1

  return points


def circle(runner, input_buffer, output_buffer, input_size, radii):
  offsets = [0]
  points = list()

  for radius in radii:
    points += create_circle(radius)
    offsets.append(len(points) + len(radii) + 1)

  settings_buffer = runner.device.create_buffer_with_data(
    data=struct.pack('III', input_size[0], input_size[1], len(radii)),
    usage=wgpu.BufferUsage.STORAGE
  )

  points_buffer = runner.device.create_buffer_with_data(
    data=np.array([*offsets, *points], dtype=np.int32),
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
  )

  pipeline_layout, bind_group = runner._create_binding([
    { 'buffer': input_buffer, 'write': False },
    { 'buffer': output_buffer, 'write': True },
    { 'buffer': settings_buffer, 'write': False },
    { 'buffer': points_buffer, 'write': False }
  ])

  shader = runner._create_shader("circle")
  compute_pipeline = runner.device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shader, "entry_point": "main"}
  )

  def run(command_encoder):
    runner._create_pass(
      command_encoder,
      bind_group=bind_group,
      compute_pipeline=compute_pipeline,
      dispatch=(input_size[0], input_size[1], len(radii))
    )

  return run
