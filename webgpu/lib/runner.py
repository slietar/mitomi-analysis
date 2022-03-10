import numpy as np
from pathlib import Path
import wgpu
import wgpu.backends.rs
import wgpu.utils

from blur import blur
from circle import circle
from edges import edges


class Runner:
  def __init__(self):
    self.device = wgpu.utils.get_default_device()

  def _create_binding(self, entries):
    bind_group_layout = self.device.create_bind_group_layout(entries=[
      { 'binding': index,
        'visibility': wgpu.ShaderStage.COMPUTE,
        'buffer': {
          'type': wgpu.BufferBindingType.storage if entry['write'] else wgpu.BufferBindingType.read_only_storage
        } } for index, entry in enumerate(entries)
    ])

    bindings = [
      { 'binding': index,
        'resource': {
          'buffer': entry['buffer'],
          'offset': 0,
          'size': entry['buffer'].size
        } } for index, entry in enumerate(entries)
    ]

    pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    bind_group = self.device.create_bind_group(layout=bind_group_layout, entries=bindings)

    return pipeline_layout, bind_group

  def _create_pass(self, command_encoder, bind_group, compute_pipeline, dispatch):
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch(*dispatch)
    compute_pass.end_pass()

  def _create_shader(self, name):
    return self.device.create_shader_module(code=(Path(__file__).parent / "../shaders" / (name + ".wgsl")).open().read())


  def blur(self, input_buffer, output_buffer, *args, **kwargs):
    return blur(self, input_buffer, output_buffer, *args, **kwargs)

  def circle(self, input_buffer, output_buffer, *args, **kwargs):
    return circle(self, input_buffer, output_buffer, *args, **kwargs)

  def edges(self, input_buffer, output_buffer, *args, **kwargs):
    return edges(self, input_buffer, output_buffer, *args, **kwargs)
