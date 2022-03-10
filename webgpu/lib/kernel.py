import numpy as np
import struct
import wgpu


def kernel(runner, input_buffer, output_buffer, input_size, kernel):
  shader = runner._create_shader("kernel")

  bind_group_layout = runner.device.create_bind_group_layout(entries=[
    { "binding": 0,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {
        "type": wgpu.BufferBindingType.read_only_storage,
      } },
    { "binding": 1,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {
        "type": wgpu.BufferBindingType.storage,
      } },
    { "binding": 2,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {
        "type": wgpu.BufferBindingType.read_only_storage,
      } },
    { "binding": 3,
      "visibility": wgpu.ShaderStage.COMPUTE,
      "buffer": {
        "type": wgpu.BufferBindingType.read_only_storage,
      } }
  ])

  settings_buffer = runner.device.create_buffer_with_data(
    data=struct.pack('IIII', input_size[0], input_size[1], round((kernel.shape[0] + 1) * 0.5), round((kernel.shape[1] + 1) * 0.5)),
    usage=wgpu.BufferUsage.STORAGE
  )

  kernel_buffer = runner.device.create_buffer_with_data(
    data=kernel.astype(np.float32),
    usage=wgpu.BufferUsage.STORAGE
  )

  bindings = [
    { "binding": 0,
      "resource": {"buffer": input_buffer, "offset": 0, "size": input_buffer.size } },
    { "binding": 1,
      "resource": {"buffer": output_buffer, "offset": 0, "size": output_buffer.size } },
    { "binding": 2,
      "resource": {"buffer": settings_buffer, "offset": 0, "size": settings_buffer.size } },
    { "binding": 3,
      "resource": {"buffer": kernel_buffer, "offset": 0, "size": kernel_buffer.size } }
  ]

  pipeline_layout = runner.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
  bind_group = runner.device.create_bind_group(layout=bind_group_layout, entries=bindings)


  compute_pipeline = runner.device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shader, "entry_point": "main"}
  )

  def run(command_encoder):
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch(input_size[0], input_size[1], 1)
    compute_pass.end_pass()

  return run
