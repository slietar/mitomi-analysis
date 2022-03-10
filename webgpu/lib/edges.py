import struct
import wgpu


def edges(runner, input_buffer, output_buffer, input_size, *,
  threshold_low = 0.2,
  threshold_high = 0.3):
  shader = runner._create_shader("edges")

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
        "type": wgpu.BufferBindingType.storage,
      } }
  ])

  settings_buffer = runner.device.create_buffer_with_data(
    data=struct.pack('IIff', input_size[0], input_size[1], threshold_low, threshold_high),
    usage=wgpu.BufferUsage.STORAGE
  )

  hidden_buffer1 = runner.device.create_buffer(
    size=input_buffer.size,
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
      "resource": {"buffer": hidden_buffer1, "offset": 0, "size": hidden_buffer1.size } }
  ]

  pipeline_layout = runner.device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
  bind_group = runner.device.create_bind_group(layout=bind_group_layout, entries=bindings)


  compute_pipeline1 = runner.device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shader, "entry_point": "main1"}
  )

  compute_pipeline2 = runner.device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": shader, "entry_point": "main2"}
  )

  def run(command_encoder):
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline1)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch(input_size[0], input_size[1], 1)
    compute_pass.end_pass()

    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline2)
    compute_pass.set_bind_group(0, bind_group, [], 0, 999999)
    compute_pass.dispatch(input_size[0], input_size[1], 1)
    compute_pass.end_pass()

  return run
