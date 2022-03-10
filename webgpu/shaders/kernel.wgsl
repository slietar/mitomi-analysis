struct DataContainer {
  data: [[stride(4)]] array<f32>;
};

struct Settings {
  width: u32;
  height: u32;

  kernelHalfWidth: u32;
  kernelHalfHeight: u32;
};

[[group(0), binding(0)]]
var<storage, read> input: DataContainer;

[[group(0), binding(1)]]
var<storage, read_write> output: DataContainer;

[[group(0), binding(2)]]
var<storage, read> settings: Settings;

[[group(0), binding(3)]]
var<storage, read> kernel: DataContainer;


fn coord_to_index(coord: vec2<u32>) -> u32 {
  return coord.x + settings.width * coord.y;
}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] index: vec3<u32>) {
    // output.data[coord_to_index(index.xy)] = input.data[coord_to_index(index.xy)];

    let kernelWidth = settings.kernelHalfWidth * 2u - 1u;
    let kernelHeight = settings.kernelHalfHeight * 2u - 1u;

    var value = 0.0;

    // for (var x = 0u; x < kernelWidth; x = x + 1u) {
    //   value = value + input.data[coord_to_index(index.xy - vec2<u32>(settings.kernelHalfWidth + 1u, 0u) + vec2<u32>(x, 0u))] * kernel.data[(settings.kernelHalfHeight - 1u) * kernelWidth + x];
    // }

    for (var y = 0u; y < kernelHeight; y = y + 1u) {
      for (var x = 0u; x < kernelWidth; x = x + 1u) {
        value = value + input.data[coord_to_index(index.xy
          - vec2<u32>(settings.kernelHalfWidth + 1u, settings.kernelHalfHeight + 1u)
          + vec2<u32>(x, y))]
          * kernel.data[y * kernelWidth + x];
      }
    }

    output.data[coord_to_index(index.xy)] = value;
}
