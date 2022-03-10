struct DataContainer {
  data: [[stride(4)]] array<f32>;
};

struct DataContainerInt {
  data: [[stride(4)]] array<u32>;
};

struct Settings {
  width: u32;
  height: u32;

  depth: u32;
};

[[group(0), binding(0)]]
var<storage, read> input: DataContainer;

[[group(0), binding(1)]]
var<storage, read_write> output: DataContainer;

[[group(0), binding(2)]]
var<storage, read> settings: Settings;

[[group(0), binding(3)]]
var<storage, read> points: DataContainerInt;


fn coord_to_index(coord: vec2<u32>) -> u32 {
  return coord.x + settings.width * coord.y;
}


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] invocation: vec3<u32>) {
  let radius_index = invocation.z;
  let center = vec2<i32>(invocation.xy);

  var value = 0.0;
  var total = 0.0;

  for (var point_index = points.data[radius_index]; point_index < points.data[radius_index + 1u]; point_index = point_index + 2u) {
    let coords = center + vec2<i32>(
      i32(points.data[point_index]),
      i32(points.data[point_index + 1u])
    );

    if (coords.x >= 0 && coords.y >= 0 && coords.x < i32(settings.width) && coords.y < i32(settings.height)) {
      value = value + input.data[coord_to_index(vec2<u32>(coords))];
      total = total + 1.0;
    }
  }

  // output.data[settings.width * settings.height * invocation.z + coord_to_index(invocation.xy)] = input.data[coord_to_index(invocation.xy)];

  output.data[settings.width * settings.height * radius_index + coord_to_index(invocation.xy)] = value; // / total
}
