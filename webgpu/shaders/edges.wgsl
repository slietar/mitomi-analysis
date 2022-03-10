let PI = 3.14159;


struct DataContainer {
  data: [[stride(4)]] array<f32>;
};

struct Settings {
  width: u32;
  height: u32;

  lowThreshold: f32;
  highThreshold: f32;
};

[[group(0), binding(0)]]
var<storage, read> input: DataContainer;

[[group(0), binding(1)]]
var<storage, read_write> output: DataContainer;

[[group(0), binding(2)]]
var<storage, read> settings: Settings;

[[group(0), binding(3)]]
var<storage, read_write> data2: DataContainer;


fn coord_to_index(coord: vec2<u32>) -> u32 {
  return coord.x + settings.width * coord.y;
}

//, source: ptr<storage, array<f32>> */
fn blur(coord: vec2<u32>) -> f32 {
  return 0.25 * input.data[coord_to_index(coord)]
    + 0.125 *   input.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y))]
    + 0.125 *   input.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y))]
    + 0.125 *   input.data[coord_to_index(vec2<u32>(coord.x,      coord.y + 1u))]
    + 0.125 *   input.data[coord_to_index(vec2<u32>(coord.x,      coord.y - 1u))]
    + 0.0625 *  input.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y + 1u))]
    + 0.0625 *  input.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y - 1u))]
    + 0.0625 *  input.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y + 1u))]
    + 0.0625 *  input.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y - 1u))];
}

fn grad(coord: vec2<u32>) -> vec2<f32> {
  let gx = data2.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y - 1u))]
    + 2.0 * data2.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y))]
    + data2.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y + 1u))]
    - data2.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y - 1u))]
    - 2.0 * data2.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y))]
    - data2.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y + 1u))];

  let gy = data2.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y - 1u))]
    + 2.0 * data2.data[coord_to_index(vec2<u32>(coord.x, coord.y - 1u))]
    + data2.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y - 1u))]
    - data2.data[coord_to_index(vec2<u32>(coord.x - 1u, coord.y + 1u))]
    - 2.0 * data2.data[coord_to_index(vec2<u32>(coord.x, coord.y + 1u))]
    - data2.data[coord_to_index(vec2<u32>(coord.x + 1u, coord.y + 1u))];

  return vec2<f32>(gx, gy);
}

fn rotate2d(v: vec2<f32>, rad: f32) -> vec2<f32> {
  let s = sin(rad);
  let c = cos(rad);

  return vec2<f32>(
    v.x * c + s * v.y,
    -v.x * s + c * v.y
  );
}

fn round_angle(v: vec2<f32>) -> vec2<f32> {
  let len = length(v);
  let n = normalize(v);
  var maximum = -1.0;
  var bestAngle = 0.0;

  var i = 0u;
  loop {
    if (i >= 8u) {
      break;
    }

    let theta = (f32(i) * 2. * PI) / 8.;
    let u = rotate2d(vec2<f32>(1., 0.), theta);
    let scalarProduct = dot(u, n);

    if (scalarProduct > maximum) {
      bestAngle = theta;
      maximum = scalarProduct;
    }

    i = i + 1u;
  }

  return len * rotate2d(vec2<f32>(1., 0.), bestAngle);
}


fn magnitude(coord: vec2<u32>) -> vec2<f32> {
  let gradient = round_angle(grad(coord));
  let step = ceil(normalize(gradient));
  let len = length(gradient);

  let pos = grad(vec2<u32>(vec2<f32>(coord) + step));
  let neg = grad(vec2<u32>(vec2<f32>(coord) - step));

  if ((length(pos) >= len) || (length(neg) >= len)) {
    return vec2<f32>(0.0);
  }

  return gradient;
}

fn threshold(coord: vec2<u32>) -> f32 {
  let gradient = magnitude(coord);
  let len = length(gradient);

  // return 0.5 * (step(settings.lowThreshold, len) + step(settings.highThreshold, len));
  return step(settings.lowThreshold, len);
}



[[stage(compute), workgroup_size(1)]]
fn main1([[builtin(global_invocation_id)]] index: vec3<u32>) {
    // let i: u32 = index.x + 512u * index.y;
    // - input.data[coord_to_index(index.xy)];
    data2.data[coord_to_index(index.xy)] = blur(index.xy);
}


[[stage(compute), workgroup_size(1)]]
fn main2([[builtin(global_invocation_id)]] invocation: vec3<u32>) {
  // data2.data[coord_to_index(index.xy)] = grad(index.xy).x; // data2.data[coord_to_index(index.xy)];
  // output.data[coord_to_index(index.xy)] = data2.data[coord_to_index(index.xy)];

  output.data[coord_to_index(invocation.xy)] = threshold(invocation.xy);
}
