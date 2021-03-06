import argparse
import json
import math
from pathlib import Path
import pandas as pd
import sys
from time import time
from tqdm import tqdm

import qsoft


parser = argparse.ArgumentParser(description="MITOMI data analysis")
parser.add_argument("--head", metavar="k", type=int, default=math.inf, help="Only process the first k images")
parser.add_argument("--layout", metavar="path", help="Path to a QSoft log file")
parser.add_argument("--only", type=int)
parser.add_argument("--out", help="Path to an output file (defaults to stdout)")
parser.add_argument("--silent", action="store_true")
parser.add_argument("--test", choices=["button_circle", "button_hough", "edges", "masks", "signal_circle"])
parser.add_argument("file1", metavar="background", help="Path to the background data")
parser.add_argument("file2", metavar="signal", help="Path to the signal data")

analysis_options = parser.add_argument_group("Analysis options")
analysis_options.add_argument("--settings", metavar="<path>", type=str, help="Path to a settings file")
analysis_options.add_argument("--save-settings", metavar="<path>", const=sys.stdout, nargs="?", type=argparse.FileType("w"), help="Path to a new settings file")

analysis_options.add_argument("--background-inner-radius", metavar="<radius>", type=int, help="(default: 100)")
analysis_options.add_argument("--background-outer-radius", metavar="<radius>", type=int, help="(default: 150)")
analysis_options.add_argument("--button-radius", metavar="<radius>", nargs="*", type=int, help="(default: [100]")
analysis_options.add_argument("--edges-sigma", metavar="<sigma>", type=float, help="(default: 1.0)")
analysis_options.add_argument("--signal-radius", metavar="<radius>", type=int, help="(default: 30)")

args = parser.parse_args()

path1 = Path(args.file1).resolve()
path2 = Path(args.file2).resolve()
path_layout = Path(args.layout).resolve() if args.layout else None
path_out = Path(args.out).resolve() if args.out else None
path_settings = Path(args.settings).resolve() if args.settings else None

path_background, path_signal = (path2, path1)\
  # if any(x in path2.name.lower() for x in ["bf", "background"]) or any(x in path1.name.lower() for x in ["cy3", "fitc", "signal"])\
  # else (path1, path2)

path_background, path_signal = (path1, path2)


layout_data = qsoft.parse(path_layout) if path_layout else None

settings_loaded = json.load(path_settings.open()) if path_settings else dict()
settings = {
  "background_inner_radius": args.background_inner_radius or settings_loaded.get("background_inner_radius", 100),
  "background_outer_radius": args.background_outer_radius or settings_loaded.get("background_outer_radius", 150),
  "button_radius": args.button_radius or settings_loaded.get("button_radius", [100]),
  "edges_sigma": args.edges_sigma or settings_loaded.get("edges", dict()).get("sigma", 1.0),
  "signal_radius": args.signal_radius or settings_loaded.get("signal_radius", 30)
}

if args.save_settings:
  json.dump(settings, args.save_settings, indent=2)
  sys.exit()


# Packages are loaded later on as they are slow to load.

import matplotlib.pyplot as plt
from nd2reader import ND2Reader
import numpy as np
import scipy.ndimage
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, disk


data_background = ND2Reader(str(path_background))
data_signal = ND2Reader(str(path_signal))

count = min(len(data_background), len(data_signal), args.head) if (args.only is None) else (args.only + 1)
start = args.only or 0

signal_radius_test = range(10, 100, 2)


# Utility functions

def detect_circle(edges, test_radius):
  hough = hough_circle(edges, radius=test_radius)
  peaks = hough_circle_peaks(hough, test_radius, total_num_peaks=1)
  value, center_x, center_y, radius = np.array(peaks)[:, 0]

  return (int(center_y), int(center_x)), int(radius), hough[0,], value

def normalize(image):
  return image / np.max(image)

def show_circle(image, center, radius):
  image = color.gray2rgb(image)

  circy, circx = circle_perimeter(*center, radius, shape=image.shape)
  image[circy, circx] = (1, 0, 0)

  show_image(image)

def show_image(image):
  plt.imshow(image, cmap=plt.cm.gray)
  plt.show()
  sys.exit()

def find_drop(sx, sy, fr = 1 - 1 / math.e):
  y_max = sy[0]

  for i, (x, y) in enumerate(zip(sx, sy)):
    y_target = y_max * fr

    if y < y_target:
      px = sx[i - 1]
      py = sy[i - 1]
      t = (y_target - py) / (y - py)
      return px + t * (x - px)
    else:
      y_max = max(y, y_max)

def disk_mask(center, radius, shape):
  out = np.zeros(shape, dtype=np.uint16)
  if radius > 0: out[disk(center, radius, shape=shape)] = 1
  return out


# Layout settings

class Progress:
  def __init__(self):
    self.cols = 64
    self.rows = 16

    self.init()

  def init(self):
    for _ in range(self.rows):
      for _ in range(self.cols):
        print("\u2591", end="", file=sys.stderr)

      print(file=sys.stderr)

  # signal: 0, 1 or 2
  def set(self, index, signal = 2):
    x = index % self.cols
    y = index // self.cols

    print("\033[s" + f"\033[{x + 1}G" + f"\033[{self.rows - y}A" + chr(0x2591 + signal) + "\033[u", end="", file=sys.stderr, flush=True)



output = pd.DataFrame(columns=["button_x", "button_y", "button_radius", "button_hough", "background", "raw_signal", "signal", "signal_x", "signal_y", "signal_radius", "number", "column", "row"] + (["sample"] if layout_data else list()))
(progress, tq) = (Progress(), tqdm(total=count)) if not (args.test or args.silent) else (None, None)



prev_center = None
start_time = time()

for index, (image_background, image_signal) in enumerate(zip(data_background[start:count], data_signal[start:count])):
  index += start

  edges = canny(image_background, sigma=settings["edges_sigma"])
  button_center, button_radius, button_hough, button_hough_value = detect_circle(edges, settings["button_radius"])


  # Center check compared to neighbor

  if prev_center:
    dist = np.linalg.norm(np.array(prev_center) - button_center)
    # TODO: do something with 'dist'

  prev_center = button_center


  # Hough transform tests

  if args.test == "edges":
    show_image(edges)

  if args.test == "button_hough":
    show_image(normalize(button_hough))

  if args.test == "button_circle":
    show_circle(normalize(image_background), button_center, button_radius)


  # Background value

  background_mask = disk_mask(button_center, settings["background_outer_radius"], image_background.shape)\
    - disk_mask(button_center, settings["background_inner_radius"], image_background.shape)
  background_data = image_signal * background_mask
  background_value = np.median(background_data[background_data.nonzero()])


  # Signal value

  signal_mask = disk_mask(button_center, settings["signal_radius"], image_signal.shape)
  signal_data = image_signal * signal_mask
  signal_value = np.median(signal_data[signal_data.nonzero()])

  signal_value_threshold = np.percentile(signal_data[signal_data.nonzero()], 20)
  signal_mass_mask = image_signal * disk_mask(button_center, 100, image_signal.shape)
  signal_mass = np.where(signal_mass_mask > signal_value_threshold, 1, 0)

  signal_center = scipy.ndimage.center_of_mass(signal_mass)


  # Signal radius determination

  signal_radius = -1
  radius_semiinc = 1
  radius_max = 100

  value_max = -math.inf
  value_prev = None

  for radius in range(radius_semiinc, radius_max, radius_semiinc * 2):
    radius_inner = radius - radius_semiinc
    radius_outer = radius + radius_semiinc

    mask = disk_mask(signal_center, radius_outer, image_background.shape)\
      - disk_mask(signal_center, radius_inner, image_background.shape)
    data = image_signal * mask
    value = np.median(data[data.nonzero()]) - background_value
    value_threshold = value_max * (1 - 1 / math.e)

    if (value_max >= 0) and (value < value_threshold):
      t = (value_threshold - value_prev) / (value - value_prev)
      prev_radius = radius - radius_semiinc * 2
      signal_radius = prev_radius + t * (radius - prev_radius)
      # signal_radius = radius

      break

    value_max = max(value_max, value)
    value_prev = value


  # Final tests

  if args.test == "masks":
    mask_fr = 0.2

    image = color.gray2rgb(normalize(image_signal / 0xffff)) * (1 - mask_fr)
    image[:, :, 0] += signal_mask * mask_fr
    image[:, :, 1] += disk_mask(signal_center, signal_radius, image_signal.shape) * mask_fr
    image[:, :, 2] += background_mask * mask_fr
    show_image(image.clip(0, 1))

  if args.test == "signal_circle":
    show_circle(normalize(image_signal), button_center, round(signal_radius))


  # Layout information

  row = math.floor(index / 64)
  col = (index % 64)

  if (row % 2) > 0:
    col = 63 - col

  chip_index = row * 64 + col


  # Write to output

  output.loc[index] = [
    button_center[1],
    button_center[0],
    button_radius,
    button_hough_value,
    background_value,
    signal_value,
    signal_value - background_value,
    signal_center[1],
    signal_center[0],
    signal_radius,
    chip_index + 1,
    col + 1,
    row + 1
  ] + ([layout_data[chip_index]] if layout_data else list())


  if tq:
    tq.update()

  if progress:
    progress.set(chip_index)

if tq:
  tq.close()


csv = output.to_csv(index_label="index")

print(csv, file=path_out.open("w") if path_out else sys.stdout)

if not args.silent:
  print(f"Done in {((time() - start_time) / count * 1000):.2f} ms per image", file=sys.stderr)
