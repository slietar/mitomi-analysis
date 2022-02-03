import argparse
import json
import math
from pathlib import Path
import pandas as pd
import sys
from time import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="MITOMI data analysis")
parser.add_argument("--head", type=int, default=math.inf, help="Only process the first k images")
parser.add_argument("--only", type=int)
parser.add_argument("--out", help="Path to an output file (defaults to stdout)")
parser.add_argument("--silent", action="store_true")
parser.add_argument("--test", choices=["background_mask", "button_circle", "button_hough", "edges", "masks", "signal_mask"])
parser.add_argument("file1", metavar="background", help="Path to the background data")
parser.add_argument("file2", metavar="signal", help="Path to the signal data")
# + layout

analysis_options = parser.add_argument_group("Analysis options")
analysis_options.add_argument("--settings", metavar="<path>", type=str, help="Path to a settings file")
analysis_options.add_argument("--save-settings", metavar="<path>", const=sys.stdout, nargs="?", type=argparse.FileType("w"), help="Path to a new settings file")

analysis_options.add_argument("--background-inner-radius", metavar="<radius>", type=int, help="(default: 100")
analysis_options.add_argument("--background-outer-radius", metavar="<radius>", type=int, help="(default: 150")
analysis_options.add_argument("--button-radius", metavar="<radius>", nargs="*", type=int, help="(default: [100]")
analysis_options.add_argument("--signal-radius", metavar="<radius>", type=int, help="(default: 50)")
analysis_options.add_argument("--edges-sigma", metavar="<sigma>", type=float, help="(default: 1)")

args = parser.parse_args()

path1 = Path(args.file1).resolve()
path2 = Path(args.file2).resolve()
path_out = Path(args.out).resolve() if args.out else None
path_settings = Path(args.settings).resolve() if args.settings else None

path_background, path_signal = (path2, path1)\
  if any(x in path2.name.lower() for x in ["bf", "background"]) or any(x in path1.name.lower() for x in ["cy3", "fitc", "signal"])\
  else (path1, path2)


options_loaded = json.load(path_settings.open()) if path_settings else dict()
options = {
  "background_inner_radius": args.background_inner_radius or options_loaded.get("background_inner_radius", 100),
  "background_outer_radius": args.background_outer_radius or options_loaded.get("background_outer_radius", 150),
  "button_radius": args.button_radius or options_loaded.get("button_radius", [100]),
  "signal_radius": args.signal_radius or options_loaded.get("signal_radius", 50),
  "edges": {
    "sigma": args.edges_sigma or options_loaded.get("edges", dict()).get("sigma", 1.0)
  }
}

if args.save_settings:
  json.dump(options, args.save_settings, indent=2)

  sys.exit()


# Packages are loaded later on as they are slow to load.

import matplotlib.pyplot as plt
from nd2reader import ND2Reader
import numpy as np
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, disk
from skimage.util import img_as_ubyte


data_background = ND2Reader(str(path_background))
data_signal = ND2Reader(str(path_signal))

count = min(len(data_background), len(data_signal), args.head) if (args.only is None) else (args.only + 1)
start = args.only or 0

start_time = time()


def detect_circle(edges, test_radius):
  hough = hough_circle(edges, radius=test_radius)
  peaks = hough_circle_peaks(hough, test_radius, total_num_peaks=1)
  accums, center_x, center_y, radius = np.array(peaks)[:, 0]

  return (int(center_y), int(center_x)), int(radius), hough[0,]

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


assign_columns = ["number", "column", "row"]

def assign(index):
  row = math.floor(index / 64)
  col = (index % 64)

  if (row % 2) > 0:
    col = 63 - col

  num = row * 64 + col + 1
  col = col + 1
  row = row + 1

  return num, col, row


output = pd.DataFrame(columns=["background", "signal", "signal_normalized", *assign_columns])
tq = tqdm(total=count) if not (args.test or args.silent) else None


prev_center = None

for index, (image_background, image_signal) in enumerate(zip(data_background[start:count], data_signal[start:count])):
  edges = canny(image_background, sigma=options["edges"]["sigma"])

  if args.test == "edges":
    show_image(edges)

  button_center, button_radius, button_hough = detect_circle(edges, options["button_radius"])

  # if prev_center:
  #   dist = np.linalg.norm(np.array(prev_center) - button_center)

  prev_center = button_center

  if args.test == "button_hough":
    show_image(normalize(button_hough))

  if args.test == "button_circle":
    show_circle(normalize(image_background), button_center, button_radius)

  # if args.test == "chamber_hough":
  #   show_image(normalize(chamber_hough))

  # if args.test == "chamber_circle":
  #   show_circle(normalize(image_background), chamber_center, chamber_radius)


  def disk_mask(center, radius, shape):
    out = np.zeros(shape)
    out[disk(center, radius, shape=shape)] = 1.0
    return out

  signal_mask = disk_mask(button_center, options["signal_radius"], image_background.shape)
  background_mask = disk_mask(button_center, options["background_outer_radius"], image_background.shape)\
    - disk_mask(button_center, options["background_inner_radius"], image_background.shape)

  background_data = image_signal * background_mask
  signal_data = image_signal * signal_mask

  if args.test == "masks":
    image = color.gray2rgb(normalize(image_signal))
    image[:, :, 2] += background_mask * 0.1
    image[:, :, 0] += signal_mask * 0.1
    show_image(image.clip(0, 1))

  if args.test == "background_mask":
    show_image(background_data)

  if args.test == "signal_mask":
    show_image(signal_data)

  background_value = np.median(background_data[background_data.nonzero()])
  signal_value = np.median(signal_data[signal_data.nonzero()])

  output.loc[index] = [background_value, signal_value, signal_value - background_value, *assign(index)]

  if tq:
    tq.update()

if tq:
  tq.close()


csv = output.to_csv(index_label="index")

print(csv, file=path_out.open("w") if path_out else sys.stdout)

if not args.silent:
  print(f"Done in {((time() - start_time) / count * 1000):.2f} ms per image", file=sys.stderr)
