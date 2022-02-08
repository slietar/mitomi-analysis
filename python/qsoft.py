import sys
import numpy as np


ref_shape = (16, 64)

def parse(path):
  data = path.open().read()

  start_index = _find_nth(data, ":", start=_find_nth(data, "\n", 46)) + 1
  end_index = _find_nth(data, "\n", 173 - 46, start=start_index)

  values = np.rot90([[int(word.strip()) for word in line.split(",")] for line in data[start_index:end_index].split("\n\n")], -1)

  result = np.zeros((16, 64), dtype=np.uint32)
  result[:values.shape[0], :values.shape[1]] = values

  if values.shape != ref_shape:
    print(f"Warning: input shape {values.shape} does not match the reference shape {ref_shape}", file=sys.stderr)

  return result.flatten().tolist()


def _find_nth(str, char, nth = 1, *, start = 0):
  index = start

  while nth > 0:
    index = str.find(char, index) + 1
    nth = nth - 1

    if index < 0:
      return -1

  return index - 1


if __name__ == "__main__":
  from pathlib import Path
  import sys

  print(parse(Path(sys.argv[1]).resolve()))
