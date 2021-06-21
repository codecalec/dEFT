import re as _re
from pathlib import Path as _Path
import numpy as _np


def convert_hwu_to_numpy(
    filename: _Path, num_of_hist: int = 1
) -> (_np.ndarray, _np.ndarray, _np.ndarray):
    """
    Convert a HwU (histogram with uncertainties) file from MadGraph into a JSON file which is compatible with dEFT.
    Only parses the first histogram in the file by default.
    """

    # Known Bug: Only considers the last histogram as the final bin and will only consider the right edge on the final bin

    with open(filename, "r") as hwu_f:
        bin_lefts = _np.empty(0)
        bin_rights = _np.empty(0)
        central_values = _np.empty(0)
        for i in range(num_of_hist):
            line = next(hwu_f)
            while (match := _re.search(r"<histogram> (\d+) ", line)) is None:
                line = next(hwu_f)
                if line is None:
                    raise RuntimeError(f"No histogram found in HwU file: {filename}")
            bins = int(match.group(1))

            line = next(hwu_f)
            for i in range(bins):
                # The 2-index in HwU are central value of bin
                values = line.strip().split()
                central_values = _np.append(central_values, float(values[2]))
                bin_lefts = _np.append(bin_lefts, float(values[0]))
                bin_rights = _np.append(bin_rights, float(values[1]))
                line = next(hwu_f)

    assert len(bin_lefts) == len(central_values)
    assert len(bin_rights) == len(central_values)
    return (bin_lefts, bin_rights, central_values)
