#!/usr/bin/env python
"""Script to generate plots for in FOV solar contamination"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pygac import get_reader_class
from pygac.calibration.vis_uncertainty import vis_uncertainty
import xarray as xr
import datetime

def sun_glint(ds,mask,plot=False):
    uncertainty = vis_uncertainty(ds,mask,plot=False)
    contam_pixels = uncertainty["solar_fov_contam"].values
    pixel = ds["pixel_index"]

    fov_pixels = np.count_nonzero(contam_pixels == 1)
    print("FOV Solar Contam flag was set", fov_pixels, "times")

    if plot:
        plt.plot(pixel, contam_pixels)
        plt.title('FOV Solar Contaminated Pixels')
        plt.show()

    sun_glint = xr.Dataset(dict(pixel=pixel,fov_contam_pixels=contam_pixels))

    return sun_glint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('output_file', type=str)


    args = parser.parse_args()


    reader_cls = get_reader_class(args.filename)
    reader = reader_cls(tle_dir="C:/Users/ny2/projectdir/pygac/gapfilled_tles",
                        tle_name="TLE_%(satname)s.txt",
                        calibration_method="noaa",
                        adjust_clock_drift=False)
    reader.read(args.filename)
    ds = reader.get_calibrated_dataset()
    mask = reader.mask

    output_file = args.output_file
    sunglint = sun_glint(ds, mask, plot=True)

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_file, 'w') as file:
        file.write(sunglint + time_string)

    print(sunglint)
