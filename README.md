# lif_summary

### automated extraction of microscope images/ videos from a .lif-file

Since manually exporting acquired entries of Leica .lif-files is quite tedious (especially exporting videos with a defined codec and the correct framerate used for acquisition), this packages provides a simple solution to automatically export all included entries.
`lif_summary` will create a new folder for each .lif-file and extract all images and videos. Exported videos are (at least for my purpose) a useful compromise between compression and quality (e.g. subsequent analysis of beating kinetics of cardiac tissues are feasible in [OpenHeartWare](https://github.com/loslab/ohw).
Thanks to [readlif](https://github.com/nimne/readlif), lif-files can be directly accessed in Python.
Following exports are currently supported:
- images (xy): export of raw image as .tif, export of image with burned in scalebar + title as .jpg
- multichannel-images (xyc): export of raw multichannel-image as single .tif
- zstacks (xyz): export of raw images as series of .tifs
- videos (xyt): export of video in full resolution and low compression (h.264, crf=17), export of video in max. 1024 px length and higher compression (h.264, crf = 23)

## Installation
required packages:
  - `Python 3`
  - `numpy`
  - `readlif` for extracting data from .lif-file
  - `imageio-ffmpeg` providing a pip-installable version of ffmpeg
  - `PIL` and `skimage` for image operations  
  - `tifffile` for writing .tifs
  - `cv2` for image rescaling operations
  - `tqdm` for progress bars during export

install required packages in your Python environment and run `lif_summary.py` for any exports

## Usage

- Calling `python lif_summary.py` will export all supported entries from all .lif-files in the current folder
- for more precise export options following syntax can be used:

```python
from lif_summary import lif_summary

inputfile = "path/to/liffile.lif"
summary = lif_summary.lif_summary(inputfile)

summary.export_xy()     # exports all xy-entries
summary.export_xyc()    # exports all xyc-entries
summary.export_xyz()    # exports all xyz-entries
summary.export_xyt()    # exports all xyt-entries
summary.export_all()    # convenience function calling all of the export options above

```

## Changelog

#### 2.1
- add colorama to fix correct coloring in Windows terminals
- add setup.py and option to call from commandline with `lifsum` or `lif_summary`
- fix bug of correct outputvideo size
- introduce export of xyct-entries as multipage-tif
- introduce export of xyt-Fluo-entries as multipage-tif

#### 2.0
- switching reader to `readlif` for a python-only based lif-interface, no need to call `bioformats` via `javabridge` anymore
- video exports are now handled by directly piping frames to ffmpeg yielding more precise control of the output video codec and compression
- videos are now exported in full res and low compression as well as in max. 1024 px sidelength and higher compression
- introduce export of extractionlog + raw metadata (xml)
- popwerpoint-creation not supported yet

#### 1.0
- first stable release allowing the export of simple images, videos, zstacks and multichannel-images
- lif-files are accessed by `bioformats` via `javabridge`, videos exported by `OpenCV (cv2)`
- creation of a .ppt-summary of exported images