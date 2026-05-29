import os
from astroquery.esa.euclid import Euclid


PSF_SIZE = 21
STAMP_SIZE = 64
POINT_PROB = 0.5
DISTANCE = 46
PIXEL_SIZE = 0.1

DATA_DIR = '/content/drive/MyDrive/Q1_VIS_CALIBRATED_DB' #code used on Colab with Google Drive, change to local path if needed
os.makedirs(DATA_DIR, exist_ok=True)
Euclid.ROW_LIMIT = -1

QUADRANTS = [
    f"{i}-{j}.{letter}"
    for i in range(1, 7)
    for j in range(1, 7)
    for letter in ["E", "F", "G", "H"]
]

QUADRANT_DIR = os.path.join(DATA_DIR, 'quadrant-data')
os.makedirs(QUADRANT_DIR, exist_ok=True)