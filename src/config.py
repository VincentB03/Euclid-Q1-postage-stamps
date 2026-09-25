import os
from astroquery.esa.euclid import Euclid


PSF_SIZE = 21           # PSF stamp side (px)
STAMP_SIZE = 64         # science / noise / mask stamp side (px)
POINT_PROB = 0.5        # max point_like_prob kept
DISTANCE = 46           # isolation radius (px), ~half the stamp diagonal
PIXEL_SIZE = 0.1        # VIS pixel scale (arcsec/px)

# Catalogue cuts (select_sources)
FLUX_MIN = 0.57544
FLUX_MAX = 575.44
MAX_SPURIOUS_PROB = 0.2

# Stamp cuts
FLAG_BITMASK = 1                 # VIS FLG bits counted as bad pixels
MAX_BAD_PIXEL_FRACTION = 0.08    # drop a stamp at/above this fraction of bad pixels
SEGMENTATION_NSIGMA = 3.0        # segmentation threshold, in units of noise_map
SEGMENTATION_CENTER_BOX = 5      # central box (px) whose regions make up the source

HF_REPO_ID = 'VincentB03/Euclid-Q1-VF'   # default --repo-id

# EUCLID_DATA_DIR, else the Colab Drive folder if mounted, else <repo>/data/Q1_VIS_CALIBRATED_DB
_DRIVE_DATA_DIR = '/content/drive/MyDrive/Q1_VIS_CALIBRATED_DB'
_LOCAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'Q1_VIS_CALIBRATED_DB',
)


def _default_data_dir():
    return _DRIVE_DATA_DIR if os.path.isdir('/content/drive/MyDrive') else _LOCAL_DATA_DIR


DATA_DIR = os.path.expanduser(os.environ.get('EUCLID_DATA_DIR') or _default_data_dir())
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