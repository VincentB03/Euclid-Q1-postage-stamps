import os
from astroquery.esa.euclid import Euclid


PSF_SIZE = 21           # side of an interpolated PSF stamp (EuclidPSFModel)
STAMP_SIZE = 64         # side of a science/noise/mask postage stamp
POINT_PROB = 0.5        # max point_like_prob kept (more point-like -> dropped)
DISTANCE = 46           # stamp "diagonal" used in the isolation cut, in pixels
PIXEL_SIZE = 0.1        # VIS pixel scale, arcsec/pixel

# Catalogue quality cuts (dataset_builder.select_sources)
FLUX_MIN = 0.57544
FLUX_MAX = 575.44
MAX_SPURIOUS_PROB = 0.2

# Stamp quality
FLAG_BITMASK = 1       # VIS FLG bits rejected ("bad pixels in Euclid Data Product Description")
MAX_BAD_PIXEL_FRACTION = 0.08    # drop a stamp at/above this fraction of flagged pixels

# Empty-stamp cut (dataset_builder.drop_empty_stamps)
EMPTY_STAMP_CENTER_FRAC = 0.05    # fraction of the stamp size used as the "central" box
EMPTY_STAMP_SNR_THRESHOLD = 3.5   # peak SNR in that box below this -> no galaxy at center

# Hugging Face target dataset
HF_REPO_ID = 'VincentB03/euclid-Q1-V2'

# Data location. Set the EUCLID_DATA_DIR environment variable to either a Google
# Drive path (Colab) or any local path. If it is unset, fall back to the Drive
# mount point when running on Colab, otherwise to a local 'data' folder at the
# repository root.
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