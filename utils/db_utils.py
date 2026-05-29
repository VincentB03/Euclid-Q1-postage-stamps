import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.esa.euclid import Euclid


def get_optimal_observation_ids(round_decimals=1, verbose=True):
    """
    Retrieves the optimal observation IDs (1st dither only) to obtain
    a unique spatial coverage on the VIS instrument in Q1 data.

    Args:
        round_decimals (int): Number of decimals for rounding RA/DEC coordinates (default: 1).
        verbose (bool): Shows or hides information messages and previews (default: True).

    Returns:
        list: A list containing the 'observation_id' as strings.
    """
    if verbose:
        print("Launching query: calculating the optimal number of images...")

    query = """
    SELECT DISTINCT observation_id, ra, dec
    FROM q1.calibrated_frame
    WHERE instrument_name = 'VIS'
    """

    # Execute the query via the Euclid API
    job = Euclid.launch_job_async(query)
    raw_observations = job.get_results()

    # Convert to pandas DataFrame
    df = raw_observations.to_pandas()

    if verbose:
        print("Calculating optimal spatial coverage (1st dither only)...\n")

    # Isolate the 1st dither by removing duplicate observation IDs
    df_first_dither = df.drop_duplicates(subset=['observation_id']).copy()

    # Spatial rounding to approximate the tiles
    df_first_dither['ra_round'] = df_first_dither['ra'].round(round_decimals)
    df_first_dither['dec_round'] = df_first_dither['dec'].round(round_decimals)

    # Spatial deduplication and sorting by declination
    optimal_tiles = df_first_dither.drop_duplicates(subset=['ra_round', 'dec_round'])
    optimal_tiles = optimal_tiles.sort_values(by='dec', ascending=False)

    if verbose:
        print(f"-> Based on this approximation, we have {len(optimal_tiles)} unique tiles.")
        print("\nPreview of the first saved tiles:")
        print(optimal_tiles[['ra', 'dec', 'ra_round', 'dec_round', 'observation_id']].head())

    # Generate and return the final observation ID list
    return optimal_tiles['observation_id'].tolist()

def sync_calibrated_frames(obs_id_list, data_dir, verbose=True):
    """
    Locates and downloads the required Euclid VIS calibrated frame FITS files 
    for the given observation IDs (specifically filtering for dither '00-1').

    Args:
        obs_id_list (list): List of observation IDs to check and download.
        data_dir (str): Directory path where FITS files are stored locally.
        verbose (bool): Shows or hides detailed console logs (default: True).

    Returns:
        dict: A dictionary mapping observation_id to its local file path.
    """
    if verbose:
        print(f"Locating file paths for {len(obs_id_list)} observations...")

    # Scan existing local files to avoid redundant downloads
    existing = glob.glob(os.path.join(data_dir, '*.fits'))
    existing_files = {os.path.basename(f): f for f in existing}

    # Build SQL LIKE conditions dynamically for the ADQL query
    like_conditions = " OR ".join([f"file_name LIKE '%-{str(obs_id).zfill(6)}-%'" for obs_id in obs_id_list])

    adql_query = f"""
    SELECT file_name
    FROM q1.calibrated_frame
    WHERE instrument_name = 'VIS'
    AND ({like_conditions})
    """

    if verbose:
        print("  Querying the Euclid Archive...")
    job = Euclid.launch_job_async(query=adql_query)
    vis_table = job.get_results()

    needed_downloads = []
    frame_files = {}

    # Parse archive results and filter for the first dither (00-1)
    for obs_id in obs_id_list:
        obs_id_padded = str(obs_id).zfill(6)

        # Find archive rows matching the padded observation ID
        matches_for_obs = [r for r in vis_table if f"-{obs_id_padded}-" in r['file_name']]

        if not matches_for_obs:
            if verbose:
                print(f"  WARNING: No files found in archive for obs_id {obs_id}.")
            continue

        # Look specifically for the first dither '00-1'
        dither_1_matches = [r for r in matches_for_obs if '-00-1-' in r['file_name']]

        if not dither_1_matches:
            if verbose:
                print(f"  WARNING: Dither '00-1' not found for obs_id {obs_id}.")
            continue

        target_file_name = dither_1_matches[0]['file_name']

        # Determine if the file exists locally or needs to be downloaded
        if target_file_name in existing_files:
            outpath = existing_files[target_file_name]
            frame_files[obs_id] = outpath
            if verbose:
                print(f"  FOUND on disk: {target_file_name}")
        else:
            needed_downloads.append((obs_id, target_file_name))
            if verbose:
                print(f"  MISSING: {target_file_name} (obs_id {obs_id}), queued for download.")

    # Process downloads if any files are missing
    if needed_downloads:
        if verbose:
            print(f"\nDownloading {len(needed_downloads)} missing files")

        for obs_id, fname in needed_downloads:
            outpath = os.path.join(data_dir, fname)
            if verbose:
                print(f"  Downloading: {fname}")

            Euclid.get_product(file_name=fname, output_file=outpath)
            frame_files[obs_id] = outpath
    else:
        if verbose:
            print("\nAll required files are already present locally.")

    if verbose:
        print("Operation completed successfully.\n")

    return frame_files


def sync_background_frames(frame_files, obs_id_list, data_dir, verbose=True):
    """
    Locates and downloads the required Euclid VIS background (BKG) FITS files 
    associated with the successfully resolved science (calibrated) frames.

    Args:
        frame_files (dict): Dictionary mapping observation IDs to their local science file paths.
        obs_id_list (list): List of observation IDs to construct the query filters.
        data_dir (str): Directory path where FITS files are stored locally.
        verbose (bool): Shows or hides detailed console logs (default: True).

    Returns:
        dict: A dictionary mapping observation_id to its local background file path.
    """
    if verbose:
        print(f"Locating background (BKG) files for {len(frame_files)} images...")

    # Find existing local background files to avoid redundant downloads
    bkg_existing = glob.glob(os.path.join(data_dir, '*BKG*.fits'))
    bkg_existing_files = {os.path.basename(f): f for f in bkg_existing}

    # Build SQL LIKE conditions dynamically for the ADQL query
    like_conditions = " OR ".join([f"file_name LIKE '%-{str(obs_id).zfill(6)}-%'" for obs_id in obs_id_list])

    adql_query_bkg = f"""
    SELECT file_name
    FROM q1.aux_calibrated
    WHERE instrument_name = 'VIS'
      AND stype = 'BKG'
      AND ({like_conditions})
    """

    if verbose:
        print("  Querying the Euclid Archive (Backgrounds)...")
    job_bkg = Euclid.launch_job_async(query=adql_query_bkg)
    bkg_table = job_bkg.get_results()

    needed_downloads_bkg = []
    bkg_files = {}

    # Match background files to their corresponding local science files
    for obs_id, sci_path in frame_files.items():
        sci_filename = os.path.basename(sci_path)
        parts = sci_filename.split('-')
        
        # Reconstruct the expected background file pattern from the science filename
        if len(parts) >= 5:
            expected_bkg_core = f"{parts[0]}-BKG-{parts[2]}-{parts[3]}-{parts[4]}"
        else:
            expected_bkg_core = f"-BKG-{str(obs_id).zfill(6)}"

        # Find the row in the query results matching the expected background core string
        matching_bkg_rows = [r for r in bkg_table if expected_bkg_core in r['file_name']]

        if not matching_bkg_rows:
            if verbose:
                print(f"  WARNING: BKG file not found for pattern '{expected_bkg_core}' (Science: {sci_filename})")
            continue

        exact_bkg_name = matching_bkg_rows[0]['file_name']

        # Determine if the background file exists locally or needs downloading
        if exact_bkg_name in bkg_existing_files:
            outpath = bkg_existing_files[exact_bkg_name]
            bkg_files[obs_id] = outpath
            if verbose:
                print(f"  FOUND on disk (BKG): {exact_bkg_name}")
        else:
            needed_downloads_bkg.append((obs_id, exact_bkg_name))
            if verbose:
                print(f"  MISSING (BKG): {exact_bkg_name}")

    # Process background downloads if any files are missing
    if needed_downloads_bkg:
        if verbose:
            print(f"\nDownloading {len(needed_downloads_bkg)} missing background files...")
            
        for obs_id, fname in needed_downloads_bkg:
            outpath = os.path.join(data_dir, fname)
            if verbose:
                print(f"  Downloading BKG: {fname}")
                
            Euclid.get_product(file_name=fname, output_file=outpath)
            bkg_files[obs_id] = outpath
    else:
        if verbose:
            print("  All required background files are already present locally.")

    if verbose:
        print("Background sync completed successfully.\n")

    return bkg_files


def sync_psf_model(data_dir, verbose=True):
    """
    Locates the global Euclid VIS Point Spread Function (PSF) model FITS file.
    Checks local storage first (including a 'psf_models' subfolder), 
    then queries the archive and downloads it if missing.

    Args:
        data_dir (str): Directory path where FITS files are stored locally.
        verbose (bool): Shows or hides detailed console logs (default: True).

    Returns:
        str: The absolute local file path to the resolved PSF model FITS file.
    """
    if verbose:
        print("Locating PSF model file...")

    # Check for existing PSF files locally (either in a subfolder or root data directory)
    psf_existing = glob.glob(os.path.join(data_dir, 'psf_models', 'EUC_VIS_GRD-PSF-*.fits'))
    if not psf_existing:
        psf_existing = glob.glob(os.path.join(data_dir, 'EUC_VIS_GRD-PSF-*.fits'))

    if psf_existing:
        psf_full_path = psf_existing[0]
        if verbose:
            print(f"  FOUND on disk: {os.path.basename(psf_full_path)}")
        return psf_full_path

    # If missing locally, query the Euclid metadata archive
    if verbose:
        print("  Querying archive for PSF file...")
        
    adql_query_psf = """
        SELECT DISTINCT file_name
        FROM q1.aux_calibrated
        WHERE instrument_name = 'VIS'
          AND stype = 'PSF MODEL'
    """
    
    job_psf = Euclid.launch_job_async(query=adql_query_psf)
    psf_res = job_psf.get_results()
    
    if len(psf_res) == 0:
        raise FileNotFoundError("Error: No PSF model file found in the Euclid archive.")

    psf_fname = psf_res['file_name'][0]
    psf_full_path = os.path.join(data_dir, psf_fname)
    
    # Process the download
    if verbose:
        print(f"  Downloading: {psf_fname}")
        
    Euclid.get_product(file_name=psf_fname, output_file=psf_full_path)
    
    if verbose:
        print("PSF model sync completed successfully.\n")
        
    return psf_full_path


def sync_observation_catalogs(obs_id_list, data_dir, verbose=True):
    """
    Computes spatial footprints from local science images, queries the Euclid Archive 
    for cross-matched multi-table catalogs (MER + PHZ), and saves them as FITS tables.

    Args:
        obs_id_list (list): List of observation IDs to process.
        data_dir (str): Directory path where science images are stored and catalogs will be saved.
        verbose (bool): Shows or hides detailed console logs (default: True).

    Returns:
        dict: A dictionary mapping observation_id to its local catalog FITS file path.
    """
    if verbose:
        print(f"Syncing catalogs for {len(obs_id_list)} observation IDs...")

    # Look for the large local science detection frames to compute spatial footprints
    large_sci_files = glob.glob(os.path.join(data_dir, '*-DET-*.fits'))
    catalog_paths = {}

    for obs_id in obs_id_list:
        obs_id_padded = str(obs_id).zfill(6)
        cat_dst = os.path.join(data_dir, f'catalogue_obs_{obs_id_padded}.fits')

        # Check if the catalog file is already present on disk
        if os.path.exists(cat_dst):
            catalog_paths[obs_id] = cat_dst
            if verbose:
                print(f"\n  SKIP: Catalog for obs_id {obs_id} already exists ({os.path.basename(cat_dst)}).")
            continue

        # Find the corresponding science image to extract WCS footprint
        matching_files = [f for f in large_sci_files if f"-{obs_id_padded}-" in os.path.basename(f)]

        if not matching_files:
            if verbose:
                print(f"\n  WARNING: Science image not found for obs_id {obs_id}. Cannot compute footprint.")
            continue

        big_image_path = matching_files[0]
        if verbose:
            print(f"\n  Computing spatial footprint from: {os.path.basename(big_image_path)}")

        ra_all, dec_all = [], []
        try:
            # Parse all .SCI extensions to map out the total bounding box
            with fits.open(big_image_path) as hdul:
                for ext in hdul:
                    if ext.name.endswith('.SCI'):
                        h = ext.header
                        w = WCS(h)
                        nx, ny = h['NAXIS1'], h['NAXIS2']

                        # Coordinate pairs for the 4 corners of this specific detector
                        corners = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny]], dtype=float)
                        ra, dec = w.all_pix2world(corners[:, 0], corners[:, 1], 0)
                        ra_all.extend(ra)
                        dec_all.extend(dec)
        except Exception as e:
            if verbose:
                print(f"  Error: Failed to read WCS coordinates for obs_id {obs_id}: {e}")
            continue

        if not ra_all:
            if verbose:
                print(f"  WARNING: No coordinates extracted for obs_id {obs_id}.")
            continue

        # Extract strict bounding box limits
        ra_min, ra_max = min(ra_all), max(ra_all)
        dec_min, dec_max = min(dec_all), max(dec_all)
        
        if verbose:
            print(f"    Global search area: RA=[{ra_min:.4f}, {ra_max:.4f}], DEC=[{dec_min:.4f}, {dec_max:.4f}]")
            print("    Launching ADQL query to Euclid Archive...")

        # Construct comprehensive cross-matched ADQL Query
        query = f"""
        SELECT
        m.object_id,
        m.right_ascension,
        m.declination,
        m.right_ascension_psf_fitting,
        m.declination_psf_fitting,

        m.FLUX_VIS_1FWHM_APER, m.FLUXERR_VIS_1FWHM_APER,
        m.flux_vis_2fwhm_aper, m.fluxerr_vis_2fwhm_aper,
        m.flux_vis_3fwhm_aper, m.fluxerr_vis_3fwhm_aper,
        m.flux_vis_psf, m.fluxerr_vis_psf,
        m.flux_detection_total, m.fluxerr_detection_total,
        m.flux_segmentation, m.fluxerr_segmentation,

        m.semimajor_axis, m.semimajor_axis_err,
        m.ellipticity, m.ellipticity_err,
        m.position_angle, m.position_angle_err,
        m.kron_radius, m.kron_radius_err, m.fwhm,

        m.point_like_flag, m.point_like_prob,
        m.extended_flag, m.extended_prob,
        m.spurious_flag, m.spurious_prob,

        m.flag_vis, m.vis_det,
        m.deblended_flag, m.det_quality_flag,

        m.mu_max, m.mumax_minus_mag, m.segmentation_area, m.segmentation_map_id,

        p.FLUX_VIS_UNIF,

        morph.disk_sersic_sersic_index

        FROM catalogue.mer_catalogue AS m

        INNER JOIN catalogue.phz_photo_z AS p
        ON m.object_id = p.object_id

        INNER JOIN catalogue.mer_morphology AS morph
        ON morph.object_id = m.object_id

        WHERE m.right_ascension BETWEEN {ra_min} AND {ra_max}
          AND m.declination BETWEEN {dec_min} AND {dec_max}
        """

        try:
            # Query the database
            job_cat = Euclid.launch_job_async(query)
            cat = job_cat.get_results()

            if len(cat) > 0:
                if verbose:
                    print(f"    SUCCESS: {len(cat)} sources found for this observation.")
                # Save as a local FITS file table
                cat.write(cat_dst, format='fits', overwrite=True)
                catalog_paths[obs_id] = cat_dst
                if verbose:
                    print(f"    File saved: {os.path.basename(cat_dst)}")
            else:
                if verbose:
                    print("    WARNING: Query succeeded but returned an empty catalog for this region.")

        except Exception as e:
            if verbose:
                print(f"    ERROR: Failed to download catalog for obs_id {obs_id}: {e}")

    if verbose:
        print("\nCatalog synchronization pipeline complete.\n")

    return catalog_paths