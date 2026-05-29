from astroquery.esa.euclid import Euclid
import os
import glob


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