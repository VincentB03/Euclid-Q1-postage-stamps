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
