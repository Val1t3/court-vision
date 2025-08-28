import pandas as pd
import numpy as np

# --- Configuration ---
# Set the name of your input and output files
input_filename = '../data/courtvision-dataset/test_q4.csv'
output_filename = '../data/courtvision-dataset/test_tracks.csv'

# --- Script ---
try:
    print(f"Reading data from '{input_filename}'...")
    
    # 1. Read the CSV with multi-level headers.
    df = pd.read_csv(input_filename, header=[0, 1, 2], skiprows=[3], index_col=0)

    print("Reshaping and processing data...")

    # 2. Reshape the data, removing the now-incompatible 'dropna' argument.
    df_stacked = df.stack(level=[0, 1], future_stack=True)

    # 3. Move the index levels to columns and rename them robustly.
    df_long = df_stacked.reset_index()
    df_long.rename(columns={'level_0': 'frame', 'level_1': 'TeamID', 'level_2': 'PlayerID'}, inplace=True)

    # 4. Create the unique ID.
    df_long['id'] = np.where(
        df_long['TeamID'] == 'BALL', 
        'BALL', 
        df_long['TeamID'].astype(str) + '_' + df_long['PlayerID'].astype(str)
    )

    # 5. Calculate bounding box coordinates.
    df_long['x1'] = df_long['bb_left']
    df_long['y1'] = df_long['bb_top']
    df_long['x2'] = df_long['bb_left'] + df_long['bb_width']
    df_long['y2'] = df_long['bb_top'] + df_long['bb_height']

    # 6. Add a placeholder for confidence.
    df_long['confidence'] = 1.0

    # 7. Select and reorder columns for the final output.
    output_df = df_long[['frame', 'id', 'x1', 'y1', 'x2', 'y2', 'confidence']]
    
    # Remove rows where all coordinate data is missing.
    output_df.dropna(subset=['x1', 'y1', 'x2', 'y2'], inplace=True)

    # 8. Save the final DataFrame to a new CSV file.
    output_df.to_csv(output_filename, index=False, float_format='%.3f')

    print(f"✅ Success! Converted data saved to '{output_filename}'.")

except FileNotFoundError:
    print(f"❌ Error: The file '{input_filename}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")