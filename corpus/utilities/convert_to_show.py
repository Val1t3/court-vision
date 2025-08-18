import pandas as pd
import json
from io import StringIO

with open('../data/courtvision-dataset/eval_1_tracks.csv', 'r') as file:
    csv_data = file.read()

# Read the CSV
df = pd.read_csv(StringIO(csv_data))

# Keep only rows where id == 1
df_id1 = df[df['id'] == 1]

# Calculate the center point
coords = [
    [
        int(row['x1'] + row['x2']) // 2,
        int(row['y1'] + row['y2']) // 2
    ]
    for _, row in df_id1.iterrows()
]

# Convert to JSON
json_str = json.dumps(coords, indent=4)
print(json_str)
