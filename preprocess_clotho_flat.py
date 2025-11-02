import pandas as pd
import os

# Load the single CSV (adjust filename if different, e.g., 'clotho_captions.csv')
csv_path = 'clotho_captions_development.csv'  # Update if your file name differs
df = pd.read_csv(csv_path)

# Generate metadata
all_data = []
for index, row in df.iterrows():
    file_name = row['file_name']
    for i in range(1, 6):  # caption_1 to caption_5
        caption = row[f'caption_{i}']
        if pd.notna(caption):  # Skip NaN captions
            all_data.append({
                'text': caption,
                'audio_path': f'clotho_16k/{file_name}'  # Relative path to resampled audio
            })

# Save to metadata.csv
pd.DataFrame(all_data).to_csv('dataset/metadata.csv', index=False)
print(f"Generated metadata.csv with {len(all_data)} entries")