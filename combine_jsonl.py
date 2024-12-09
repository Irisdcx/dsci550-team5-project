import pandas as pd
import glob
# List all .jsonl files in the folder
file_paths = glob.glob("*.jsonl")

# Read and combine all .jsonl files into a single DataFrame
combined_df = pd.concat([pd.read_json(file, lines=True) for file in file_paths], ignore_index=True)
# Convert columns with dictionaries to strings
for col in combined_df.columns:
    if combined_df[col].apply(type).eq(dict).any():
        combined_df[col] = combined_df[col].apply(str)
# Convert list columns to strings
for col in combined_df.columns:
    if combined_df[col].apply(type).eq(list).any():
        combined_df[col] = combined_df[col].apply(str)

combined_df = combined_df.drop_duplicates()
combined_df.to_json("combined_file.jsonl", orient='records', lines=True)


# (Optional) Save to a CSV for easier inspection
combined_df.to_csv("combined_file.csv", index=False)