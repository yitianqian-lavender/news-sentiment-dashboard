import pandas as pd
import numpy as np
from datetime import datetime

RAW_FILE   = "bbc_news.csv"         # rename if needed
OUT_FILE   = "bbc_clean_5k.csv"
CHUNK_SIZE = 100_000                # adjust if you have plenty of RAM
TARGET_N   = 5_000                  # final sample size

# ─── reservoir for random sample ───
reservoir = []
seen = 0

start = datetime(2022, 1, 1).date()
end   = datetime(2025, 8, 1).date()

use_cols = ["title", "pubDate"]     # adjust if column names differ

for chunk in pd.read_csv(
        RAW_FILE,
        usecols=use_cols,
        chunksize=CHUNK_SIZE):
    
    # keep only 2022-01-01 .. 2025-08-01
    chunk["pubDate"] = pd.to_datetime(chunk["pubDate"], errors="coerce").dt.date
    chunk = chunk[(chunk["pubDate"] >= start) & (chunk["pubDate"] <= end)]
    chunk = chunk.dropna(subset=["title", "pubDate"])

    for row in chunk.itertuples(index=False):
        seen += 1
        if len(reservoir) < TARGET_N:
            reservoir.append(row)
        else:
            j = np.random.randint(0, seen)
            if j < TARGET_N:
                reservoir[j] = row

# convert reservoir → DataFrame
sample_df = pd.DataFrame(reservoir, columns=["title", "date"]).sort_values("date")

# save exactly two columns, UTF-8
sample_df.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"✓ Saved {len(sample_df)} rows → {OUT_FILE}")
