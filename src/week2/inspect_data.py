from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "dataset" / "job_skill_set.parquet"


def main():
    df = pd.read_parquet(DATASET_PATH)

    print("Dataset inspection")
    print("==================")
    print(f"Path: {DATASET_PATH}")
    print(f"Number of rows: {len(df)}")
    print(f"Column names: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nCategory counts:")
    print(df["category"].value_counts())
    print("\njob_skill_set Python value types:")
    print(df["job_skill_set"].map(lambda value: type(value).__name__).value_counts())
    print("\nExample fields:")
    for _, row in df.head(3).iterrows():
        print("-" * 80)
        print(f"job_id: {row['job_id']}")
        print(f"category: {row['category']}")
        print(f"job_title: {row['job_title']}")
        print(f"job_description sample: {str(row['job_description'])[:500]}")
        print(f"job_skill_set: {row['job_skill_set']}")


if __name__ == "__main__":
    main()
