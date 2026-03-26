import pandas as pd

print("Loading datasets...")

clean_df = pd.read_csv("data/final/cleaned_issues.csv")
rl_df = pd.read_csv("data/final/final_rl_dataset.csv")

print("Creating clean_text...")

clean_df["clean_text"] = clean_df["clean_title"].fillna("") + " " + clean_df["clean_body"].fillna("")

print("Merging clean_text into RL dataset...")

rl_df = rl_df.merge(
    clean_df[["issue_number", "clean_text"]],
    on="issue_number",
    how="left"
)

missing = rl_df["clean_text"].isna().sum()
print(f"Missing clean_text rows: {missing}")

if missing > 0:
    print(" Warning: Some rows missing text!")

print("Saving updated dataset...")

rl_df.to_csv("data/final/final_rl_dataset.csv", index=False)

print("Done! clean_text added successfully.")