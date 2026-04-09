# import pandas as pd

# # List of names to match
# names = [
#     "Alisa Williams",
#     "Ma'Riya Vincent",
#     "Janae Walker",
#     "Yacine Ndiaye",
#     "Makylah Moore",
#     "Audi Crooks",
#     "Divine Bourrage",
#     "Jada Williams",
#     "Addy Brown",
#     "Lilly Taulelei",
#     "Reese Beaty",
#     "Reagan Wilson",
#     "Aili Tanke",
#     "Amiya Redus",
#     "Kylie Torrence",
#     "Tanyuel Welch",
#     "Ramiyah Byrd",
#     "Yonta Vaughn",
#     "Donavia Hall",
#     "Victoria Rivera",
#     "Phoenix Stotijn",
#     "Faith Wiseman",
#     "Antonia Bates",
#     "Nene Ndiaye"
# ]

# # Convert to set for faster lookup
# name_set = set(names)

# # Read the CSV
# df = pd.read_csv("womens_all_players.csv")

# # Filter rows where full_name matches
# filtered_df = df[df["player_name"].isin(name_set)]

# # Write to new CSV
# filtered_df.to_csv("womens_transfers.csv", index=False)

# print(f"Saved {len(filtered_df)} matching players to womens_transfers.csv")

# import pandas as pd

# # Load the CSV
# df = pd.read_csv("womens_all_players.csv")

# # Create the new column
# df["general_minutes"] = df["gp"] * df["avg_min"]

# # (Optional) Save back to a new file
# df.to_csv("womens_all_players_updated.csv", index=False)

# # Preview
# print(df.head())



import pandas as pd

# Load the CSV
df = pd.read_csv("womens_transfers.csv")

# Add general_minutes column
df["general_minutes"] = df["gp"] * df["avg_min"]

# Rename columns
df = df.rename(columns={
    "player_name": "full_name",
    "team": "team_display_name",
    "position": "position_id",
    "class": "experience_years"
})

# (Optional) Save to a new file
df.to_csv("womens_transfers.csv", index=False)

# Preview
print(df.head())