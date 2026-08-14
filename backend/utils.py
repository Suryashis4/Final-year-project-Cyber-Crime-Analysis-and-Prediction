import pandas as pd

# Official Chicago Police Department Districts used in your dataset
district_map = {
    1: "Central",
    2: "Wentworth",
    3: "Grand Crossing",
    4: "South Chicago",
    5: "Calumet",
    6: "Gresham",
    7: "Englewood",
    8: "Chicago Lawn",
    9: "Deering",
    10: "Ogden",
    11: "Harrison",
    12: "Near West",
    15: "Austin",
    17: "Albany Park",
    18: "Near North",
    19: "Town Hall",
    20: "Lincoln",
    24: "Rogers Park",
    25: "Grand Central",
}

# Map non-official / community areas to official CPD districts
community_to_official = {
    "Jefferson Park": "Albany Park",   # 17th District
    "Shakespeare": "Near North",       # 18th District
    "Morgan Park": "Wentworth",        # 2nd District
}

def get_district_table():
    return pd.DataFrame(
        [(k, v) for k, v in district_map.items()],
        columns=["District No", "District Name"]
    ).sort_values("District No")

def normalize_district(df):
    """
    1. Convert district numbers to names using district_map
    2. Replace community area names with official district names
    3. Keep only rows that belong to official CPD districts
    4. Add a clean 'District_Name' column
    """
    df = df.copy()

    # If District is numeric, map to name
    if pd.api.types.is_numeric_dtype(df["District"]):
        df["District_Name"] = df["District"].map(district_map)
    else:
        # If District is already text, use it directly
        df["District_Name"] = df["District"]

    # Replace community names with official districts
    df["District_Name"] = df["District_Name"].replace(community_to_official)

    # Keep only official districts
    df = df[df["District_Name"].isin(district_map.values())]

    return df