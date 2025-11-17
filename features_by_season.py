import pandas as pd
import time
import requests
from functools import reduce

TEAM_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",

    # old team mappings to current names
    "Charlotte Bobcats": "CHA",
    "New Jersey Nets": "BRK",
    "New Orleans Hornets": "NOP",
}

# mapping for duplicate column names
rename_map = {
    '2P_y': 'freq_2P',
    '0-3': 'freq_0_3',
    '3-10': 'freq_3_10',
    '10-16': 'freq_10_16',
    '16-3P': 'freq_16_3P',
    '3P_y': 'freq_3P',
    '2P.1': 'fg_2P',
    '0-3.1': 'fg_0_3',
    '3-10.1': 'fg_3_10',
    '10-16.1': 'fg_10_16',
    '16-3P.1': 'fg_16_3P',
    '3P.1': 'fg_3P',
    '2P.2': 'astd_2P_rate',
    '3P.2': 'astd_3P_rate',
    '%FGA': 'freq_dunks',
    'Md.': 'made_dunks',
    '%FGA.1': 'freq_layups',
    'Md..1': 'made_layups',
    '%3PA': 'freq_3PA_corner',
    '3P%_y': 'fg_3PA_corner',
    'Att.': "half-court_attempts",
    'Md..2': 'made_half-court',
}

def clean_team_col(df):
    # Your existing team cleaner (or a simple one like this)
    df = df.copy()

    df["Team"] = (
        df["Team"].astype(str)
        .str.replace(r"\*", "", regex=True)
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.strip()
    )
    bad = df["Team"].str.lower().isin(
        ["team", "rk", "league average", "western conference", "eastern conference"]
    )
    return df[~bad]


def flatten_and_clean_columns(df):
    """Flatten MultiIndex columns and drop 'Unnamed:' columns."""
    # Flatten multi-level headers to single strings
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in df.columns.to_list()
        ]
    else:
        df.columns = [str(c) for c in df.columns]

    # IMPORTANT: keep columns that do NOT start with 'Unnamed'
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def clean_simple_table(df):
    """
    For per-100 & advanced tables: flatten headers, drop Unnamed,
    standardize the Team column, and drop non-team rows.
    """
    if df is None:
        return None

    df = flatten_and_clean_columns(df)

    # Find Team/Tm
    team_like = [c for c in df.columns if c.strip().lower() in ("team", "tm")]
    if not team_like:
        return None

    if team_like[0] != "Team":
        df = df.rename(columns={team_like[0]: "Team"})

    # Use your existing team cleaner
    df = clean_team_col(df)
    return df


def fetch_league_tables_for_year(year):
            # response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
        try:
            tables_h0 = pd.read_html(url, header=0)
            tables_h1 = pd.read_html(url, header=1)
        except Exception as e:
            print(f"Error collecting {year}: {e}")
            return None
        
        if year < 2016:
            per_100_index = 6
            advanced_index = 8
            shooting_index = 9
        else:
            per_100_index = 8
            advanced_index = 10
            shooting_index = 11
        try:
            per100_raw = tables_h0[per_100_index].copy()
            # print("per 100 raw")
            # print(per100_raw.head())
            advanced_raw = tables_h1[advanced_index].copy()
            shooting_raw = tables_h1[shooting_index].copy()
        except Exception as e:
            print(f"Table index mismatch for {year}: {e}")
            return None
            
        per100 = clean_simple_table(per100_raw)
        advanced = clean_simple_table(advanced_raw)
        shooting = clean_simple_table(shooting_raw)

        frames = [t for t in (per100, advanced, shooting) if t is not None]
        if not frames:
            return pd.DataFrame(columns=["Team", "Season_Year"])

        # merge tables by Team
        merged = reduce(
            lambda left, right: pd.merge(left, right, on="Team", how="outer"),
            frames,
        )

        merged = merged.rename(columns={k: v for k, v in rename_map.items() if k in merged.columns})

        # add season year as column
        merged["Season_Year"] = int(year)
        return merged


def test_single_year(year):

    print(f"=== Testing fetch_league_tables_for_year({year}) ===")
    df = fetch_league_tables_for_year(year)

    if df is None or df.empty:
        print("No data returned! (df is None or empty)")
        return
    
    columns_to_remove = ['Rk_x', 'G_x', 'MP_x', 'Rk_y', 'Age', 'W', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'DRtg', 'NRtg', 'eFG%.1', 'TOV%.1', 'DRB%', 'FT/FGA.1', 
                         'Arena', 'Attend.', 'Attend./G', 'Rk', 'G_y', 'MP_y', 'FG%_y']
    
    df = df.drop(columns=columns_to_remove, errors='ignore')

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    with pd.option_context('display.max_columns', None):
        print(df.head())

    print("\nUnique teams:")
    print(sorted(df['Team'].unique()))

    print("\nSeason years present (should just be the test year):")
    print(df['Season_Year'].unique())

    return df

def build_dataframe(start_year=2010, end_year=2025, save_csv=False):
    years = range(start_year, end_year)
    dfs = []
    for y in years:
        df = fetch_league_tables_for_year(y)
        if df is None:
            print(f"[WARN] Skipping year {y} due to fetch error.")
            continue
        dfs.append(df)
        time.sleep(3)
    
    if not dfs:
        return pd.DataFrame()
    
    all_df = pd.concat(dfs, ignore_index=True)

    columns_to_remove = ['Rk_x', 'G_x', 'MP_x', 'Rk_y', 'Age', 'W', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'DRtg', 'NRtg', 'eFG%.1', 'TOV%.1', 'DRB%', 'FT/FGA.1', 
                            'Arena', 'Attend.', 'Attend./G', 'Rk', 'G_y', 'MP_y', 'FG%_y']
        
    # drop irrelevant columns
    all_df = all_df.drop(columns=columns_to_remove, errors='ignore')

    all_df["Season_Year"] = pd.to_numeric(all_df["Season_Year"], errors="coerce").astype("Int64")

    # if "Team_Acronym" not in all_df.columns:
    #     # map team names to abbreviations
    #     all_df["Team_Acronym"] = all_df["Team"].map(TEAM_ABBR)
    # map team names to abbreviations
    all_df["Team_Acronym"] = all_df["Team"].map(TEAM_ABBR)
    
    if save_csv:
            all_df.to_csv('test_raw_data.csv', index=False)
            print("Test data saved to 'test_raw_data.csv'")

    return all_df
    
def main(single_year, save_csv):
    if single_year:
        test_year = 2015
        df_older = test_single_year(test_year)

        test_year = 2016
        df_newer = test_single_year(test_year)
    else:
        years = range(2010, 2025)
        dfs = []

        for y in years:
            df = fetch_league_tables_for_year(y)
            if df is None:
                print(f"[WARN] Skipping year {y} due to fetch error.")
                continue
            dfs.append(df)
            time.sleep(3)

        all_df = pd.concat(dfs, ignore_index=True)

        columns_to_remove = ['Rk_x', 'G_x', 'MP_x', 'Rk_y', 'Age', 'W', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'DRtg', 'NRtg', 'eFG%.1', 'TOV%.1', 'DRB%', 'FT/FGA.1', 
                            'Arena', 'Attend.', 'Attend./G', 'Rk', 'G_y', 'MP_y', 'FG%_y']
        
        # drop irrelevant columns
        all_df = all_df.drop(columns=columns_to_remove, errors='ignore')

        # map team names to abbreviations
        all_df["Team_Acronym"] = all_df["Team"].map(TEAM_ABBR)

        missing = all_df[all_df["Team"].isna()]
        print(missing["Team"].unique())

        print("=== Combined DataFrame Shape ===")
        print(f"{all_df.shape[0]} rows × {all_df.shape[1]} columns\n")
        
        print("=== Combined DataFrame Columns ===")
        print(all_df.columns.tolist())

        print("=== Combined DataFrame Preview ===")
        print(all_df.head())
        
        # Save to csv
        if save_csv:
            all_df.to_csv('test_raw_data.csv', index=False)
            print("Test data saved to 'test_raw_data.csv'")
    

if __name__ == "__main__":
    # set True to save csv or False to skip
    main(single_year = False, save_csv=True)
