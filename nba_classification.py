import pandas as pd
import time
import random

from features_by_season import build_dataframe

TEAM_MAPPING = {
    'ATL': 'ATL',  # Atlanta Hawks
    'BOS': 'BOS',  # Boston Celtics
    'NJN': 'BRK',  # Brooklyn Nets (includes New Jersey history)
    'CHA': 'CHA',  # Charlotte Hornets (Note: BBR uses CHO, your CSV uses CHA)
    'CHI': 'CHI',  # Chicago Bulls
    'CLE': 'CLE',  # Cleveland Cavaliers
    'DAL': 'DAL',  # Dallas Mavericks
    'DEN': 'DEN',  # Denver Nuggets
    'DET': 'DET',  # Detroit Pistons
    'GSW': 'GSW',  # Golden State Warriors
    'HOU': 'HOU',  # Houston Rockets
    'IND': 'IND',  # Indiana Pacers
    'LAC': 'LAC',  # Los Angeles Clippers
    'LAL': 'LAL',  # Los Angeles Lakers
    'MEM': 'MEM',  # Memphis Grizzlies (includes Vancouver history)
    'MIA': 'MIA',  # Miami Heat
    'MIL': 'MIL',  # Milwaukee Bucks
    'MIN': 'MIN',  # Minnesota Timberwolves
    'NOH': 'NOP',  # New Orleans Pelicans
    'NYK': 'NYK',  # New York Knicks
    'OKC': 'OKC',  # Oklahoma City Thunder (includes Seattle history)
    'ORL': 'ORL',  # Orlando Magic
    'PHI': 'PHI',  # Philadelphia 76ers
    'PHO': 'PHX',  # Phoenix Suns
    'POR': 'POR',  # Portland Trail Blazers
    'SAC': 'SAC',  # Sacramento Kings
    'SAS': 'SAS',  # San Antonio Spurs
    'TOR': 'TOR',  # Toronto Raptors
    'UTA': 'UTA',  # Utah Jazz
    'WAS': 'WAS'   # Washington Wizards
}

def collect_team_data(start_year=2010, end_year=2024):
    all_data = []
    total_teams = len(TEAM_MAPPING)
    
    for idx, (url_abbr, current_abbr) in enumerate(TEAM_MAPPING.items(), 1):
        # print(f"[{idx}/{total_teams}] Collecting {current_abbr} from URL: {url_abbr}")
        
        url = f"https://www.basketball-reference.com/teams/{url_abbr}/stats_basic_totals.html"
        
        try:
            tables = pd.read_html(url)
            team_data = tables[0].copy()
            
            team_data['Team_Acronym'] = current_abbr
            
            all_data.append(team_data)
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Error collecting {url_abbr}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data collected")
    
    print(f"\nCollected data from {len(all_data)} teams")
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Extract starting year from Season
    df['Season_Year'] = df['Season'].astype(str).str.extract(r'^(\d{4})')[0]
    df['Season_Year'] = pd.to_numeric(df['Season_Year'], errors='coerce')
    
    # Filter to year range
    df = df[(df['Season_Year'] >= start_year) & (df['Season_Year'] <= end_year)]
    
    print(f"Filtered to years {start_year}-{end_year}: {len(df)} team-seasons")
    
    return df

def clean_team_data(df):
    #Returns
        #Cleaned up dataframe
    
    clean_df = df.copy()

    # print(clean_df.columns.tolist())

    clean_df = clean_df.loc[:, ~clean_df.columns.str.contains('^Unnamed')]

    columns_to_remove = ['Lg', 'Tm', 'Ht.', 'Wt.']
    
    clean_df = clean_df.drop(columns_to_remove, axis=1, errors='ignore')

    clean_df = clean_df[clean_df['Season'] != 'Season']

    clean_df = clean_df[~clean_df['Season'].str.contains('Career', na=False)]

    numeric_columns = ['W', 'L', 'Finish', 'Age', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 
               'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season_Year']
    
    for col in numeric_columns:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

    #missing values in columns
    # print(df.isnull().sum())

    return clean_df

def combine_dataframe(csv1, csv2):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    combined_df = pd.merge(
        df1,
        df2,
        how='outer',
        on=['Team_Acronym', 'Season_Year'],
        suffixes=('_total', '_per100')
    )

    other_cols = [col for col in combined_df.columns if col not in ['Team_Acronym', 'Season_Year']]

    combined_df = combined_df[['Team_Acronym', 'Season_Year'] + other_cols]

    return combined_df

def main():
    start_year = 2010
    end_year = 2025

    start_time = time.time()
    # df = collect_team_data()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    
    # clean_df = clean_team_data(df)
    # n_rows = len(df)

    print("Exporting dataframe into CSV file")
    # clean_df.to_csv('team_data_1.csv', index=False)

    # df2 = build_dataframe(start_year + 1, end_year + 1, save_csv=True)

    csv1 = "team_data_1.csv"
    csv2 = "test_raw_data.csv"

    combined_df = combine_dataframe(csv1, csv2)

    combined_df.to_csv("combined_dataframe.csv", index=False)
    
    
if __name__ == "__main__":
    main()
