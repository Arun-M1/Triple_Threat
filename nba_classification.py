import pandas as pd
import time

def collect_team_data(start_year=2010, end_year=2025):
    #Returns:
        #Dataframe of team statistics
    
    # nba_teams = ["ATL", "BOS", "NJN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW","HOU", "IND", "LAC", "LAL", "MEM",
    #              "MIA", "MIL", "MIN", "NOH", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
    nba_teams = ['ATL']
    
    dataset = []
    num_teams = len(nba_teams)

    for i, team in enumerate(nba_teams, 1):
        url = f"https://www.basketball-reference.com/teams/{team}/stats_basic_totals.html"

        try:
            tables = pd.read_html(url)
            team_data = tables[0].copy()
            team_data['Team_Acronym'] = team
            # print(f"Data retrieved: {team_data}")

            dataset.append(team_data)

            time.sleep(2)

        except Exception as e:
            print(f"Error collecting {team}: {e}")
            continue
    
    #check if all team data collected
    if not dataset:
        raise ValueError("No data collected.")
    else:
        print(f"All data collected")

    df = pd.concat(dataset, ignore_index=True)

    df['Season_Year'] = df['Season'].astype(str).str.split('-').str[-1]
    df['Season_Year'] = '20' + df['Season_Year']  #add 20 to end year to become 2024, etc.
    df['Season_Year'] = pd.to_numeric(df['Season_Year'], errors='coerce')

    df = df[(df['Season_Year'] > start_year) & (df['Season_Year'] <= end_year)]

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

def main():
    df = collect_team_data()
    clean_df = clean_team_data(df)

    print(f"Total rows: {len(clean_df)}")
    print(f"Missing values: {clean_df.isnull().sum().sum()}")
    print(clean_df.columns.tolist())

    return clean_df
    
if __name__ == "__main__":
    df = main()
    print(df)
