from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def create_initial_labels(data, features):
    #returns dataframe

    #extract features
    df = data.copy()
    X = df[features]

    #check for missing values
    missing = X.isnull().sum()
    if missing.any():
        X = X.fillna(X.mean())
    
    #standardize features, make dataframe
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

    #K-Means clustering, attempt 10 times
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    cluster_labels = kmeans.fit_predict(X_scaled)

    #get unique clusters
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} teams ({count/len(cluster_labels)*100:.1f}%)")

    centers = kmeans.cluster_centers_

    centers_og = scaler.inverse_transform(centers)
    centers_df = pd.DataFrame(centers_og, columns=features)
    centers_df.index = ['Cluster 0', 'Cluster 1', 'Cluster 2']

    #print(f'Centers dataframe {centers_df}')
    cluster_3par = centers_df['3PAr'].values
    print(cluster_3par)

    #distinguish clusters
    three_pt_cluster = np.argmax(cluster_3par)
    paint_cluster = np.argmin(cluster_3par)
    print(three_pt_cluster, paint_cluster)

    all_clusters = [0, 1, 2]
    balanced_cluster = [c for c in all_clusters if c not in [three_pt_cluster, paint_cluster]][0]
    print(balanced_cluster)


    cluster_playstyle = {
        three_pt_cluster: 'three_point_focused',
        paint_cluster: 'paint_focused',
        balanced_cluster: 'balanced',
    }

    df['initial_cluster'] = cluster_labels
    df['initial_cluster_name'] = df['initial_cluster'].map(cluster_playstyle)

    # print(df)

    return df


def refine_labels_with_percentiles(data, refinement_features):
    # """
    # Step 2: Refine cluster labels using percentile-based thresholds
    # """
    # print("\n[STEP 2] Refining labels with percentiles...")
    
    df = data.copy()
    
    # Calculate percentiles within each season
    # print(f"Calculating percentiles for: {refinement_features}")
    
    for feat in refinement_features:
        if feat in df.columns:
            df[f'{feat}_percentile'] = df.groupby('Season_Year')[feat].rank(pct=True)
        else:
            print(f"Warning: {feat} not found in data")
    
    # Define thresholds (33% for each classification)
    HIGH_THRESHOLD = 0.67
    LOW_THRESHOLD = 0.33
    
    def refine_single_label(row):
        initial = row['initial_cluster_name']
        
        # Strong 3-point indicators: High 3PA rate + longer avg distance
        is_strong_3pt = (row.get('3PAr_percentile', 0.5) > HIGH_THRESHOLD and 
                        row.get('Dist._percentile', 0.5) > HIGH_THRESHOLD)
        
        # Strong paint indicators: High rim frequency + low 3PA rate
        is_strong_paint = (row.get('freq_0_3_percentile', 0.5) > HIGH_THRESHOLD and 
                          row.get('3PAr_percentile', 0.5) < LOW_THRESHOLD)
        
        # Balanced indicators: Middle on both 3PA rate and rim frequency OR high mid-range
        is_balanced = ((LOW_THRESHOLD < row.get('3PAr_percentile', 0.5) < HIGH_THRESHOLD and
                       LOW_THRESHOLD < row.get('freq_0_3_percentile', 0.5) < HIGH_THRESHOLD) or
                       row.get('freq_16_3P_percentile', 0) > HIGH_THRESHOLD)
        
        # Adjust labels if thresholds met (percentiles)
        if is_strong_3pt and initial != 'three_point_focused':
            return 'three_point_focused'
        elif is_strong_paint and initial != 'paint_focused':
            return 'paint_focused'
        elif is_balanced and initial != 'balanced':
            return 'balanced'
        else:
            return initial
    
    # Show initial distribution
    print(f"\n  Initial distribution:")
    print(f"    {df['initial_cluster_name'].value_counts().to_dict()}")

    # Apply refinement
    df['final_playstyle_label'] = df.apply(refine_single_label, axis=1)
    
    # Count changes
    changes = (df['final_playstyle_label'] != df['initial_cluster_name']).sum()
    change_percent = changes / len(df) * 100
    print(f"\n  Refinement changed {changes} labels ({change_percent:.1f}%)")
    
    # Show final distribution
    print(f"\n  Final distribution:")
    print(f"    {df['final_playstyle_label'].value_counts().to_dict()}")
    
    # Show examples of changed labels
    changed = df[df['final_playstyle_label'] != df['initial_cluster_name']]
    if len(changed) > 0:
        print(f"\n  Sample reclassified teams:")
        print(changed[['Team_Acronym', 'Season_Year', 'initial_cluster_name', 
                      'final_playstyle_label', '3PAr', 'freq_0_3', 'Dist.']].head(10).to_string(index=False))
        
    # Drop all temporary columns: initial clusters and percentiles
    cols_to_drop = ['initial_cluster', 'initial_cluster_name'] + \
                   [col for col in df.columns if col.endswith('_percentile')]
    df = df.drop(columns=cols_to_drop)    
    
    return df


def main():
    features = ['3PAr',
            '3P%_per100',
            'freq_3PA_corner',
            'freq_0_3',
            'freq_layups',
            'freq_dunks',
            'freq_10_16',
            'freq_16_3P',
            'Pace',
            'AST_per100',
            'TOV%',
            'ORB%',
            'FTr',
            'Dist.',
            ]
    
    refinement_features = [
        '3PAr',
        'freq_0_3',
        'Dist.',
        'freq_16_3P',
    ]
    
    df = pd.read_csv('combined_dataframe.csv')
    train_data = df[df['Season_Year'] <= 2020]
    test_data = df[(df['Season_Year'] >= 2021) & (df['Season_Year'] <= 2024)]

    labeled_train_data = create_initial_labels(train_data, features)
    labeled_test_data = create_initial_labels(test_data, features)
    print(labeled_train_data)
    print(labeled_test_data)

    refined_train_data = refine_labels_with_percentiles(labeled_train_data, refinement_features)
    refined_test_data = refine_labels_with_percentiles(labeled_test_data, refinement_features)

    #send data to csv
    refined_train_data.to_csv('labeled_training_data.csv', index=False)
    refined_test_data.to_csv('labeled_test_data.csv', index=False)

if __name__ == '__main__':
    main()