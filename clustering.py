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
    
    df = pd.read_csv('combined_dataframe.csv')
    train_data = df[df['Season_Year'] <= 2020]

    labeled_data = create_initial_labels(train_data, features)
    print(labeled_data)

if __name__ == '__main__':
    main()