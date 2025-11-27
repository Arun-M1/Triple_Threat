import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier


def train_svm(data, features):
    #return SVM, scaler, X_scaled, y
    X = data[features].copy()
    y = data['playstyle_label'].copy()

    X = X.fillna(X.mean())

    print("Before scaling:")
    print(X.describe())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nAfter scaling:")
    print(pd.DataFrame(X_scaled, columns=features).describe())

    svm_model = SVC(kernel='rbf', random_state=42, C=1)
    svm_model.fit(X_scaled, y)

    return svm_model, scaler, X_scaled, y

def test_svm(data, features, SVM, scaler):
    #return dataframe
    df = data.copy()
    X_test = df[features].copy()
    X_test = X_test.fillna(X_test.mean())

    X_test_scaled = scaler.transform(X_test)

    y_pred = SVM.predict(X_test_scaled)
    df['svm_pred'] = y_pred

    return df

def train_catboost(data, features):
    X = data[features].copy()
    y = data['playstyle_label'].copy()

    X = X.fillna(X.mean())
    print("Training data summary:")
    print(X.describe())

    model = CatBoostClassifier(
        loss_function = 'MultiClass',
        eval_metric = 'Accuracy',
        learning_rate = 0.1,
        depth = 6,
        iterations = 500,
        random_seed = 42,
        verbose = False
    )

    model.fit(X, y)

    return model, X, y

def test_catboost(data, features, model):
    df = data.copy()
    X_test = df[features].copy()
    X_test = X_test.fillna(X_test.mean())

    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).ravel()

    df['catboost_pred'] = y_pred

    return df

def evaluate_model(name, y_true, y_pred, label_order):
    print(f"\n{name} Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_order,
        yticklabels=label_order,
        cmap='Blues' if "Train" in name else 'Greens'
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

def run_svm(train_df, test_df, features, label_order):
    svm_model, scaler, X_scaled, y_train = train_svm(train_df, features)
    y_train_pred = svm_model.predict(X_scaled)
    evaluate_model("SVM Train", y_train, y_train_pred, label_order)

    svm_test_df = test_svm(test_df, features, svm_model, scaler)
    y_test_true = svm_test_df['playstyle_label']
    y_test_pred = svm_test_df['svm_pred']
    evaluate_model("SVM Test", y_test_true, y_test_pred, label_order)

    return svm_model, scaler, svm_test_df

def run_catboost(train_df, test_df, features, label_order):
    catboost_model, X_train, y_train = train_catboost(train_df, features)
    y_train_pred = np.array(catboost_model.predict(X_train)).ravel()
    evaluate_model("Catboost Train", y_train, y_train_pred, label_order)

    catboost_test_df = test_catboost(test_df, features, catboost_model)
    y_test_true = catboost_test_df['playstyle_label']
    y_test_pred = catboost_test_df['catboost_pred']
    evaluate_model("Catboost Test", y_test_true, y_test_pred, label_order)

    return catboost_model, catboost_test_df

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
    
    labels_order = ['balanced', 'paint_focused', 'three_point_focused']

    #svm
    train_df = pd.read_csv('labeled_training_data.csv')
    test_df = pd.read_csv('labeled_test_data.csv')


    percentile_features = [f"{feat}_pct" for feat in features]

    svm_model, scaler, svm_test_df = run_svm(train_df, test_df, percentile_features, labels_order)

    cb_model, cb_test_df = run_catboost(train_df, test_df, percentile_features, labels_order)

if __name__ == '__main__':
    main()