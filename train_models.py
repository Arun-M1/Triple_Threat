import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
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


def train_mlp(data, features):
    X = data[features].copy()
    y = data['playstyle_label'].copy()

    X = X.fillna(X.mean())

    print("Before scaling:")
    print(X.describe())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nAfter scaling:")
    print(pd.DataFrame(X_scaled, columns=features).describe())

    # Encode string labels -> ints
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=42,
    )

    mlp_model.fit(X_scaled, y_enc)
    return mlp_model, scaler, le, X_scaled, y


def test_mlp(data, features, model, scaler, label_encoder):
    df = data.copy()
    X_test = df[features].copy()
    X_test = X_test.fillna(X_test.mean())

    X_test_scaled = scaler.transform(X_test)
    y_pred_enc = model.predict(X_test_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    df['mlp_pred'] = y_pred
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

    svm_train_df = train_df.copy()
    svm_train_df["svm_pred"] = y_train_pred

    svm_test_df = test_svm(test_df, features, svm_model, scaler)
    y_test_true = svm_test_df['playstyle_label']
    y_test_pred = svm_test_df['svm_pred']
    evaluate_model("SVM Test", y_test_true, y_test_pred, label_order)

    return svm_model, scaler, svm_train_df, svm_test_df

def run_catboost(train_df, test_df, features, label_order):
    catboost_model, X_train, y_train = train_catboost(train_df, features)
    y_train_pred = np.array(catboost_model.predict(X_train)).ravel()
    evaluate_model("Catboost Train", y_train, y_train_pred, label_order)

    catboost_train_df = train_df.copy()
    catboost_train_df["catboost_pred"] = y_train_pred

    catboost_test_df = test_catboost(test_df, features, catboost_model)
    y_test_true = catboost_test_df['playstyle_label']
    y_test_pred = catboost_test_df['catboost_pred']
    evaluate_model("Catboost Test", y_test_true, y_test_pred, label_order)

    return catboost_model, catboost_train_df, catboost_test_df

def run_mlp(train_df, test_df, features, label_order):
    mlp_model, scaler, le, X_scaled, y_train = train_mlp(train_df, features)
    y_train_pred_enc = mlp_model.predict(X_scaled)
    y_train_pred = le.inverse_transform(y_train_pred_enc)
    evaluate_model("MLP Train", y_train, y_train_pred, label_order)

    mlp_train_df = train_df.copy()
    mlp_train_df["mlp_pred"] = y_train_pred

    mlp_test_df = test_mlp(test_df, features, mlp_model, scaler, le)
    y_test_true = mlp_test_df['playstyle_label']
    y_test_pred = mlp_test_df['mlp_pred']
    evaluate_model("MLP Test", y_test_true, y_test_pred, label_order)

    return mlp_model, scaler, mlp_train_df, mlp_test_df

def add_exponential_weights(df, year_col="Season_Year", alpha=0.9):
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    max_year = df[year_col].max()

    df["exp_weight"] = alpha ** (max_year - df[year_col])
    return df

def compute_overall_playstyle(df, pred_col, weight_col):
    # computes playstyle for single df (train or test)
    # pred_col is the predicted labels by model (e.g. svm_pred)
    # weight_col is the type of weighting used for final classifiation (exponential, linear, bucket)

    df = df.copy()
    df = df.dropna(subset=["Team_Acronym", pred_col, weight_col])

    # Weighted one-hot for each class
    df["w_three"] = df[weight_col] * (df[pred_col] == "three_point_focused").astype(float)
    df["w_bal"]   = df[weight_col] * (df[pred_col] == "balanced").astype(float)
    df["w_paint"] = df[weight_col] * (df[pred_col] == "paint_focused").astype(float)

    grouped = df.groupby("Team_Acronym")[["w_three", "w_bal", "w_paint"]].sum()

    weight_sums = grouped.copy()
    weight_sums.columns = [
        f"weight_sum_{weight_col}_three_point_focused",
        f"weight_sum_{weight_col}_balanced",
        f"weight_sum_{weight_col}_paint_focused",
    ]

    label_map = {
        "w_three": "three_point_focused",
        "w_bal":   "balanced",
        "w_paint": "paint_focused",
    }
    max_cols = grouped.idxmax(axis=1)
    overall_labels = max_cols.map(label_map)
    overall_labels.name = f"overall_playstyle_{weight_col}"

    result = pd.concat([weight_sums, overall_labels], axis=1).reset_index()
    return result

def compute_model_overall_playstyles(train_pred_df, test_pred_df, pred_col, alpha=0.9, year_col="Season_Year"):
    # concats training and test predictions
    # adds weights and computes overall playstyle
    all_df = pd.concat(
        [
            train_pred_df[["Team_Acronym", year_col, pred_col]],
            test_pred_df[["Team_Acronym", year_col, pred_col]],
        ],
        ignore_index=True,
    )

    all_df = add_exponential_weights(all_df, year_col=year_col, alpha=alpha)

    base = compute_overall_playstyle(all_df, pred_col, weight_col="exp_weight")

    #extract prefix and append to column names
    prefix = pred_col.replace("_pred", "")
    rename_map = {
        c: c if c == "Team_Acronym" else f"{prefix}_{c}"
        for c in base.columns
    }
    base = base.rename(columns=rename_map)
    return base

def main():
    features = [
        '3PAr',
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

    svm_model, svm_scaler, svm_train_df, svm_test_df = run_svm(train_df, test_df, percentile_features, labels_order)

    cb_model, cb_train_df, cb_test_df = run_catboost(train_df, test_df, percentile_features, labels_order)

    mlp_model, mlp_scaler, mlp_train_df, mlp_test_df = run_mlp(train_df, test_df, percentile_features, labels_order)

    # Weighted class/playstyle for each model (exponential)
    svm_overall = compute_model_overall_playstyles(svm_train_df, svm_test_df, pred_col="svm_pred", alpha=0.9)

    cb_overall = compute_model_overall_playstyles(cb_train_df, cb_test_df, pred_col="catboost_pred", alpha=0.9)

    mlp_overall = compute_model_overall_playstyles(mlp_train_df, mlp_test_df, pred_col="mlp_pred", alpha=0.9)

    print("\n=== OVERALL PLAYSTYLES (SVM) ===")
    print(svm_overall[["Team_Acronym", "svm_overall_playstyle_exp_weight"]].to_string(index=False))

    print("\n=== OVERALL PLAYSTYLES (CatBoost) ===")
    print(cb_overall[["Team_Acronym", "catboost_overall_playstyle_exp_weight"]].to_string(index=False))

    print("\n=== OVERALL PLAYSTYLES (MLP) ===")
    print(mlp_overall[["Team_Acronym", "mlp_overall_playstyle_exp_weight"]].to_string(index=False))

    # merge overall classifications by team
    final_df = pd.DataFrame({
        "Team_Acronym": svm_overall["Team_Acronym"],
        "svm_playstyle": svm_overall["svm_overall_playstyle_exp_weight"],
        "catboost_playstyle": cb_overall["catboost_overall_playstyle_exp_weight"],
        "mlp_playstyle": mlp_overall["mlp_overall_playstyle_exp_weight"],
    })

    final_df.to_csv("overall_team_playstyles_by_model.csv", index=False)
    print("\nSaved overall_team_playstyles_by_model.csv")
    
if __name__ == '__main__':
    main()