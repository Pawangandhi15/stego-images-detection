# =============================
# 13. Classical ML Models
# =============================
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

models = {
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight=class_weights),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight=class_weights
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    results[name] = {
        "report": classification_report(y_test, y_pred, output_dict=True),
        "auc": roc_auc_score(y_test, y_prob),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    print(classification_report(y_test, y_pred))
    print("AUC:", results[name]["auc"])
