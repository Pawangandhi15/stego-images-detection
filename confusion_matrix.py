# =============================
# 14. Confusion Matrix Plot
# =============================
plt.figure(figsize=(12,4))

for i, (name, res) in enumerate(results.items()):
    plt.subplot(1, 2, i+1)
    sns.heatmap(res["conf_matrix"], annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# =============================
# 15. ROC Curve
# =============================
plt.figure(figsize=(7,6))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Steganalysis Detection")
plt.legend()
plt.grid(True)
plt.show()
