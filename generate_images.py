#!/usr/bin/env python3
"""
Generate all visualizations for the Heart Disease Prediction project.
Saves images to the images/ folder for README documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Set style
plt.style.use('default')
np.random.seed(42)

print("=" * 60)
print("📸 GENERATING VISUALIZATIONS FOR README")
print("=" * 60)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("\n📂 Loading dataset...")
df = pd.read_csv('heart_disease_prediction.csv')
df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
print(f"   ✅ Loaded {len(df)} records")

# =============================================================================
# 2. CLASS DISTRIBUTION
# =============================================================================
print("\n📊 Generating class distribution plot...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

class_counts = df['Heart Disease'].value_counts()
colors = ['#2ecc71', '#e74c3c']
labels = ['Absence (0)', 'Presence (1)']

axes[0].bar(labels, [class_counts[0], class_counts[1]], color=colors, edgecolor='black')
axes[0].set_xlabel('Heart Disease Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate([class_counts[0], class_counts[1]]):
    axes[0].text(i, v + 2, str(v), ha='center', fontsize=12, fontweight='bold')

axes[1].pie([class_counts[0], class_counts[1]], labels=labels, colors=colors, 
            autopct='%1.1f%%', startangle=90, explode=(0.02, 0.02),
            textprops={'fontsize': 11})
axes[1].set_title('Class Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/class_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/class_distribution.png")

# =============================================================================
# 3. FEATURE DISTRIBUTIONS BY CLASS
# =============================================================================
print("\n📊 Generating feature distributions plot...")

features_to_plot = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 'Number of vessels fluro']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(features_to_plot):
    absence = df[df['Heart Disease'] == 0][feature]
    presence = df[df['Heart Disease'] == 1][feature]
    
    axes[idx].hist(absence, bins=20, alpha=0.7, label='Absence (0)', color='#2ecc71', edgecolor='black')
    axes[idx].hist(presence, bins=20, alpha=0.7, label='Presence (1)', color='#e74c3c', edgecolor='black')
    axes[idx].set_xlabel(feature, fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'{feature} by Class', fontsize=12, fontweight='bold')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('images/feature_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/feature_distributions.png")

# =============================================================================
# 4. CORRELATION MATRIX
# =============================================================================
print("\n📊 Generating correlation matrix...")

correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
im = plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right', fontsize=9)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, fontsize=9)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# Add correlation values
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        val = correlation_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.tight_layout()
plt.savefig('images/correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/correlation_matrix.png")

# =============================================================================
# 5. PREPARE DATA FOR MODEL
# =============================================================================
print("\n🔧 Preparing data for model training...")

# Feature selection
feature_columns = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
                   'Max HR', 'ST depression', 'Number of vessels fluro']

X = df[feature_columns].values
y = df['Heart Disease'].values

# Train-test split (70/30)
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.7 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Normalize features
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

print(f"   ✅ Train set: {len(X_train)} samples")
print(f"   ✅ Test set: {len(X_test)} samples")

# =============================================================================
# 6. LOGISTIC REGRESSION FUNCTIONS
# =============================================================================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    m = len(y)
    z = np.dot(X, w) + b
    h = sigmoid(z)
    h = np.clip(h, 1e-10, 1 - 1e-10)
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, w, b, alpha, iterations, lambda_reg=0):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        z = np.dot(X, w) + b
        h = sigmoid(z)
        
        dw = (1/m) * np.dot(X.T, (h - y)) + (lambda_reg/m) * w
        db = (1/m) * np.sum(h - y)
        
        w = w - alpha * dw
        b = b - alpha * db
        
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            if lambda_reg > 0:
                cost += (lambda_reg / (2*m)) * np.sum(w**2)
            cost_history.append(cost)
    
    return w, b, cost_history

# =============================================================================
# 7. TRAIN MODEL AND PLOT COST CONVERGENCE
# =============================================================================
print("\n🧠 Training logistic regression model...")

n_features = X_train_norm.shape[1]
w_init = np.zeros(n_features)
b_init = 0
alpha = 0.1
iterations = 2000

w_trained, b_trained, cost_history = gradient_descent(
    X_train_norm, y_train, w_init, b_init, alpha, iterations
)

print(f"   ✅ Final cost: {cost_history[-1]:.4f}")

# Plot cost convergence
plt.figure(figsize=(10, 6))
plt.plot(range(0, iterations, 100), cost_history, 'b-', linewidth=2, marker='o', markersize=4)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
plt.title('Cost Function Convergence During Training', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/cost_convergence.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/cost_convergence.png")

# =============================================================================
# 8. DECISION BOUNDARIES
# =============================================================================
print("\n📊 Generating decision boundary plots...")

def plot_decision_boundary(X, y, feature_names, w_2d, b_2d, title, filename):
    """Plot decision boundary for 2D features."""
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict for mesh
    Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], w_2d) + b_2d)
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, levels=50, cmap='RdYlGn_r', alpha=0.6)
    plt.colorbar(label='P(Heart Disease)')
    
    # Plot decision boundary line
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Plot data points
    colors = ['#2ecc71', '#e74c3c']
    for cls in [0, 1]:
        mask = y == cls
        label = 'Absence' if cls == 0 else 'Presence'
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[cls], label=label, 
                   edgecolors='black', s=60, alpha=0.8)
    
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'images/{filename}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# Feature pairs for decision boundaries
feature_pairs = [
    (0, 4, 'Age', 'Cholesterol'),  # Age vs Cholesterol
    (3, 5, 'BP', 'Max HR'),        # BP vs Max HR
    (6, 7, 'ST depression', 'Number of vessels fluro')  # ST depression vs Vessels
]

for f1_idx, f2_idx, f1_name, f2_name in feature_pairs:
    # Get 2D features (normalized)
    X_2d = X_train_norm[:, [f1_idx, f2_idx]]
    
    # Train 2D model
    w_2d = np.zeros(2)
    b_2d = 0
    w_2d, b_2d, _ = gradient_descent(X_2d, y_train, w_2d, b_2d, 0.1, 2000)
    
    # Plot
    filename = f"decision_boundary_{f1_name.lower().replace(' ', '_')}_{f2_name.lower().replace(' ', '_')}.png"
    plot_decision_boundary(X_2d, y_train, [f1_name, f2_name], w_2d, b_2d, 
                          f'Decision Boundary: {f1_name} vs {f2_name}', filename)
    print(f"   ✅ Saved: images/{filename}")

# =============================================================================
# 9. REGULARIZATION COMPARISON
# =============================================================================
print("\n📊 Generating regularization comparison...")

# Train models with different lambda values
lambda_values = [0, 0.01, 0.1, 1.0]
colors = ['blue', 'green', 'orange', 'red']

# Use Age vs Cholesterol for visualization
f1_idx, f2_idx = 0, 4
X_2d = X_train_norm[:, [f1_idx, f2_idx]]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (lam, color) in enumerate(zip(lambda_values, colors)):
    w_2d = np.zeros(2)
    b_2d = 0
    w_2d, b_2d, cost_hist = gradient_descent(X_2d, y_train, w_2d, b_2d, 0.1, 2000, lambda_reg=lam)
    
    ax = axes[idx]
    
    # Create mesh grid
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], w_2d) + b_2d)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=50, cmap='RdYlGn_r', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    for cls in [0, 1]:
        mask = y_train == cls
        label = 'Absence' if cls == 0 else 'Presence'
        c = '#2ecc71' if cls == 0 else '#e74c3c'
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=c, label=label, 
                  edgecolors='black', s=40, alpha=0.7)
    
    ax.set_xlabel('Age (normalized)', fontsize=11)
    ax.set_ylabel('Cholesterol (normalized)', fontsize=11)
    ax.set_title(f'λ = {lam} | ||w|| = {np.linalg.norm(w_2d):.3f}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)

plt.suptitle('L2 Regularization Effect on Decision Boundary', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/regularization_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/regularization_comparison.png")

# =============================================================================
# 10. METRICS EVALUATION
# =============================================================================
print("\n📊 Generating metrics comparison...")

def predict(X, w, b, threshold=0.5):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= threshold).astype(int)

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

# Predictions
y_train_pred = predict(X_train_norm, w_trained, b_trained)
y_test_pred = predict(X_test_norm, w_trained, b_trained)

train_metrics = compute_metrics(y_train, y_train_pred)
test_metrics = compute_metrics(y_test, y_test_pred)

# Plot metrics comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, train_metrics, width, label='Train Set', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, test_metrics, width, label='Test Set', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/metrics_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/metrics_comparison.png")

# =============================================================================
# 11. CONFUSION MATRIX
# =============================================================================
print("\n📊 Generating confusion matrix...")

# Test set confusion matrix
tp = np.sum((y_test_pred == 1) & (y_test == 1))
tn = np.sum((y_test_pred == 0) & (y_test == 0))
fp = np.sum((y_test_pred == 1) & (y_test == 0))
fn = np.sum((y_test_pred == 0) & (y_test == 1))

conf_matrix = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix, cmap='Blues')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted: Absence', 'Predicted: Presence'], fontsize=11)
ax.set_yticklabels(['Actual: Absence', 'Actual: Presence'], fontsize=11)

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, conf_matrix[i, j], ha="center", va="center", 
                      fontsize=20, fontweight='bold',
                      color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")

ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Count')
plt.tight_layout()
plt.savefig('images/confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/confusion_matrix.png")

# =============================================================================
# 12. WEIGHT ANALYSIS
# =============================================================================
print("\n📊 Generating feature weights plot...")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if w > 0 else '#2ecc71' for w in w_trained]
bars = ax.barh(feature_columns, w_trained, color=colors, edgecolor='black')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Weight Value', fontsize=12)
ax.set_title('Feature Weights (Trained Model)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, w in zip(bars, w_trained):
    width = bar.get_width()
    ax.annotate(f'{w:.3f}', xy=(width, bar.get_y() + bar.get_height()/2),
                xytext=(5 if w > 0 else -5, 0), textcoords="offset points",
                ha='left' if w > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/feature_weights.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: images/feature_weights.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\nImages saved in 'images/' folder:")
print("   1. class_distribution.png")
print("   2. feature_distributions.png")
print("   3. correlation_matrix.png")
print("   4. cost_convergence.png")
print("   5. decision_boundary_age_cholesterol.png")
print("   6. decision_boundary_bp_max_hr.png")
print("   7. decision_boundary_st_depression_number_of_vessels_fluro.png")
print("   8. regularization_comparison.png")
print("   9. metrics_comparison.png")
print("  10. confusion_matrix.png")
print("  11. feature_weights.png")
print("\n📝 Add these to README.md to showcase your work!")
