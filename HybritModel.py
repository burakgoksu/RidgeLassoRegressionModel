import numpy as np
import matplotlib.pyplot as plt

# Lasso regresyonu için gradient descent fonksiyonu
def lasso_gradient_descent(X, y, alpha, learning_rate, threshold, max_iterations=10000000):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]  # Bias terimini ekleyelim
    theta = np.random.randn(8, 1)  # Başlangıçta rastgele ağırlıklar
    prev_cost = float('inf')  # Başlangıçta maliyeti sonsuz olarak ayarlayalım
    cost_history = []

    iteration = 0
    while iteration < max_iterations:
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) + alpha * np.sign(theta)
        theta = theta - learning_rate * gradients

        # Maliyet kontrolü
        cost = lasso_cost_function(X, y, theta, alpha)

        if abs(prev_cost - cost) < threshold:
            print(f"Iterasyon {iteration}: Cost change ({abs(prev_cost - cost)}) below threshold ({threshold}). Durduruluyor.")
            break

        prev_cost = cost
        iteration += 1
    return theta

# Lasso regresyonu için cost function
def lasso_cost_function(X, y, theta, alpha):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]
    predictions = X_b.dot(theta)
    error = predictions - y
    cost = np.mean(np.square(error)) + alpha * np.sum(np.abs(theta[1:]))
    return cost

def z_score_normalize(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

# Veriyi yükleme
data = np.genfromtxt("Doviz_Satislari.csv", delimiter=',', skip_header=1)
print("500. Data:")
print(data[499])  # Assuming the indexing starts from 0


# Veriyi karıştırma ve train-test ayırma
np.random.seed(42)
np.random.shuffle(data)

train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]

# Veriyi normalleştir
X_train = z_score_normalize(train_data[:, [0,1,2,3,4,5,6]].astype(float))  # TP DK USD S YTL,TP DK EUR S YTL,TP DK GBP S YTL,TP DK SEK S YTL,TP DK CHF S YTL,TP DK CAD S YTL,TP DK KWD S YTL
y_train = z_score_normalize(train_data[:, 7].astype(float).reshape(-1, 1))  # TP DK SAR S YTL

X_test = z_score_normalize(test_data[:, [0,1,2,3,4,5,6]].astype(float))   # TP DK USD S YTL,TP DK EUR S YTL,TP DK GBP S YTL,TP DK SEK S YTL,TP DK CHF S YTL,TP DK CAD S YTL,TP DK KWD S YTL
y_test = z_score_normalize(test_data[:, 7].astype(float).reshape(-1, 1))   # TP DK SAR S YTL


# Lasso regresyonu modelini eğitme
lasso_alpha_values = np.logspace(-10, 0, 2) # 2 tane alpha değeri var 10 tane koyunca çok fazla uzuyor
lasso_cost_values = []

for alpha in lasso_alpha_values:
    # Her alpha değeri için ayrı bir theta değeri kullan
    theta = lasso_gradient_descent(X_train, y_train, alpha, learning_rate=0.0001, threshold=1e-9)

    # Cost değerini hesapla
    cost = lasso_cost_function(X_train, y_train, theta, alpha)
    lasso_cost_values.append(cost)

# En düşük cost değerini ve karşılık gelen alpha değerini bulma
lasso_min_cost_index = np.argmin(lasso_cost_values)
lasso_best_alpha = lasso_alpha_values[lasso_min_cost_index]
lasso_best_cost = lasso_cost_values[lasso_min_cost_index]

# Test seti için ayrı bir theta değeri kullanarak maliyeti hesapla
theta_test_lasso = lasso_gradient_descent(X_test, y_test, lasso_best_alpha, learning_rate=0.0001, threshold=1e-9)
test_cost_lasso = lasso_cost_function(X_test, y_test, theta_test_lasso, lasso_best_alpha)

print(f'Test Seti Mean Squared Error (Lasso): {test_cost_lasso}')
print(f"En düşük cost: {lasso_best_cost} (alpha={lasso_best_alpha})")


def ridge_gradient_descent(X, y, alpha, learning_rate, threshold,max_iterations=10000000):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]  # Bias terimini ekleyelim
    theta = np.random.randn(8, 1)  # Başlangıçta rastgele ağırlıklar
    prev_cost = float('inf')  # Başlangıçta maliyeti sonsuz olarak ayarlayalım

    iteration = 0
    while iteration < max_iterations:
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y) + 2 * alpha * theta
        theta = theta - learning_rate * gradients

        # Maliyet kontrolü
        cost = ridge_cost_function(X, y, theta, alpha)
        if abs(prev_cost - cost) < threshold:
            print(
                f"Iterasyon {iteration}: Cost change ({abs(prev_cost - cost)}) below threshold ({threshold}). Durduruluyor.")
            break

        prev_cost = cost
        iteration += 1

    return theta

# Ridge regresyonu için cost function
def ridge_cost_function(X, y, theta, alpha):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]
    predictions = X_b.dot(theta)
    error = predictions - y
    cost = np.mean(np.square(error)) + alpha * np.sum(np.square(theta[1:]))
    return cost


# Ridge regresyonu modelini eğitme
ridge_alpha_values = np.logspace(-10, 0, 2) # 2 tane alpha değeri var 10 tane koyunca çok fazla uzuyor
ridge_cost_values = []

for alpha in ridge_alpha_values:
    # Her alpha değeri için ayrı bir theta değeri kullan
    theta = ridge_gradient_descent(X_train, y_train, alpha, learning_rate=0.0001, threshold=1e-9)

    # Cost değerini hesapla
    cost = ridge_cost_function(X_train, y_train, theta, alpha)
    ridge_cost_values.append(cost)

# En düşük cost değerini ve karşılık gelen alpha değerini bulma
ridge_min_cost_index = np.argmin(ridge_cost_values)
ridge_best_alpha = ridge_alpha_values[ridge_min_cost_index]
ridge_best_cost = ridge_cost_values[ridge_min_cost_index]

# Test seti için ayrı bir theta değeri kullanarak maliyeti hesapla
theta_test_ridge = ridge_gradient_descent(X_test, y_test, ridge_best_alpha, learning_rate=0.0001, threshold=1e-9)
test_cost_ridge = ridge_cost_function(X_test, y_test, theta_test_ridge, ridge_best_alpha)

print(f'Test Seti Mean Squared Error (Ridge Model): {test_cost_ridge}')
print(f"En düşük cost: {ridge_best_cost} (alpha={ridge_best_alpha})")


# Lasso ve Ridge modellerinin tahminlerini birleştiren hibrit model
def hybrid_model(X, theta_lasso, theta_ridge, lasso_weight=0.5):
    predictions_lasso = X.dot(theta_lasso)
    predictions_ridge = X.dot(theta_ridge)

    # Hibrit tahmin: Ağırlıklı ortalamayı kullanabilirsiniz
    hybrid_predictions = lasso_weight * predictions_lasso + (1 - lasso_weight) * predictions_ridge

    return hybrid_predictions

def denormalize(data, mean_val, std_val):
    denormalized_data = data * std_val + mean_val
    return denormalized_data

# Hibrit modelin tahminlerini elde etme
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
hybrid_predictions = hybrid_model(X_test_b, theta_test_lasso, theta_test_ridge)

# Hibrit modelin performansını değerlendirme
hybrid_cost = lasso_cost_function(X_test, y_test, theta_test_lasso, lasso_best_alpha)  # L1 (Lasso) maliyetini kullanabilirsiniz
print(f'Test Seti Mean Squared Error (Hybrid Model): {hybrid_cost}')

y_test_denormalized = denormalize(y_test, np.mean(y_test), np.std(y_test))

# Manually enter a new data point for prediction
new_data_point = np.array([1.93, 2.56, 2.99, 0.30, 2.08, 1.87, 6.81])

# Normalize the new data point using the same normalization function
new_data_point_normalized = z_score_normalize(new_data_point)

# Add bias term
new_data_point_b = np.concatenate([[1], new_data_point_normalized])

# Use the hybrid model for prediction
manual_prediction_hybrid = hybrid_model(new_data_point_b, theta_test_lasso, theta_test_ridge)

# Denormalize the prediction
manual_prediction_denormalized_hybrid = denormalize(manual_prediction_hybrid, np.mean(y_train), np.std(y_train))

# Print the manual hybrid prediction
print("\nManual Hybrid Prediction:")
print(f"Predicted value (Hybrid): {manual_prediction_denormalized_hybrid}")


# Test seti üzerinde tahmin ile gerçek değerleri karşılaştırma
plt.scatter(y_test_denormalized, hybrid_predictions)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Hibrit Tahminler")
plt.title("Hybrid Model - Gerçek Değerler vs Tahminler")

# Lasso ve Ridge regresyonları tarafından öğrenilen doğruları çizme
plt.plot([min(y_test_denormalized), max(y_test_denormalized)], [min(y_test_denormalized), max(y_test_denormalized)], color='red', linestyle='-', linewidth=2, label='Hybrid Model Doğrusu (Lasso ve Ridge)')

plt.legend()
plt.show()
