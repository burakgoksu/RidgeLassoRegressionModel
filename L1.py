import numpy as np
import matplotlib.pyplot as plt

# Lasso regresyonu için gradient descent fonksiyonu
def lasso_gradient_descent(X, y, alpha, learning_rate, threshold, max_iterations=10000000):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]  # Bias terimini ekleyelim
    theta = np.random.randn(8, 1)  # Başlangıçta rastgele ağırlıklar
    prev_cost = float('inf')  # Başlangıçta maliyeti sonsuz olarak ayarlayalım

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
lasso_alpha_values = np.logspace(-10, 0, 10)
print(lasso_alpha_values)
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

print(f'Test Seti Mean Squared Error: {test_cost_lasso}')
print(f"En düşük cost: {lasso_best_cost} (alpha={lasso_best_alpha})")


# Manually enter a new data point for prediction
new_data_point = np.array([1.33, 1.73, 2.50,0.19, 1.12, 1.08, 4.57])

# Normalize the new data point using the same normalization function
new_data_point_normalized = z_score_normalize(new_data_point)

# Add bias term
new_data_point_b = np.concatenate([[1], new_data_point_normalized])

# Use the trained theta for prediction
manual_prediction = new_data_point_b.dot(theta_test_lasso)

# Denormalize the prediction
manual_prediction_denormalized = manual_prediction * np.std(y_train) + np.mean(y_train)

# Print the manual prediction
print("\nManual Prediction:")
print(f"Predicted value: {manual_prediction_denormalized}")


# Test seti üzerinde tahmin yapma
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]  # Bias terimini ekleyelim
predictions = X_test_b.dot(theta_test_lasso)

# Normalleştirmeyi geri alma
predictions_denormalized = predictions * np.std(y_test) + np.mean(y_test)
y_test_denormalized = y_test * np.std(y_test) + np.mean(y_test)

# Test seti üzerinde tahmin ile gerçek değerleri karşılaştırma
plt.scatter(y_test_denormalized, predictions_denormalized)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahminler")
plt.title("Ridge Regresyonu - Gerçek Değerler vs Tahminler")

# Ridge regresyonu tarafından öğrenilen doğruyu çizme
plt.plot([min(y_test_denormalized), max(y_test_denormalized)], [min(y_test_denormalized), max(y_test_denormalized)], color='red', linestyle='-', linewidth=2, label='Ridge Regresyonu Doğrusu')

plt.legend()
plt.show()

