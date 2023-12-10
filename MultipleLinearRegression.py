import numpy as np
import matplotlib.pyplot as plt

# Veriyi yükle
data = np.genfromtxt("Doviz_Satislari.csv", delimiter=',', skip_header=1, dtype=str)

# Veriyi incele
print(data[:5])

# Fonksiyonları tanımla
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    square_error = (predictions - y) ** 2
    cost = 1 / (2 * m) * np.sum(square_error)
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations, tolerance):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors) / m
        theta = theta - learning_rate * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i > 0 and abs(cost_history[i-1] - cost) < tolerance:
            break

    return theta, cost_history

def z_score_normalize(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

# Veriyi eğitim ve test setlerine bölelim
np.random.seed(42)  # Sabit bir rastgelelik için
np.random.shuffle(data)

train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]

# Veriyi normalleştir
X_train = z_score_normalize(train_data[:, [0,1,2,3,4,5,6]].astype(float))  # TP DK USD S YTL,TP DK EUR S YTL,TP DK GBP S YTL,TP DK SEK S YTL,TP DK CHF S YTL,TP DK CAD S YTL,TP DK KWD S YTL
y_train = z_score_normalize(train_data[:, 7].astype(float).reshape(-1, 1))  # TP DK SAR S YTL

X_test = z_score_normalize(test_data[:, [0,1,2,3,4,5,6]].astype(float))   # TP DK USD S YTL,TP DK EUR S YTL,TP DK GBP S YTL,TP DK SEK S YTL,TP DK CHF S YTL,TP DK CAD S YTL,TP DK KWD S YTL
y_test = z_score_normalize(test_data[:, 7].astype(float).reshape(-1, 1))   # TP DK SAR S YTL

# Bias terimi için X'e bir sütun ekleyin
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# Başlangıç ağırlıkları rastgele seçin
theta_initial = np.random.randn(8, 1)  # 8 çünkü bias ve iki özellik var

# Hyperparameters
learning_rate = 0.0001
num_iterations = 10000000
tolerance = 1e-9

# Gradient Descent
theta, cost_history = gradient_descent(X_train_b, y_train, theta_initial, learning_rate, num_iterations, tolerance)

print("Final Theta:", theta)

# Eğitim sonrası parametre değerleri
w0, w1, w2 ,w3, w4, w5, w6= theta[0], theta[1], theta[2],theta[3], theta[4], theta[5], theta[6]

# Test setini kullanarak tahminler yap
y_pred = X_test_b.dot(theta)

# Test seti için maliyeti hesapla
test_cost = compute_cost(X_test_b, y_test, theta)
print(f'Test Seti Mean Squared Error: {test_cost}')

# Eğitim sonrası modeli görselleştir
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, label='Gerçek Veri')
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='Tahminler')
ax.set_xlabel('USD')
ax.set_ylabel('EUR')
ax.set_zlabel('SAR')
ax.legend()
plt.show()
