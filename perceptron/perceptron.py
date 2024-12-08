import numpy as np
import matplotlib.pyplot as plt


# Perceptron 
class Perceptron:
    def __init__(self, input_dim, taxa_de_aprendizado=0.1, epoca=10):
        self.weights = np.random.randn(input_dim + 1)  # Somando ao Bias
        self.taxa_de_aprendizado = taxa_de_aprendizado
        self.epoca = epoca
        self.errors = []


    def sigmoid(self, z):
        """Calcula a função sigmoidal."""
        return 1 / (1 + np.exp(-z))
    
    def predict(self, x):
        """Predição com ativação sigmoidal."""
        weighted_sum = np.dot(x, self.weights)  # Soma ponderada
        return self.sigmoid(weighted_sum)

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights)
        return np.where(weighted_sum >= 0, 1, 0)

    def fit(self, X, y):
        X = np.c_[X, np.ones((X.shape[0]))]  # Somando ao Bias
        for epoch in range(self.epoca):
            total_error = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                total_error += abs(error)
                self.weights += self.taxa_de_aprendizado * error * xi
            self.errors.append(total_error)
            self.plot_decisao_limite(X, y, epoch)

    def plot_decisao_limite(self, X, y, epoch):
        plt.figure()
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="Class 0")
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Class 1")

        x_vals = np.linspace(-0.5, 1.5, 100)
        if len(self.weights) == 3:  # 2D case + bias
            w1, w2, b = self.weights
            y_vals = -(w1 * x_vals + b) / w2
            plt.plot(x_vals, y_vals, color="black", label="Decisão")

        plt.title(f"Decisão por Época {epoch + 1}")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.show()


# Function AND/OR/XOR Test
def gerador_func_log(logic_fn, user_input):
    """Gerador de funções logicas"""
    from itertools import product

    X = np.array(list(product([0, 1], repeat=user_input)))
    if logic_fn == "AND":
        y = np.all(X, axis=1).astype(int)
    elif logic_fn == "OR":
        y = np.any(X, axis=1).astype(int)
    elif logic_fn == "XOR":
        y = np.logical_xor.reduce(X, axis=1).astype(int)
    
    return X, y


# Main Program
if __name__ == "__main__":
    
    user_input = int(input("Digite o número de entradas: "))
    logic_fn = "OR"  # Altere aqui a porta logica que vc deseja <-
    taxa_de_aprendizado = 0.1
    epoca = 10

    X, y = gerador_func_log(logic_fn, user_input)

    perceptron = Perceptron(input_dim=user_input, taxa_de_aprendizado=taxa_de_aprendizado, epoca=epoca)
    perceptron.fit(X, y)

    plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Total Errors")
    plt.title(f"Training Error Trend ({logic_fn})")
    plt.show()
