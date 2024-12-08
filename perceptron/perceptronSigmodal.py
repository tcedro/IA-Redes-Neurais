import numpy as np
import matplotlib.pyplot as plt

# Perceptron com Sigmoidal
class PerceptronSigmoid:
    def __init__(self, input_dim, taxa_de_aprendizado=0.1, epoca=10):
        self.weights = np.random.randn(input_dim + 1)  # Inclui o termo de bias
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

    def fit(self, X, y):
        """Treina o modelo usando a sigmoidal."""
        X = np.c_[X, np.ones((X.shape[0]))]  # Adiciona o termo de bias
        for epoch in range(self.epoca):
            total_error = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                gradient = error * prediction * (1 - prediction)  # Derivada da sigmoidal
                self.weights += self.taxa_de_aprendizado * gradient * xi
                total_error += abs(error)
            self.errors.append(total_error)
            self.plt_limit(X, y, epoch)

    def plt_limit(self, X, y, epoch):
        """Plota o hiperplano de decisão ao longo do treinamento."""
        plt.figure()
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="Class 0")
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Class 1")

        # Hiperplano de decisão
        x_vals = np.linspace(-0.5, 1.5, 100)
        if len(self.weights) == 3:  # Caso 2D + bias
            w1, w2, b = self.weights
            y_vals = -(w1 * x_vals + b) / w2
            plt.plot(x_vals, y_vals, color="black", label="Decisão Limite")

        plt.title(f"Decisão por Época {epoch + 1}")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.show()


# Função para gerar os dados (AND, OR, XOR)
def gerador_func_log(logic_fn, user_input):
    """Gera os dados booleanos para a função lógica escolhida."""
    from itertools import product

    X = np.array(list(product([0, 1], repeat=user_input)))
    if logic_fn == "AND":
        y = np.all(X, axis=1).astype(int)
    elif logic_fn == "OR":
        y = np.any(X, axis=1).astype(int)
    elif logic_fn == "XOR":
        y = np.logical_xor.reduce(X, axis=1).astype(int)

    return X, y


# Função Principal (main)
if __name__ == "__main__":
    # Entrada do usuário
    user_input = int(input("Digite o número de entradas: "))
    logic_fn = "AND"  # Altere aqui a porta logica que vc deseja <-
    taxa_de_aprendizado = 0.25
    epoca = 10
    # Geração dos dados
    X, y = gerador_func_log(logic_fn, user_input)

    # Treinamento do Perceptron com sigmoidal
    perceptron = PerceptronSigmoid(input_dim=user_input, taxa_de_aprendizado=taxa_de_aprendizado, epoca=epoca)
    perceptron.fit(X, y)

    # Plot do erro ao longo das épocas
    plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker="o")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Total")
    plt.title(f"Erro ao longo do Treinamento ({logic_fn})")
    plt.show()
