import numpy as np
import matplotlib.pyplot as plt

# Funções de ativação e suas derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Gerar entradas e saídas para n entradas booleanas
def gerador_func_log(logic_function, n):
    X = np.array([[int(x) for x in format(i, f'0{n}b')] for i in range(2**n)])
    
    if logic_function == "AND":
        y = np.array([[np.all(row)] for row in X])
    elif logic_function == "OR":
        y = np.array([[np.any(row)] for row in X])
    elif logic_function == "XOR":
        y = np.array([[np.sum(row) % 2] for row in X])
    
    return X, y
# Função de visualização para separação das classes
def plot_limite(X, y, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, activation_function):
    # Geração do grid para plotagem
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Cálculo da saída da rede para o grid
    hidden_layer_input = np.dot(grid, weights_input_hidden) + bias_hidden
    hidden_layer_output = activation_function(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predictions = activation_function(output_layer_input).reshape(xx.shape)

    # Plotagem do grid de decisão
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolor='k', cmap="viridis", s=100, label="Dados")
    plt.title("Separação das Classes")
    plt.xlabel("Entrada 1")
    plt.ylabel("Entrada 2")
    plt.colorbar(label="Probabilidade da classe 1")
    plt.grid()
    plt.legend()
    plt.show()

# Substituir a função train_neural_network para retornar pesos e biases
def train_neural_network(X, y, activation_function, activation_derivative, learning_rate, bias, epochs, hidden_neurons):
    np.random.seed(42)

    input_neurons = X.shape[1]
    output_neurons = 1

    # Inicialização dos pesos e bias
    weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
    weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)
    bias_hidden = np.random.rand(hidden_neurons) if bias else np.zeros(hidden_neurons)
    bias_output = np.random.rand(output_neurons) if bias else np.zeros(output_neurons)

    # Treinamento
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = activation_function(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = activation_function(output_layer_input)

        # Cálculo do erro
        error = y - predicted_output

        # Backpropagation
        d_predicted_output = error * activation_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * activation_derivative(hidden_layer_output)

        # Atualização de pesos e bias
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        if bias:
            bias_output += np.sum(d_predicted_output, axis=0) * learning_rate
            bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, predicted_output

# Configurações do experimento
logic_function = "AND" # altere aqui para mudar a porta logica <-
n = int(input("Qntd. de input? ")) #inputs
taxa_aprendizado = 0.1 # taxa de aprendizado 
bias = 's' # INCLUIR bias = s ou não incluir = n
ativacao_tipo = "sigmoid" 
epochs = 10
hidden_neurons = 4 # Qntd. de neuronios na segunda camada

# Mapeando a função de ativação
ativacao = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}
activation_function, activation_derivative = ativacao[ativacao_tipo]

# Gerar dados e treinar a rede
X, y = gerador_func_log(logic_function, n)
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, predicted_output = train_neural_network(
    X, y, activation_function, activation_derivative, taxa_aprendizado, bias, epochs, hidden_neurons
)

# # Exibir resultados
# print("Resultados após o treinamento:")
# for i in range(len(X)):
#     print(f"Entrada: {X[i]} - Saída esperada: {y[i]} - Saída do modelo: {predicted_output[i]}")

# Exibir resultados
print("Resultados após o treinamento:")
for i in range(len(X)):
    print(f"Entrada: {X[i]} - Saída esperada: {y[i]} - Saída do modelo: {predicted_output[i]}")

# Plotar a separação das classes (somente para 2 entradas)
if n == 2:
    plot_limite(X, y, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, activation_function)
else:
    print("A visualização da separação das classes só é suportada para 2 entradas.")