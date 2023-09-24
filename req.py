import requests
from random import randint, gauss
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Função para obter avaliações da API com base em um usuário, probabilidade de crossover e probabilidade de mutação
def get_evaluations(user, p_crossover, p_mutation):
    # Define a URL da API
    url = "http://127.0.0.1:8000/api/recommender"

    # Define os parâmetros como um dicionário
    parameters = {
        "query_search": user,
        "individual_size": 10,
        "population_size": 50,
        "p_crossover": p_crossover,
        "p_mutation": p_mutation,
        "max_generations": 30,
        "size_hall_of_fame": 3,
        "seed": 42
    }

    # Envia uma solicitação POST com os parâmetros
    response = requests.post(url, json=parameters)

    # Verifica a resposta
    if response.status_code == 200:
        # A solicitação foi bem-sucedida
        response = response.json()["statisticsData"]#Pega os parametros do usuário
    evaluation = response[-1]["avg"]

    return evaluation

# Função para gerar uma lista de usuários aleatórios
def get_user_array():
    # Define o intervalo (inclusive)
    start_range = 1  # Substitua pelo valor de início desejado
    end_range = 611  # Substitua pelo valor final desejado
    pop_size = 3

    user_list = []

    # Gera e armazena valores aleatórios
    for _ in range(pop_size):
        random_value = randint(start_range, end_range)
        user_list.append(random_value)

    return user_list

# Função para calcular a média de uma lista de avaliações
def get_average_fitness(evaluations):
    return sum(evaluations) / len(evaluations)

# Função para obter o resultado em lote para uma dada probabilidade de crossover e probabilidade de mutação
def get_batch_result(p_crossover, p_mutation):
    user_array = get_user_array()
    evaluations = []
    for user in user_array:
        evaluation = get_evaluations(user, p_crossover, p_mutation)
        evaluations.append(evaluation)

    return get_average_fitness(evaluations), (p_crossover, p_mutation), user_array

# Inicializa uma lista vazia para armazenar os dados
data = []

# Função para atualizar o gráfico
def update_plot():
    plt.clf()  # Limpa o gráfico anterior
    plt.plot(data)
    plt.xlabel('Iteração')
    plt.ylabel('Fitness Média')
    plt.title('Gráfico Atualizado em Tempo Real')
    plt.pause(0.1)  # Pausa para permitir a atualização do gráfico

# Função para obter o resultado da época
def get_epoch_result():
    def objective_function(args):
        x, y = args
        value = get_batch_result(x, y)[0]

        print(value, x, y)

        data.append(value)
        update_plot()

        return -value

    param_bounds = [(0, 100), (0, 100)]

    result = differential_evolution(objective_function, bounds=param_bounds)

    return result

# Chama a função get_epoch_result para otimização
get_epoch_result()

plt.show()  # Mantém a janela do gráfico aberta
