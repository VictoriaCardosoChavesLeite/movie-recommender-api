import requests
from random import randint, gauss
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def get_evaluations(user, p_crossover, p_mutation):
    # Define the URL
    url = "http://127.0.0.1:8000/api/recommender"

    # Define the parameters as a dictionary
    parameters = {
        "query_search": user,
        "individual_size": 10,
        "population_size": 30,
        "p_crossover": p_crossover,
        "p_mutation": p_mutation,
        "max_generations": 20,
        "size_hall_of_fame": 3,
        "seed": 42
    }

    # Send a POST request with the parameters
    response = requests.post(url, json=parameters)

    # Check the response
    if response.status_code == 200:
        # Request was successful
        response = response.json()["statisticsData"]
    evaluation = response[-1]["avg"]

    return evaluation


def get_user_array():
    # Define the range (inclusive)
    start_range = 1  # Replace with your desired start value
    end_range = 611  # Replace with your desired end value
    pop_size = 3

    user_list = []

    # Generate and print n random values
    for _ in range(pop_size):
        random_value = randint(start_range, end_range)
        user_list.append(random_value)

    return user_list


def get_average_fitness(evaluations):
    return sum(evaluations) / len(evaluations)


def get_batch_result(p_crossover, p_mutation):
    user_array = get_user_array()
    evaluations = []
    for user in user_array:
        evaluation = get_evaluations(user, p_crossover, p_mutation)
        evaluations.append(evaluation)

    return get_average_fitness(evaluations), (p_crossover, p_mutation)


# Initialize an empty list to store your data
data = []


# Create a function to update the plot
def update_plot():
    plt.clf()  # Clear the previous plot
    plt.plot(data)
    plt.xlabel('Iteração')
    plt.ylabel('Fitness Média')
    plt.title('Live Updating Plot')
    plt.pause(0.1)  # Pause to allow the plot to update


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


get_epoch_result()

plt.show()  # Keep the plot window open
