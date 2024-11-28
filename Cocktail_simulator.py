import streamlit as st
import numpy as np

def get_regular_config():
    # Weight, Multiplier
    config = [
        (250, 60),
        (250, 80),
        (320, 100),
        (160, 200),
        (30, 500),
        (5, 1500)
    ]
    return config


def get_probabilities(weights_list):
    weights_sum = sum(weights_list)
    prob_list = [weight / weights_sum for weight in weights_list]
    return prob_list


def run_simulation(iteration_num, cocktail_num, win_all_prob, extra_drink_prob_list):
    weights, multipliers = zip(*get_regular_config())
    iterations_result_list = []

    for iteration in range(iteration_num):
        iteration_weights = weights
        iteration_multipliers = multipliers
        iteration_multiplier = 0
        iteration_cocktail_num = cocktail_num

        if np.random.rand() < win_all_prob:
            iteration_multiplier = sum(iteration_multipliers)
        else:
            for i in range(len(extra_drink_prob_list)):
                if np.random.rand() < extra_drink_prob_list[i]:
                    iteration_cocktail_num += 1

            for cocktail in range(iteration_cocktail_num):
                iteration_probabilities = get_probabilities(iteration_weights)
                chosen_index = np.random.choice(len(iteration_probabilities), p=iteration_probabilities)
                iteration_multiplier += iteration_multipliers[chosen_index]
                iteration_weights = iteration_weights[:chosen_index] + iteration_weights[chosen_index + 1:]
                iteration_multipliers = iteration_multipliers[:chosen_index] + iteration_multipliers[chosen_index + 1:]

        iterations_result_list.append(iteration_multiplier)

    return np.mean(iterations_result_list), np.median(iterations_result_list)


st.title("Cocktail Simulation")
st.sidebar.header("Simulation Parameters")

iteration_num = st.sidebar.number_input("Number of Iterations", min_value=1000, max_value=1000000, value=100000, step=1000)
cocktail_num = st.sidebar.number_input("Number of Cocktails", min_value=1, max_value=10, value=5, step=1)
win_all_prob = st.sidebar.slider("Probability of Winning All", min_value=0.0, max_value=1.0, value=0.015, step=0.001)

extra_drink_prob_list = []
for i in range(5):
    extra_drink_prob = st.sidebar.slider(f"Probability of Extra Drink {i+1}", min_value=0.0, max_value=1.0, value=0.01, step=0.001)
    extra_drink_prob_list.append(extra_drink_prob)

st.sidebar.header("Configuration (Weight, Multiplier)")
config = []
for i in range(6):
    weight = st.sidebar.number_input(f"Weight {i+1}", min_value=1, max_value=1000, value=get_regular_config()[i][0], step=1)
    multiplier = st.sidebar.number_input(f"Multiplier {i+1}", min_value=1, max_value=5000, value=get_regular_config()[i][1], step=1)
    config.append((weight, multiplier))

if st.button("Run Simulation"):
    avg_multiplier, median_multiplier = run_simulation(iteration_num, cocktail_num, win_all_prob, extra_drink_prob_list, config)
    st.write(f'**Average Multiplier:** {avg_multiplier}')
    st.write(f'**Median Multiplier:** {median_multiplier}')
