import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Sidebar for input parameters
st.header('Reward weights should replace only the last pick in each board!')
cycles_count = st.sidebar.number_input('Number of Cycles', min_value=1, value=10000)
board_1_reward = st.sidebar.checkbox('Board 1 Reward', value=False)
board_2_reward = st.sidebar.checkbox('Board 2 Reward', value=True)
board_3_reward = st.sidebar.checkbox('Board 3 Reward', value=False)
board_4_reward = st.sidebar.checkbox('Board 4 Reward', value=False)
board_5_reward = st.sidebar.checkbox('Board 5 Reward', value=False)
board_1_reward_weight = st.sidebar.number_input('Board 1 Reward Weight', min_value=1, max_value=99999, value=1500)
board_2_reward_weight = st.sidebar.number_input('Board 2 Reward Weight', min_value=1, max_value=99999, value=6000)
board_3_reward_weight = st.sidebar.number_input('Board 3 Reward Weight', min_value=1, max_value=99999, value=6000)
board_4_reward_weight = st.sidebar.number_input('Board 4 Reward Weight', min_value=1, max_value=99999, value=6000)
board_5_reward_weight = st.sidebar.number_input('Board 5 Reward Weight', min_value=1, max_value=99999, value=6000)

# List to collect data for DataFrame
data_for_df = []

board_rewards_bool_list = [board_1_reward, board_2_reward, board_3_reward, board_4_reward, board_5_reward]

weights_and_rewards = []

# Board 1
weights_and_rewards.append(
    [25000,
     2000,
     12000,
     2000,
     1000,
     2000,
     board_1_reward_weight,
     12000])

# Board 2
weights_and_rewards.append(
    [11000,
     11000,
     6000,
     6000,
     6000,
     6000,
     6000,
     6000,
     6000,
     6000,
     1000,
     1000,
     6000,
     board_2_reward_weight,
     6000])

# Board 3
weights_and_rewards.append([
    11000,
    11000,
    11000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    1000,
    1000,
    1000,
    11000,
    6000,
    6000,
    board_3_reward_weight,
    2000])

# Board 4
weights_and_rewards.append([
    11000,
    11000,
    11000,
    11000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    1000,
    1000,
    1000,
    1000,
    6000,
    6000,
    6000,
    board_4_reward_weight,
    2500])

# Board 5
weights_and_rewards.append([
    11000,
    11000,
    11000,
    11000,
    11000,
    20000,
    20000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    6000,
    1000,
    1000,
    1000,
    1500,
    1000,
    6000,
    2000,
    6000,
    6000,
    6000,
    board_5_reward_weight,
    3000])


def start_simulation():
    for cycle in range(0, cycles_count):
        default_cycle_weights = weights_and_rewards.copy()
        picks_used = 0
        rewards = 0

        # Each board
        for board_num in range(0, len(default_cycle_weights)):
            board_weights = default_cycle_weights[board_num]
            found_blast = False
            found_reward = False

            # If the user didn't find the blast
            while not found_blast:
                # Adding a pick that was used
                picks_used += 1

                weights_sum = sum(board_weights)
                picks_probabilities_array = np.array([weight / weights_sum for weight in board_weights])
                chosen_pick = np.random.choice(len(picks_probabilities_array), p=picks_probabilities_array)

                # The user found the blast
                if chosen_pick == len(picks_probabilities_array) - 1:
                    found_blast = True

                # The user found a reward
                elif chosen_pick == len(picks_probabilities_array) - 2 and board_rewards_bool_list[board_num] and not found_reward:
                    rewards += 1
                    found_reward = True

                # The user didn't find the blast
                else:
                    # Removing the weights that was found
                    board_weights = board_weights[:chosen_pick] + board_weights[chosen_pick + 1:]

        # Collect data for DataFrame
        data_for_df.append({
            "user_num": cycle,
            "picks_used": picks_used,
            "rewards": rewards,
        })

    # Convert list of dictionaries to DataFrame and return it
    return pd.DataFrame(data_for_df)


def plot_results(df):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Adjust for vertical layout

    # Plot 1: Distribution of Picks Used to Complete Cycles
    axs[0].hist(df['picks_used'], bins=range(min(df['picks_used']), max(df['picks_used']) + 1, 1), color='skyblue', edgecolor='black', alpha=0.7, density=True, weights=np.ones(len(df['picks_used'])) / len(df['picks_used']))
    axs[0].set_title('Distribution of Picks Used to Complete Cycles')
    axs[0].set_xlabel('Picks Used')
    axs[0].set_ylabel('Percentage of Cycles')
    axs[0].set_xticks(range(min(df['picks_used']), max(df['picks_used']) + 1, 5))
    axs[0].grid(axis='y', alpha=0.75)

    # Plot 2: Distribution of Rewards per Cycle
    axs[1].hist(df['rewards'], bins=range(min(df['rewards']), max(df['rewards']) + 1, 1), color='orange', edgecolor='black', alpha=0.7, density=True, weights=np.ones(len(df['rewards'])) / len(df['rewards']))
    axs[1].set_title('Distribution of Rewards per Cycle')
    axs[1].set_xlabel('Rewards')
    axs[1].set_ylabel('Percentage of Cycles')
    axs[1].set_xticks(range(min(df['rewards']), max(df['rewards']) + 1, 1))
    axs[1].grid(axis='y', alpha=0.75)

    # Adjusting the y-axis to show percentages
    for ax in axs:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Display the plots
    st.pyplot(fig)

    # Calculating and printing averages and medians
    avg_picks_used = df['picks_used'].mean()
    median_picks_used = df['picks_used'].median()
    avg_rewards = df['rewards'].mean()
    median_rewards = df['rewards'].median()

    st.write(f"Average Picks Used: {avg_picks_used:.2f}")
    st.write(f"Median Picks Used: {median_picks_used}")
    st.write(f"Average Rewards: {avg_rewards:.2f}")
    st.write(f"Median Rewards: {median_rewards}")


if st.button('Simulate'):
    results_df = start_simulation()
    plot_results(results_df)
