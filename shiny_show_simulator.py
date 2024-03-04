import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to run the simulation based on the floors configuration
def run_simulation(floors_config, cycles):
    # Initialize lists to store results
    cycle_floor_reached_list = []
    cycle_gems_reward_list = []

    # Run the simulation for the given number of cycles
    for _ in range(cycles):
        floor_index = 0
        gems_rewarded = 0

        while floor_index < len(floors_config):
            floor_index += 1
            floor = floors_config[floor_index - 1]
            choices = [i for i in range(len(floor))]
            weights = [config[0] for config in floor]
            choice = np.random.choice(choices, p=np.array(weights) / sum(weights))

            # Update gems rewarded based on the choice
            gems_rewarded += floor[choice][1]

            # Stop the simulation if a bomb is encountered
            if floor[choice][2] == 1:
                break

        cycle_floor_reached_list.append(floor_index)
        cycle_gems_reward_list.append(gems_rewarded)

    return cycle_floor_reached_list, cycle_gems_reward_list


# Streamlit app
st.title("Floor Simulation")

# User inputs
cycles = st.sidebar.number_input("Number of cycles", min_value=1, value=1000)
floors_config = []

for i in range(15):  # Assuming 15 floors
    st.sidebar.header(f"Floor {i + 1}")
    weight1 = st.sidebar.number_input(f"Weight 1 for Floor {i + 1}", value=20, min_value=0)
    reward1 = st.sidebar.number_input(f"Reward 1 for Floor {i + 1}", value=0, min_value=0)
    bomb1 = st.sidebar.checkbox(f"Bomb 1 for Floor {i + 1}", value=False)

    # Add more options for weights, rewards, and bombs as per your configuration
    # This is just for one option; you can add more as per your original config

    floors_config.append([(weight1, reward1, 1 if bomb1 else 0)])

# Run simulation
if st.button("Run Simulation"):
    floor_reached, gems_rewarded = run_simulation(floors_config, cycles)

    # Data processing for visualization
    df = pd.DataFrame({
        "Floor Reached": floor_reached,
        "Gems Rewarded": gems_rewarded
    })

    avg_gems_per_floor = df.groupby("Floor Reached")["Gems Rewarded"].mean().reset_index()

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(avg_gems_per_floor["Floor Reached"], avg_gems_per_floor["Gems Rewarded"])
    ax.set_xlabel("Floor Number")
    ax.set_ylabel("Average Gems Rewarded")
    ax.set_title("Average Gems Rewarded Per Floor")
    st.pyplot(fig)
