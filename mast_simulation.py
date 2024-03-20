import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Mast Simulation')

# Users parameters
days_to_simulate = st.number_input('Days to Simulate', min_value=1, max_value=365*50, value=365*30, step=1)
n_iteration = st.number_input('Number of Iterations', min_value=1, max_value=10000, value=1, step=1)
daily_probability_to_mast_on_taking_money = st.number_input('Average Daily Mast on Taking Money', min_value=0.0, max_value=5.0, value=0.01, step=0.01, format="%.2f")
daily_mast_not_taking_money = st.number_input('Average Daily Mast Not Taking Money', min_value=0, max_value=10, value=2, step=1, format="%.2f")
yearly_return_on_investment = st.slider('Yearly Return on Investment', min_value=0.0, max_value=0.20, value=0.05, step=0.01)
invest_the_money = st.checkbox('Invest the Money', value=True)

# Initialize arrays to store results
all_taking_money_history = np.zeros((n_iteration, days_to_simulate))
all_not_taking_money_history = np.zeros((n_iteration, days_to_simulate))

# Constants
TAKE_THE_MONEY_AMOUNT = 1000000
EVERY_MAST_LOSS = 10000
EVERY_MAST_WIN = 100

# Daily interest calculation
daily_interest_rate = (1 + yearly_return_on_investment) ** (1 / 365) - 1

# Simulation iterations loop
for iteration in range(n_iteration):
    money_taking_balance = TAKE_THE_MONEY_AMOUNT
    not_taking_money_balance = 0

    # Single simulation loop
    for day in range(days_to_simulate):
        # When taking the money
        if np.random.rand() < daily_probability_to_mast_on_taking_money:
            money_taking_balance -= EVERY_MAST_LOSS
        if invest_the_money:
            money_taking_balance += money_taking_balance * daily_interest_rate
        all_taking_money_history[iteration, day] = money_taking_balance

        # When not taking the money
        not_taking_money_balance += int(daily_mast_not_taking_money) * EVERY_MAST_WIN
        decimal_part = daily_mast_not_taking_money - int(daily_mast_not_taking_money)
        if np.random.rand() < decimal_part:
            not_taking_money_balance += EVERY_MAST_WIN

avg_taking_money_history = np.mean(all_taking_money_history, axis=0)
avg_not_taking_money_history = np.mean(all_not_taking_money_history, axis=0)

# Plotting
plt.figure(figsize=(10, 6))
if days_to_simulate > 365 * 2:
    years = np.arange(days_to_simulate) / 365
    plt.plot(years, avg_taking_money_history, label='Average Taking the Money')
    plt.plot(years, avg_not_taking_money_history, label='Average Not Taking the Money')
    plt.xlabel('Years')
else:
    plt.plot(avg_taking_money_history, label='Average Taking the Money')
    plt.plot(avg_not_taking_money_history, label='Average Not Taking the Money')
    plt.xlabel('Days')

plt.ylabel('Average Balance')
plt.title('Average Balance Over Time')
plt.legend()
st.pyplot(plt)
