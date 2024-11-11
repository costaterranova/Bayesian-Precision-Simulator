import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as b
from sklearn.linear_model import LinearRegression
import app as a


uploaded_file = st.file_uploader(
    "Choose a CSV file. It needs to have a column called' Verdict' and a column called 'date", accept_multiple_files=False)

if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    col3, col4 = st.columns(2)
    with col3: 
        date_column = st.selectbox('Which is the Date Column?', options=(dataframe.columns))
    with col4:
        verdict_column = st.selectbox('Which is the Verdict Column?', options=(dataframe.columns))
if uploaded_file is None:
    dataframe = pd.read_csv('Samples for NCC Review - Windows.csv')
    verdict_column = "Verdict (cred theft)"
    date_column = "Date review requested"



col1, col2 = st.columns(2)
with col1:
    approach = st.selectbox('Methodology', options=(["Frequentist","Bayesian", "Rolling Bayesian"]))
    if approach == 'Rolling Bayesian':
        length_of_memory = st.slider('Length of Memory', value = 50)
    else:
        length_of_memory = 0
    sample_size = st.slider('Samples per Period', value = 20)
    bootstrap_sample_size = sample_size

with col2:
    scenario = st.selectbox('Scenario', options=(["Flat", "Increasing"] ))

    if scenario != "Flat":
        weekly_change = st.text_input('Weekly Rate of Change', value = 0.005)
        weekly_change = float(weekly_change)
    number_of_weeks = st.slider('Number of Weeks', value = 60) 



    



############################
#### EXECUTION SECTION ####
############################


#bootstrap_sample_size = boo  ## I am sampling chunks of 20 cause that s what we realistically wwill see. But I am not sure that s the correct way of doing it. I am thinking the prior size should be better. We dont want to know the variance of the sample but
 # the belief about population variance.
num_bootstraps = 1000
#length_of_memory = a.length_of_memory
#approach = a.approach
#scenario = a.scenario
#weekly_change = a.weekly_change
#number_of_weeks = a.number_of_weeks
#sample_size = a.sample_size

## Clean up the data
data = dataframe
data = data[data[verdict_column].isin(["Positive", "Negative"])]
data["verdict"] = data[verdict_column].map({"Positive": 1, "Negative": 0})
data = data[[date_column, "verdict"]]
data = data.sort_values(by=date_column, ascending=True)
verdicts = data["verdict"].tolist()[-length_of_memory:]



## Bootstrap the data
bootstrap_estimates = np.zeros(num_bootstraps)
for i in range(num_bootstraps):
   resampled_data = data.sample(n=bootstrap_sample_size, replace=True)
   resampled_verdicts = resampled_data["verdict"]
   resampled_num_verdicts_1 = np.sum(resampled_verdicts == 1)
   resampled_total_verdicts = len(resampled_verdicts)
   resampled_verdict_precision = resampled_num_verdicts_1 / resampled_total_verdicts
   bootstrap_estimates[i] = resampled_verdict_precision

variance = np.var(bootstrap_estimates)
mean = np.mean(bootstrap_estimates)
alpha = (mean * (1 - mean) / variance) - 1
beta = alpha * (1 - mean) / mean
# Plot the prior distribution
x = np.linspace(0, 1, 100)
y = b.pdf(x, alpha, beta)
fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(x, y)
ax.set_xlabel("Precision")
ax.set_ylabel("Probability Density")
ax.set_title("Beta Distribution")



if approach != 'Frequentist':
    with st.expander("Show Beta Distribution (where we think precision would fall)"):
            st.pyplot(fig)


# generate the new data based on the scenario (if statement)

mean_by_day = []
lower_bound_by_day = []
upper_bound_by_day = []
np.random.seed(0)
noise = np.random.normal(0, np.sqrt(variance), number_of_weeks)
updating_alpha = alpha
updating_beta = beta


for i in range(1, number_of_weeks):
   if scenario == "Flat":
       resampled_data_new = data.sample(n=sample_size, replace=True)
       resampled_verdicts = resampled_data_new["verdict"]
       resampled_num_verdicts_1 = np.sum(resampled_verdicts == 1)
       resampled_verdict_precision = resampled_num_verdicts_1 / sample_size
   elif scenario == "Increasing":
       pre_noise = mean + (i * weekly_change)
       resampled_verdict_precision = min(pre_noise + noise[i], 1)
   elif scenario == "Decreasing":
       pre_noise = mean - (i * weekly_change)
       resampled_verdict_precision = min(pre_noise + noise[i], 1)

   mean_by_day.append(resampled_verdict_precision)

for i in range(1, number_of_weeks):
   if approach == "Bayesian":
       alpha_prime = alpha + sample_size * mean_by_day[i-1]
       beta_prime = beta + sample_size * (1 - mean_by_day[i-1])
       lower_CI = b.ppf(0.025, alpha_prime, beta_prime)
       upper_CI = b.ppf(0.975, alpha_prime, beta_prime)
   elif approach == "Frequentist":
       lower_CI = mean_by_day[i-1] - 1.96 * np.sqrt((mean_by_day[i-1] * (1 - mean_by_day[i-1])) / sample_size )
       upper_CI = mean_by_day[i-1] + 1.96 * np.sqrt((mean_by_day[i-1] * (1 - mean_by_day[i-1]))/ sample_size)
   elif approach == "Rolling Bayesian":
       new_precision = np.mean(verdicts)
       updating_alpha = (new_precision * (1 - new_precision) / variance) - 1
       updating_beta = alpha * (1 - new_precision) / new_precision
       alpha_prime = updating_alpha + sample_size * mean_by_day[i-1]
       beta_prime = updating_beta + sample_size * (1 - mean_by_day[i-1])
       new_datapoints = np.random.binomial(1, mean_by_day[i-1], sample_size)
       verdicts = verdicts[sample_size:] + new_datapoints.tolist()
       lower_CI = b.ppf(0.025, alpha_prime, beta_prime)
       upper_CI = b.ppf(0.975, alpha_prime, beta_prime)

   lower_bound_by_day.append(lower_CI)
   upper_bound_by_day.append(upper_CI)


# plot the new data
means = np.array(mean_by_day)
lower_bounds = np.array(lower_bound_by_day)
upper_bounds = np.array(upper_bound_by_day)
days = np.arange(len(means))

# calculate the new metrics
average_difference = np.mean(upper_bounds - lower_bounds)
max_span = np.max(upper_bounds) - np.min(lower_bounds)
middle_of_ci = (upper_bounds + lower_bounds) / 2
## regression metric
days_reshaped = days.reshape(-1, 1)
model = LinearRegression()
model.fit(days_reshaped, middle_of_ci)
r_squared = model.score(days_reshaped, middle_of_ci)
pred_slope = model.coef_[0]



show_lr = st.checkbox("Show Line Fitted to CIs")

if show_lr:
    # Create the plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.fill_between(days, lower_bounds, upper_bounds, alpha=0.2)
    
    # Plot the line as a solid line (you can modify this for your "means" line if needed)
    ax2.plot(days, means, "-", label="Means")
    
    # Add the regression line as a dotted line
    regression_line = model.predict(days_reshaped)  # Get the regression line values
    ax2.plot(days, regression_line, 'r--', label="Fitted Line")  # 'r--' makes it red and dotted
    
    # Title and labels
    ax2.set_title("Weekly Precision with Error Bars", fontsize=16)
    ax2.set_xlabel("Week", fontsize=14)
    ax2.set_ylabel("Precision", fontsize=14)
    ax2.set_ylim(0, 1)
    
    # Optional: Add legend
    ax2.legend()

    # Show the plot
    st.pyplot(fig2)

else:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.fill_between(days, lower_bounds, upper_bounds, alpha=0.2)
    ax2.plot(days, means, "-")
    ax2.set_title("Weekly Precision with Error Bars", fontsize=16)
    ax2.set_xlabel("Week", fontsize=14)
    ax2.set_ylabel("Precision", fontsize=14)
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)


c5,c6,c7 = st.columns(3)
with c5:
    st.metric(label="Average Width of CI", value=round(average_difference,2))
with c6:
    st.metric(label="Predicted Slope", value=round(pred_slope,4))
with c7:
    st.metric(label="How Confident About Prediction?", value=round(r_squared,2))



# ### fix the rolling bayesian then deploy 
# ### Why does it change the overall data that it displays?
# the problem is the part I commented out