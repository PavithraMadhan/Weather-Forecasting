import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import pandas as pd
import seaborn as sns
import tensorflow as tf
from colorama import Fore, Style
from IPython.core.display import HTML
from plotly.subplots import make_subplots
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)
weather = pd.read_csv(
    "Dataset.csv",
    parse_dates=["DATE"],
    index_col="DATE"
)

weather.head()

weather.sample()

weather.tail()

weather.info()

temp = weather[["TAVG"]].copy()
temp.describe().T.drop("count", axis=1).rename(columns=str.title)

temp.query("TAVG == -22.3")

temp.query("TAVG == 29.1")

fig = px.line(
    temp,
    y="TAVG",
    labels={"DATE": "Date", "TAVG": "Average Temperature (\u2103)"},  # Celsius degree.
    title="Average Daily Temperatures in Warsaw, Poland in 1993-2022 (Last 30 Years)",
    height=420,
    width=840,
)
fig.update_layout(
    
    title_font_size=18
)
fig.update_traces(
    line=dict(width=1.0, color="#2A357D"),
    opacity=0.5,
)
fig.show()

resampled_2022 = temp.loc["2022"].resample("7D").mean()

fig = px.area(
    temp.loc["2022"],
    y="TAVG",
    labels={"DATE": "Date", "TAVG": "Average Temperature (\u2103)"},
    title="Average Daily Temperatures in Warsaw, Poland in 2022",
    height=420,
    width=840,
)
fig.update_traces(line=dict(width=1.5, color="#67727e"))
fig.add_scatter(
    x=resampled_2022.index,
    y=resampled_2022.TAVG,
    showlegend=False,
    text="7 Day Frequency",
    line_shape="spline",
    line=dict(dash="solid", color="#d4674c", width=3),
)
fig.update_layout(
    
    title_font_size=18,
)
fig.show()

tavg_kde = gaussian_kde(temp.TAVG)

tavg_range = np.linspace(temp.TAVG.min(), temp.TAVG.max(), len(temp))

kde_estimated = tavg_kde.evaluate(tavg_range)

fig = px.histogram(
    temp,
    x="TAVG",
    marginal="box",
    histnorm="probability density",
    title="Probability Density of Daily Temperatures (Based on 30 Years of Measurements)",
    color_discrete_sequence=["#2A357D"],
    nbins=100,
    height=540,
    width=840,
)
fig.add_scatter(
    x=tavg_range,
    y=kde_estimated,
    showlegend=False,
    text="Average Temperature KDE",
    line=dict(dash="solid", color="#d4674c", width=4),
)
fig.update_layout(
    
    title_font_size=18,
    
    bargap=0.25,
    xaxis_title_text="Average Temperature (\u2103)",
    yaxis_title_text="Probability Density",
)
fig.show()

tavg_monthly = (
    temp.groupby(temp.index.month_name(), sort=False).mean(numeric_only=True).reset_index()
)

fig = px.bar(
    tavg_monthly,
    x="DATE",
    y="TAVG",
    labels={"TAVG": "Mean Average Temperature (\u2103)", "DATE": "Month"},
    title="Mean Average Temperature by Month (Based on 30 Years of Measurements)",
    text_auto=".2f",
    color="TAVG",
    color_continuous_scale=px.colors.sequential.Burgyl_r,
    height=500,
    width=840,
)
fig.update_layout(
    
    title_font_size=18,
   
    coloraxis_colorbar_title_text="Temperature (\u2103)",
)
fig.show()

adf_result = adfuller(temp.TAVG)

print("Augmented Dickey-Fuller Test for data within daily frequency:")

print("● critical value:".ljust(52), f"{adf_result[0]:.2f}")
print("● p-value:".ljust(52), f"{adf_result[1]:.1e}")
print("● number of lags used:".ljust(52), f"{adf_result[2]}")
print("● number of observations:".ljust(52), f"{adf_result[3]}")
print(
    "● t-values at 1%, 5% and 10% confidence intervals:".ljust(52),
    f"{np.array(list(adf_result[4].values())).round(2)}",
)

adf_result = adfuller(temp.TAVG.resample("3M").mean())

print("Augmented Dickey-Fuller Test for data within three-month frequency:")
print("=" * 72)
print("● critical value:".ljust(52), f"{adf_result[0]:.2f}")
print("● p-value:".ljust(52), f"{adf_result[1]:.2f}")
print("● number of lags used:".ljust(52), f"{adf_result[2]}")
print("● number of observations:".ljust(52), f"{adf_result[3]}")
print(
    "● t-values at 1%, 5% and 10% confidence intervals:".ljust(52),
    f"{np.array(list(adf_result[4].values())).round(2)}",
)
print("=" * 72)


fig = px.line(
    temp.diff(365),
    y="TAVG",
    labels={"DATE": "Date", "TAVG": "Temperature Difference (\u2103)"},
    title="Average Temperature Difference on a Given Day Year-by-Year (Stationary Signal)",
    height=420,
    width=840,
)
fig.update_layout(
    
    title_font_size=18,
    
)
fig.update_traces(
    line=dict(width=1.0, color="#2A357D"),
    opacity=0.5,
)
fig.show()

decomposition = seasonal_decompose(temp.TAVG, model="additive", period=365)

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    x_title="Date",
    y_title="Temperature (\u2103)",
    subplot_titles=["Observed Values", "Trend", "Seasonality", "Residuals"],
)

observed = go.Scatter(
    x=decomposition.observed.index,
    y=decomposition.observed,
    name="Observed Temperature",
    line=dict(width=0.7, color="#2A357D"),
)
trend = go.Scatter(
    x=decomposition.trend.index,
    y=decomposition.trend,
    name="Trend",
    line=dict(color="#2A357D"),
)
seasonal = go.Scatter(
    x=decomposition.seasonal.index,
    y=decomposition.seasonal,
    name="Seasonality",
    line=dict(color="#2A357D"),
)
residuals = go.Scatter(
    x=decomposition.resid.index,
    y=decomposition.resid,
    name="Residuals",
    mode="markers",
    marker_size=1,
    line=dict(color="#2A357D"),
)

fig.add_trace(observed, row=1, col=1)
fig.add_trace(trend, row=2, col=1)
fig.add_trace(seasonal, row=3, col=1)
fig.add_trace(residuals, row=4, col=1)

fig.update_annotations(font_size=14)
fig.update_layout(
    
    title_font_size=18,
    
    title_text="Average Daily Temperatures - Seasonal Decomposition",
    showlegend=False,
    height=800,
    width=840,
)
fig.show()

ef draw_acf(series, n_lags, marker_size=12):
    corr_array = acf(series, alpha=0.05, nlags=n_lags)
    corr_values = corr_array[0]
    lags = np.arange(len(corr_values))
    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()

    for l in lags:
        fig.add_scatter(
            x=(l, l), y=(0, corr_values[l]), mode="lines", line_color="black"
        )

    fig.add_scatter(
        x=lags,
        y=corr_values,
        mode="markers",
        marker_color="#2A357D",
        marker_size=marker_size,
        name="ACF",
    )
    fig.add_scatter(x=lags, y=upper_y, mode="lines", line_color="rgba(255,255,255,0)")
    fig.add_scatter(
        x=lags,
        y=lower_y,
        mode="lines",
        fillcolor="rgba(32, 146, 230, 0.3)",
        fill="tonexty",
        line_color="rgba(255, 255, 255, 0)",
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, n_lags + 1])
    fig.update_yaxes(zerolinecolor="black")
    fig.update_layout(
        
        title_font_size=18,
        
        title_text="Average Temperature Autocorrelation (ACF)",
        xaxis_title="Lag (Months)",
        yaxis_title="ACF",
        height=500,
        width=840,
    )
    fig.show()

df_monthly = temp.resample("M").mean()
draw_acf(df_monthly, n_lags=12)

draw_acf(df_monthly, n_lags=120, marker_size=6)

train_weekly = temp.TAVG["2017":"2021"].resample("7D").mean()
train_daily = temp.TAVG["2017":"2021"]

avg_forecast = (
    train_daily.groupby(train_daily.index.isocalendar().week)
    .mean()
    .set_axis(pd.date_range("2022-01-01", periods=53, freq="7D"))
)

train_test_weekly = temp.TAVG["2017":"2022"].resample("7D").mean()
train_with_forecast = pd.concat([train_weekly, avg_forecast])

fig = px.line(
    x=train_test_weekly.index,
    y=train_test_weekly,
    labels={"x": "Date", "y": "Average Temperature (\u2103)"},
    title="Average Weekly Temperatures & Average Forecast (Based on Last 5 Years)",
    height=420,
    width=840,
)
fig.update_traces(line=dict(width=1.5, color="#2A357D"), opacity=0.7)
fig.add_scatter(
    x=avg_forecast.index,
    y=avg_forecast,
    mode="markers+lines",
    marker=dict(symbol="x", size=6),
    line=dict(color="#d4674c", width=1.5),
    name="Average Forecast",
)
fig.update_layout(
   
    title_font_size=18,
   
    legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.02, x=1),
)
fig.show()

actual_weekly = temp.TAVG["2022"].resample("7D").mean()
avg_method_mae = mean_absolute_error(actual_weekly, avg_forecast)
print("Average Method - Mean Absolute Error: ", f"{avg_method_mae:.2f}")

naive_forecast = temp.TAVG["2021"].resample("7D").mean()
naive_method_mae = mean_absolute_error(actual_weekly, naive_forecast)
print("Naive Method - Mean Absolute Error: ", f"{naive_method_mae:.2f}")

p = d = q = range(0, 2)
pdq = list(product(p, d, q))
s_pdq = list((*combination, 53) for combination in pdq)

np.random.seed(42)

for i_pdq, i_spdq in zip(np.random.permutation(pdq), np.random.permutation(s_pdq)):
    print(
        "ARIMA",
         "({}, {}, {})".format(*i_pdq),
        "x", "({}, {}, {}, {})".format(*i_spdq),
    )

train = temp.TAVG["2017":"2021"].resample("7D").mean()
model = ARIMA(train, order=(1, 1, 1), seasonal_order=(1, 2, 1, 53)).fit()
model.summary()

fig = plt.figure(figsize=(11.7, 11), tight_layout=True)
model.plot_diagnostics(fig=fig)
ax = fig.get_axes()[2]
ax.get_lines()[0].set_markersize(4.0)
ax.get_lines()[0].set_alpha(0.5)
ax.get_lines()[1].set_linewidth(3.0)
ax.get_lines()[1].set_color("#67727e")
plt.show()

arima_forecast = model.predict(start="2022-01-01", end="2022-12-31")

fig = px.line(
    x=train_test_weekly.index,
    y=train_test_weekly,
    labels={"x": "Date", "y": "Average Temperature (\u2103)"},
    title="Average Weekly Temperatures & ARIMA Forecast (Based on Last 5 Years)",
    height=420,
    width=840,
)
fig.update_traces(line=dict(width=1.5, color="#2A357D"), opacity=0.7)
fig.add_scatter(
    x=arima_forecast.index,
    y=arima_forecast,
    mode="markers+lines",
    marker=dict(symbol="x", size=6),
    line=dict(color="#d4674c", width=1.5),
    name="Seasonal ARIMA Forecast",
)
fig.update_layout(
    
    title_font_size=18,
    
    legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.02, x=1),
)
fig.show()

arima_method_mae = mean_absolute_error(actual_weekly, arima_forecast)
print("ARIMA Method - Mean Absolute Error: ", f"{arima_method_mae:.2f}")

example_series = np.arange(0, 10)
seq_length = 5  # For example [0, 1, 2, 3, 4].
ahead_steps = 2  # For example [5, 6].
batch_size = 2

example_ds = keras.preprocessing.timeseries_dataset_from_array(
    example_series,
    targets=None,
    sequence_length=seq_length + ahead_steps,
    batch_size=batch_size,
).map(lambda series: (series[:, :-ahead_steps], series[:, -ahead_steps:]))

list(example_ds)

train = temp.TAVG["2007":"2016"].resample("7D").mean()
valid = temp.TAVG["2017":"2021"].resample("7D").mean()

seq_length = 104  # Window over 2 years.
ahead_steps = 53  # Forecast for the next year.
batch_size = 32

tf.random.set_seed(42)

train_ds = keras.preprocessing.timeseries_dataset_from_array(
    train.to_numpy(),
    targets=None,
    sequence_length=seq_length + ahead_steps,
    batch_size=batch_size,
    shuffle=True,
    seed=42,
).map(lambda series: (series[:, :-ahead_steps], series[:, -ahead_steps:]))

valid_ds = keras.preprocessing.timeseries_dataset_from_array(
    valid.to_numpy(),
    targets=None,
    sequence_length=seq_length + ahead_steps,
    batch_size=batch_size,
).map(lambda series: (series[:, :-ahead_steps], series[:, -ahead_steps:]))

tf.random.set_seed(42)

rnn_model = keras.Sequential(
    [
        layers.Normalization(input_shape=[None, 1]),
        layers.LSTM(32, return_sequences=True, dropout=0.2),
        layers.LayerNormalization(),
        layers.LSTM(32, return_sequences=True, dropout=0.2),
        layers.LayerNormalization(),
        layers.LSTM(32, dropout=0.2),
        layers.LayerNormalization(),
        layers.Dense(ahead_steps),
    ]
)

n_epochs = 200
n_steps = n_epochs * len(list(train_ds))

early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_mae", patience=50, restore_best_weights=True
)

scheduled_lr = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.05, decay_steps=n_steps, decay_rate=0.1
)

optimizer = keras.optimizers.RMSprop(learning_rate=scheduled_lr)

rnn_model.compile(loss="huber", optimizer=optimizer, metrics=["mae"])

history = rnn_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=n_epochs,
    callbacks=[early_stopping_cb],
    verbose=0,
)

rnn_history = pd.DataFrame(history.history)

fig = px.line(
    rnn_history,
    labels={"variable": "Variable", "value": "Value", "index": "Epoch"},
    title="RNN Training Process",
    color_discrete_sequence=px.colors.diverging.balance_r,
    height=420,
    width=840,
)
fig.update_layout(
    
    title_font_size=18,
    
    legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.02, x=1),
)
fig.show()

rnn_test = valid.to_numpy()[np.newaxis, :]
rnn_forecast = pd.Series(
    rnn_model.predict(rnn_test, verbose=0).flatten(),
    index=pd.date_range("2022-01-01", periods=ahead_steps, freq="7D"),
)

fig = px.line(
    x=train_test_weekly.index,
    y=train_test_weekly,
    labels={"x": "Date", "y": "Average Temperature (\u2103)"},
    title="Average Weekly Temperatures & RNN Forecast (Based on Last 5 Years)",
    height=420,
    width=840,
)
fig.update_traces(line=dict(width=1.5, color="#2A357D"), opacity=0.7)
fig.add_scatter(
    x=rnn_forecast.index,
    y=rnn_forecast,
    mode="markers+lines",
    marker=dict(symbol="x", size=6),
    line=dict(color="#d4674c", width=1.5),
    name="RNN Forecast",
)
fig.update_layout(
    
    title_font_size=18,
    
    legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.02, x=1),
)
fig.show()

nn_method_mae = mean_absolute_error(actual_weekly, rnn_forecast)
print("RNN Method - Mean Absolute Error: ", f"{rnn_method_mae:.2f}")

actual_plus_forecats = pd.DataFrame(
    {
        "Actual": actual_weekly,
        "Average Forecast": avg_forecast.to_numpy(),
        "Arima Forecast": arima_forecast.to_numpy(),
        "RNN Forecast": rnn_forecast.to_numpy(),
    },
    index=pd.date_range("2022-01-01", periods=53, freq="7D", name="Date")
)

fig = px.line(
    actual_plus_forecats,
    labels={"variable": "Method", "value": "Average Temperature (\u2103)"},
    title="Actual Weekly Temperatures and Predictions within Different Methods",
    color_discrete_sequence=px.colors.diverging.balance_r,
    line_dash="variable",
    line_dash_sequence=["solid", "dash", "dot", "dashdot"],
    height=520,
    width=840,
)
fig.update_layout(
    
    title_font_size=18,

    legend=dict(orientation="h", yanchor="bottom", xanchor="right", y=1.02, x=1),
)
fig.show()
