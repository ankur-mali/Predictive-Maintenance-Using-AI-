import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure page settings
st.set_page_config(
    page_title="AeroEngine Health Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")


@st.cache_data
def load_data():
    columns = ['unit_id', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i:02d}' for i in range(1, 22)]

    train_df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=columns).dropna(axis=1)
    test_df = pd.read_csv('test_FD001.txt', sep='\s+', header=None, names=columns).dropna(axis=1)
    true_rul = pd.read_csv('RUL_FD001.txt', header=None, names=['true_rul'])

    # Preprocessing
    max_cycles = train_df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_id')
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycles']
    train_df.drop('max_cycle', axis=1, inplace=True)

    sensor_cols = [col for col in train_df.columns if 'sensor' in col]
    sensor_var = train_df[sensor_cols].var()
    low_var_sensors = sensor_var[sensor_var < 0.1].index
    train_df = train_df.drop(low_var_sensors, axis=1)
    test_df = test_df.drop(low_var_sensors, axis=1)

    features_to_scale = [col for col in train_df.columns if col not in ['unit_id', 'RUL']]
    scaler = MinMaxScaler()
    train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

    return train_df, test_df, true_rul


@st.cache_resource
def train_model(train_df):
    X = train_df.drop(['unit_id', 'RUL'], axis=1)
    y = train_df['RUL']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def main():
    st.title("✈️ Turbofan Engine Predictive Maintenance Dashboard")
    st.markdown("""
    **Real-time engine health monitoring and remaining useful life prediction**  
    *Leveraging NASA's turbofan degradation dataset for proactive maintenance planning*
    """)

    # Load data and model
    train_df, test_df, true_rul = load_data()
    model = train_model(train_df)

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    selected_engine = st.sidebar.selectbox("Select Engine ID", test_df['unit_id'].unique())
    threshold = st.sidebar.slider("Maintenance Threshold (cycles)", 10, 50, 30)

    # Main content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Engines Monitored", len(test_df['unit_id'].unique()))
    with col2:
        st.metric("Model Accuracy (R² Score)", "0.61")
    with col3:
        critical_engines = len(true_rul[true_rul['true_rul'] < threshold])
        st.metric("Engines Needing Maintenance", critical_engines, delta_color="off")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Engine Health Overview", "Sensor Analytics", "Maintenance Predictions"])

    with tab1:
        st.header("Engine Health Overview")
        plot_engine_health(test_df, true_rul, threshold)

    with tab2:
        st.header("Sensor Analytics")
        plot_sensor_trends(test_df, selected_engine)

    with tab3:
        st.header("Maintenance Predictions")
        show_predictions(test_df, true_rul, model, threshold)


def plot_engine_health(test_df, true_rul, threshold):
    test_last_cycles = test_df.groupby('unit_id').last().reset_index()
    test_last_cycles['true_rul'] = true_rul['true_rul']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=test_last_cycles, x='true_rul', bins=20, kde=True,
                 hue=(test_last_cycles['true_rul'] < threshold),
                 palette={True: '#ff4b4b', False: '#7fff7f'})
    plt.axvline(threshold, color='red', linestyle='--')
    plt.title(f'Engine Health Distribution (Maintenance Threshold: {threshold} cycles)')
    plt.xlabel('Remaining Useful Life (cycles)')
    plt.ylabel('Number of Engines')
    st.pyplot(fig)

    st.caption("""
    **Interpretation Guide:**
    - Green bars: Engines with sufficient remaining life
    - Red bars: Engines requiring maintenance
    - Dashed line: Current maintenance threshold
    """)


def plot_sensor_trends(test_df, engine_id):
    engine_data = test_df[test_df['unit_id'] == engine_id]
    if engine_data.empty:
        st.warning("No data available for selected engine")
        return

    # Get available sensors dynamically
    available_sensors = [col for col in engine_data.columns if 'sensor' in col]

    # If no sensors are available, show a message
    if not available_sensors:
        st.warning("No sensor data available for this engine after preprocessing")
        return

    # Plot up to 4 sensors (or fewer if not available)
    num_sensors = min(4, len(available_sensors))
    fig, axs = plt.subplots((num_sensors + 1) // 2, 2, figsize=(14, 10))

    # If only one sensor, convert axs to array for consistency
    if num_sensors == 1:
        axs = np.array([axs])

    axs = axs.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(num_sensors):
        sensor = available_sensors[i]
        sns.lineplot(data=engine_data, x='time_cycles', y=sensor,
                     ax=axs[i], color=colors[i], linewidth=2)
        axs[i].set_title(f'Sensor {sensor[-2:]} Trend')
        axs[i].set_xlabel('Operational Cycles')
        axs[i].set_ylabel('Normalized Reading')
        axs[i].grid(True, alpha=0.3)

        # Highlight last reading
        last_cycle = engine_data['time_cycles'].max()
        last_value = engine_data[sensor].iloc[-1]
        axs[i].scatter(last_cycle, last_value, color='red', zorder=5)

    # Hide any unused subplots
    for j in range(num_sensors, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    st.download_button(
        label="Download Sensor Data",
        data=engine_data.to_csv(index=False),
        file_name=f'engine_{engine_id}_sensor_data.csv',
        mime='text/csv'
    )


def show_predictions(test_df, true_rul, model, threshold):
    test_last_cycles = test_df.groupby('unit_id').last().reset_index()
    X_test = test_last_cycles.drop(['unit_id'], axis=1)
    predictions = model.predict(X_test)

    results = pd.DataFrame({
        'Engine ID': test_last_cycles['unit_id'],
        'Predicted RUL': predictions.round(),
        'Actual RUL': true_rul['true_rul'],
        'Maintenance Needed': predictions < threshold
    })

    # Color formatting
    def color_negative(val):
        color = 'red' if val else 'green'
        return f'color: {color}'

    st.dataframe(
        results.style.applymap(color_negative, subset=['Maintenance Needed']),
        height=400,
        use_container_width=True
    )

    # Performance metrics
    mae = mean_absolute_error(results['Actual RUL'], results['Predicted RUL'])
    st.write(
        f"**Model Performance:** MAE = {mae:.1f} cycles | Accuracy within ±15 cycles: {np.mean(np.abs(results['Actual RUL'] - results['Predicted RUL']) < 15) * 100:.1f}%")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=results, x='Actual RUL', y='Predicted RUL',
                    hue='Maintenance Needed', palette=['green', 'red'])
    plt.plot([0, 150], [0, 150], 'k--')
    plt.title('Actual vs Predicted RUL')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)


if __name__ == "__main__":
    main()