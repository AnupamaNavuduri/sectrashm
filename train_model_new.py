import re
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from influxdb_client import InfluxDBClient


INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = "0PNkpyhWRt8GHHHUuicx6XITSqR68MjWYfkbmBRdvBXLAEHRRZHAJGXKuzUdOKJZnHFOw2RKWg40rv_aZQ5xIQ=="
INFLUXDB_ORG = "965feae5fdbd9a75"


def fetch_latest_influxdb_data(node):
    query = '''
    from(bucket: "node2")
    |> range(start: -30d)
    |> filter(fn: (r) => r["_measurement"] == "TODAY")
    |> filter(fn: (r) => exists r._value)
    |> limit(n:500)
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> drop(columns: ["_start", "_stop", "_measurement"])
    |> sort(columns: ["_time"])
    '''

    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query_api = client.query_api()
        df = query_api.query_data_frame(query, org=INFLUXDB_ORG)

    def extract_last6_floats(row):
        val_str = row['value']
        if val_str is None or not isinstance(val_str, str):
            return pd.Series([np.nan]*6 + [row['_time']])
        ts_match = re.search(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}", val_str)
        timestamp = ts_match.group() if ts_match else None
        val_str_wo_ts = val_str.replace(timestamp, "") if timestamp else val_str
        nums = re.findall(r"-?\d+\.\d+|-?\d+", val_str_wo_ts)
        if len(nums) < 6:
            return pd.Series([np.nan]*6 + [row['_time']])
        sensor_values = nums[-6:]
        sensor_floats = [float(x) for x in sensor_values]
        return pd.Series(sensor_floats + [row['_time']])

    parsed = df.apply(extract_last6_floats, axis=1)
    parsed.columns = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'timestamp']
    parsed = parsed.dropna()
    print(parsed.head())
    return parsed


class MahonyFilter:
    def __init__(self, sample_period=1/256, kp=0.5, ki=0.1):
        self.sample_period = sample_period
        self.kp = kp
        self.ki = ki
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.integral_error = np.zeros(3)

    def update(self, gyro, acc):
        gyro = np.asarray(gyro, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)

        q = self.quaternion
        if np.linalg.norm(acc) == 0:
            return q
        acc = acc / np.linalg.norm(acc)
        v = np.array([
            2*(q[1]*q[3] - q[0]*q[2]),
            2*(q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2,
        ])
        error = np.cross(v, acc)
        self.integral_error += error * self.ki * self.sample_period
        gyro_adj = gyro + self.kp * error + self.integral_error
        q_dot = 0.5 * self.quaternion_multiply(q, np.insert(gyro_adj, 0, 0.0))
        q = q + q_dot * self.sample_period
        self.quaternion = q / np.linalg.norm(q)
        return self.quaternion

    def quaternion_multiply(self, q, r):
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array([
            -x0*x1 - y0*y1 - z0*z1 + w0*w1,
             x0*w1 + y0*z1 - z0*y1 + w0*x1,
            -x0*z1 + y0*w1 + z0*x1 + w0*y1,
             x0*y1 - y0*x1 + z0*w1 + w0*z1
        ])


def extract_features(acc_win, gyro_win, euler_win):
    acc_win = np.asarray(acc_win, dtype=np.float64)
    gyro_win = np.asarray(gyro_win, dtype=np.float64)
    euler_win = np.asarray(euler_win, dtype=np.float64)

    feats = {}
    feats['acc_mag_mean'] = np.mean(np.linalg.norm(acc_win.astype(np.float64), axis=1))
    feats['gyro_mag_mean'] = np.mean(np.linalg.norm(gyro_win.astype(np.float64), axis=1))
    feats['acc_std'] = np.mean(np.std(acc_win, axis=0))
    feats['gyro_std'] = np.mean(np.std(gyro_win, axis=0))
    feats['roll_mean'] = np.mean(euler_win[:, 0])
    feats['pitch_mean'] = np.mean(euler_win[:, 1])
    feats['yaw_mean'] = np.mean(euler_win[:, 2])
    feats['roll_std'] = np.std(euler_win[:, 0])
    feats['pitch_std'] = np.std(euler_win[:, 1])
    feats['yaw_std'] = np.std(euler_win[:, 2])
    return feats


def assign_motion_labels(features_df, cluster_col='motion_cluster'):
    cluster_stats = features_df.groupby(cluster_col)[['acc_mag_mean', 'gyro_mag_mean', 'acc_std', 'gyro_std']].mean()
    clustermap = {}
    # Define thresholds for "still"
    acc_threshold = 6.0  # adjust as per your still window stats
    gyro_threshold = 2.5 # adjust as per your still window stats
    

    for idx, row in cluster_stats.iterrows():
        if row['acc_mag_mean'] < acc_threshold and row['gyro_mag_mean'] < gyro_threshold:
            clustermap[idx] = 'still'
    # Map others by ranking remaining by gyro_mag_mean etc.
    remaining = [i for i in cluster_stats.index if i not in clustermap]
    if remaining:
        # Highest gyro: drop
        drop_cluster = cluster_stats.loc[remaining, 'gyro_mag_mean'].idxmax()
        clustermap[drop_cluster] = 'drop'
        remaining.remove(drop_cluster)
    if remaining:
        # Next highest gyro: shake
        shake_cluster = cluster_stats.loc[remaining, 'gyro_mag_mean'].idxmax()
        clustermap[shake_cluster] = 'shake'
        remaining.remove(shake_cluster)
    if remaining:
        # Leftover is tilt
        clustermap[remaining[0]] = 'tilt'
    return features_df[cluster_col].map(clustermap)


def pipeline_process(acc_data, gyro_data, window_size=50, step_size=25, sample_period=1/256, n_clusters=4, epochs=50):
    mahony = MahonyFilter(sample_period)
    n_samples = acc_data.shape[0]
    quats = np.zeros((n_samples, 4))
    for i in range(n_samples):
        gyro_i = np.array(gyro_data[i], dtype=np.float64)
        acc_i = np.array(acc_data[i], dtype=np.float64)
        quats[i] = mahony.update(gyro_i, acc_i)
    rotations = R.from_quat(quats[:, [1, 2, 3, 0]])
    euler_angles = rotations.as_euler('xyz')

    feature_list = []
    for start in range(0, n_samples - window_size + 1, step_size):
        acc_win = acc_data[start:start + 50]
        gyro_win = gyro_data[start:start + 50]
        euler_win = euler_angles[start:start + 50]
        feats = extract_features(acc_win, gyro_win, euler_win)
        feature_list.append(feats)

    features_df = pd.DataFrame(feature_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    # Save fitted scaler to disk
    joblib.dump(scaler, 'scaler.joblib')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    motion_clusters = kmeans.fit_predict(X_scaled)
    features_df['motion_cluster'] = motion_clusters
    features_df['motion_label'] = assign_motion_labels(features_df)

    # Add LabelEncoder
    label_encoder = LabelEncoder()
    features_df['motion_label_encoded'] = label_encoder.fit_transform(features_df['motion_label'])
    joblib.dump(label_encoder, 'label_encoder.joblib')

    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    x = Dense(16, activation='relu')(input_layer)
    x = Dense(8, activation='relu')(x)
    output_layer = Dense(len(label_encoder.classes_), activation='softmax')(x)
    motion_model = Model(input_layer, output_layer)
    motion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(features_df['motion_label_encoded'], num_classes=len(label_encoder.classes_))
    motion_model.fit(X_scaled, y, epochs=epochs, batch_size=32, validation_split=0.2)
    motion_model.save('cnn_lstm_model.h5')

    input_layer_ae = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer_ae)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer_ae, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=32, validation_split=0.2)
    autoencoder.save('lstm_autoencoder_model.h5')

    X_pred = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)
    threshold = np.percentile(reconstruction_error, 95)
    features_df['is_anomaly'] = (reconstruction_error > threshold).astype(int)

    output_df = features_df[(features_df['motion_label'] != 'still') & (features_df['is_anomaly'] == 1)][['motion_label', 'is_anomaly']]

    return output_df


if __name__ == "__main__":
    node_name = "node2"
    df = fetch_latest_influxdb_data(node_name)
    print("Fetched data:", df.head())
    print("total rows:", len(df))
    acc_data = df[['AccX', 'AccY', 'AccZ']].to_numpy()
    gyro_data = df[['GyroX', 'GyroY', 'GyroZ']].to_numpy()

    results = pipeline_process(acc_data, gyro_data)
    print(f"No. of anomalies detected : {results.shape[0]}")
    print(results)
