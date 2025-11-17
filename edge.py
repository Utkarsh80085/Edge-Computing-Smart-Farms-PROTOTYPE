# edge_prototype.py
# Prototype for "Efficient productivity prediction model based on edge computing in smart farms"

import os, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure scikit-learn and joblib are installed (run pip install scikit-learn joblib if missing)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import joblib
except Exception as e:
    raise SystemExit("Missing dependencies. Run: pip install scikit-learn joblib pandas matplotlib")

out_dir = Path('./edge_prototype')
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Generate synthetic dataset for training (sensor + image features -> productivity)
def generate_synthetic_dataset(n_samples=2000, seed=42):
    rng = np.random.RandomState(seed)
    soil = rng.uniform(5, 60, size=n_samples)
    temp = rng.uniform(12, 36, size=n_samples)
    hum = rng.uniform(30, 95, size=n_samples)
    ph = rng.uniform(4.5, 7.5, size=n_samples)
    ndvi = np.clip(0.8 - 0.01*(temp-25) + 0.002*(hum-60) + rng.normal(0,0.05,size=n_samples), 0,1)
    productivity = (0.5*soil + 2.0*ndvi*100 - 0.3*(temp-25)**2 + rng.normal(0,5,size=n_samples))
    df = pd.DataFrame({
        'soil_moisture': soil,
        'temperature': temp,
        'humidity': hum,
        'ph': ph,
        'ndvi': ndvi,
        'productivity': productivity
    })
    return df

df = generate_synthetic_dataset(1500)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=1)
features = ['soil_moisture','temperature','humidity','ph','ndvi']
X_train = train_df[features].values
y_train = train_df['productivity'].values
X_val = val_df[features].values
y_val = val_df['productivity'].values

# Train initial 'cloud' model
model = RandomForestRegressor(n_estimators=50, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rmse = math.sqrt(mean_squared_error(y_val, y_pred))
model_path = out_dir / 'model_v1.joblib'
joblib.dump({'model': model, 'features': features, 'version':1}, model_path)
print(f"Trained initial model_v1, validation RMSE: {rmse:.3f}, saved to {model_path}")

# 2. Simulate edge streaming data and implement edge loop
def simulate_sensor_stream(n_events=200, seed=7):
    rng = np.random.RandomState(seed)
    events = []
    timestamp = 0
    for i in range(n_events):
        soil = max(0, 30 + rng.normal(0,8))
        temp = 25 + rng.normal(0,3)
        hum = 60 + rng.normal(0,8)
        ph = 6 + rng.normal(0,0.3)
        ndvi = np.clip(0.7 + rng.normal(0,0.07) - 0.01*(temp-28), 0,1)
        stress_prob = float(max(0, 0.35 - ndvi + rng.normal(0,0.05)))
        events.append({'ts': timestamp, 'soil_moisture':soil, 'temperature':temp, 'humidity':hum, 'ph':ph, 'ndvi':ndvi, 'stress_prob':stress_prob})
        timestamp += rng.randint(20,80)
    return pd.DataFrame(events)

stream_df = simulate_sensor_stream(300)

# 3. Edge preprocessing: exponential moving average smoothing
def preprocess_series(series, alpha=0.25):
    out = []
    s = None
    for v in series:
        if s is None:
            s = v
        else:
            s = alpha*v + (1-alpha)*s
        out.append(s)
    return np.array(out)

for col in ['soil_moisture','temperature','humidity','ph','ndvi']:
    stream_df[col+'_sm'] = preprocess_series(stream_df[col].values, alpha=0.25)

# 4. Priority classification rule
def classify_priority(row):
    if row['soil_moisture_sm'] < 15 or row['stress_prob']>0.4:
        return 'high'
    if row['soil_moisture_sm'] < 25 or row['stress_prob']>0.25:
        return 'medium'
    return 'low'

stream_df['priority'] = stream_df.apply(classify_priority, axis=1)

# 5. Simulate region-aware clipping -> create small ROI features
def clip_image_roi_simulated(ndvi, stress_prob):
    lesion_score = max(0, stress_prob + np.random.normal(0,0.05))
    return np.array([ndvi, ndvi*(1-ndvi), lesion_score])

stream_df['roi_feat'] = stream_df.apply(lambda r: clip_image_roi_simulated(r['ndvi'], r['stress_prob']), axis=1)

# 6. Edge inference using model_v1
loaded = joblib.load(model_path)
edge_model = loaded['model']

def edge_infer(row):
    feat = np.array([row['soil_moisture_sm'], row['temperature_sm'], row['humidity_sm'], row['ph_sm'], row['ndvi_sm']]).reshape(1,-1)
    pred = edge_model.predict(feat)[0]
    return pred

# 7. Edge loop simulation: immediate actions for high priority, batching others
batch = []
uploaded_batches = []
local_predictions = []
batch_size = 30
for idx, row in stream_df.iterrows():
    pr = row['priority']
    pred = None
    if pr == 'high':
        pred = edge_infer(row)
        if pred < 40:
            action = 'actuate_irrigation'
        else:
            action = 'monitor'
        local_predictions.append({'idx':int(idx),'ts':int(row['ts']),'priority':pr,'prediction':float(pred),'action':action})
    else:
        batch.append({'idx':int(idx),'ts':int(row['ts']),'soil':float(row['soil_moisture_sm']),
                      'temp':float(row['temperature_sm']),'hum':float(row['humidity_sm']),
                      'ph':float(row['ph_sm']),'ndvi':float(row['ndvi_sm']),'priority':pr})
    if len(batch) >= batch_size or idx==stream_df.index[-1]:
        batch_file = out_dir / f'batch_{len(uploaded_batches)+1}.json'
        with open(batch_file,'w') as f:
            json.dump(batch, f)
        uploaded_batches.append(str(batch_file))
        batch = []

pred_log = out_dir / 'local_predictions.json'
with open(pred_log,'w') as f:
    json.dump(local_predictions, f)

print(f"Simulated edge loop complete. Local predictions saved to {pred_log}. Uploaded batch files: {len(uploaded_batches)}")

# 8. Cloud retrain using uploaded batches
def load_batches(paths):
    rows = []
    for p in paths:
        with open(p,'r') as f:
            rows += json.load(f)
    return pd.DataFrame(rows)

cloud_df = load_batches(uploaded_batches)
if not cloud_df.empty:
    cloud_df['productivity'] = 0.5*cloud_df['soil'] + 2.0*cloud_df['ndvi']*100 - 0.3*(cloud_df['temp']-25)**2 + np.random.normal(0,5,size=len(cloud_df))
    X_cloud = cloud_df[['soil','temp','hum','ph','ndvi']].values
    y_cloud = cloud_df['productivity'].values
    Xc_train, Xc_val, yc_train, yc_val = train_test_split(X_cloud, y_cloud, test_size=0.2, random_state=0)
    model_v2 = RandomForestRegressor(n_estimators=60, random_state=1)
    model_v2.fit(Xc_train, yc_train)
    yv = model_v2.predict(Xc_val)
    rmse_v2 = math.sqrt(mean_squared_error(yc_val, yv))
    model_v2_path = out_dir / 'model_v2.joblib'
    joblib.dump({'model': model_v2, 'features': features, 'version':2}, model_v2_path)
    print(f"Cloud retrained model_v2, val RMSE: {rmse_v2:.3f}, saved to {model_v2_path}")
else:
    print("No batches uploaded; skipping cloud retrain.")

# 9. Edge poll and update simulation
def edge_poll_and_update(edge_model_path, cloud_model_path):
    edge_meta = joblib.load(edge_model_path)
    cloud_meta = joblib.load(cloud_model_path)
    if cloud_meta.get('version',0) > edge_meta.get('version',0):
        joblib.dump(cloud_meta, edge_model_path)
        return True, cloud_meta.get('version',None)
    return False, edge_meta.get('version',None)

if (out_dir / 'model_v2.joblib').exists():
    updated, new_ver = edge_poll_and_update(model_path, out_dir / 'model_v2.joblib')
    print(f"Edge polled cloud; updated: {updated}, new_version: {new_ver}")

# 10. Save outputs & small plot
stream_df.to_csv(out_dir / 'stream_sample.csv', index=False)
pd.DataFrame(local_predictions).to_csv(out_dir / 'local_predictions.csv', index=False)
if not cloud_df.empty:
    cloud_df.to_csv(out_dir / 'cloud_retrain_dataset.csv', index=False)

plt.figure(figsize=(8,3))
plt.plot(stream_df['ts'], stream_df['soil_moisture'], label='raw')
plt.plot(stream_df['ts'], stream_df['soil_moisture_sm'], label='smoothed')
plt.xlabel('time (s)'); plt.ylabel('soil moisture (%)'); plt.title('Sample Soil Moisture Stream')
plt.legend()
plt.grid(True)
plt.savefig(out_dir / 'soil_moisture_stream.png', dpi=200, bbox_inches='tight')
plt.close()

print("Prototype run complete. Check the 'edge_prototype' folder for outputs.")
