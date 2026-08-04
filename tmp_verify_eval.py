import sys
from pathlib import Path
import numpy as np
import joblib
import torch
sys.path.append(str(Path('.')))
from src.model import Autoencoder

base = Path('.')
proc = base / 'data' / 'processed'
models = base / 'models'

X_test = np.load(proc / 'X_test.npy')
y_test = np.load(proc / 'y_test.npy')
X_val = np.load(proc / 'X_val.npy')
y_val = np.load(proc / 'y_val.npy')
scaler = joblib.load(proc / 'scaler.joblib')

model = Autoencoder(input_dim=X_test.shape[1], bottleneck_dim=16)
model.load_state_dict(torch.load(models / 'autoencoder_licit.pth', map_location='cpu'))
model.eval()

with torch.no_grad():
    X_test_rec = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    X_val_rec = model(torch.tensor(X_val, dtype=torch.float32)).numpy()

X_test_orig = scaler.inverse_transform(X_test)
X_val_orig = scaler.inverse_transform(X_val)
X_test_rec_orig = scaler.inverse_transform(X_test_rec)
X_val_rec_orig = scaler.inverse_transform(X_val_rec)

err_test = np.mean(np.square(X_test_orig - X_test_rec_orig), axis=1)
err_val = np.mean(np.square(X_val_orig - X_val_rec_orig), axis=1)

print('TEST_STATS')
print('test_mean', round(float(err_test.mean()), 6))
print('test_median', round(float(np.median(err_test)), 6))
print('test_p95', round(float(np.percentile(err_test, 95)), 6))
print('test_licit_mean', round(float(err_test[y_test == 0].mean()), 6))
print('test_fraud_mean', round(float(err_test[y_test == 1].mean()), 6))
print('VAL_STATS')
print('val_mean', round(float(err_val.mean()), 6))
print('val_median', round(float(np.median(err_val)), 6))
print('val_p95', round(float(np.percentile(err_val, 95)), 6))
print('val_licit_mean', round(float(err_val[y_val == 0].mean()), 6))
print('val_fraud_mean', round(float(err_val[y_val == 1].mean()), 6))
print('THRESHOLD_CONFIG')
print(joblib.load(proc / 'threshold_config.joblib'))
