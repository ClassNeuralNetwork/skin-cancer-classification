import shap
import numpy as np

def normalize_shap_values(shap_values, epilson=1e-8):
    normalized_shap = []
    for val in shap_values:
        min_val = np.min(val)
        max_val = np.max(val)

        range_val = max_val - min_val + epilson
        normalized_shap.append((val - min_val) / range_val)

    return np.array(normalized_shap)

background = train_ds[np.random.choice(train_ds.shape[0], 100, replace=False)]
explainer = shap.DeepExplainer(model, background)

shap_values = explainer.shap_values(test_ds[:10])

normalized_shap_values = normalize_shap_values(shap_values)