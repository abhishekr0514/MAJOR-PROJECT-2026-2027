import torch
from client.ml_models.lstm_model import ECGBiLSTM
from client.ml_models.tabular_model import TabularEncoder
from client.ml_models.text_model import BioClinicalBERTFeatureExtractor
from client.ml_models.gnn_fusion import GNNMultimodalFusion
from client.ml_models.full_model import MedShieldDiagnosticNet

print("--- MedShield FL ML Models Verification ---")

# 1. ECGBiLSTM
print("\n[lstm_model.py: ECGBiLSTM]")
ecg_model = ECGBiLSTM(in_channels=12, hidden_dim=64, embedding_dim=128)
ecg_model.eval()
x_ecg = torch.randn(2, 12, 1000) # batch_size=2, 12 leads, 1000 timestamp sequence length
with torch.no_grad():
    y_ecg = ecg_model(x_ecg)
print(f"Input shape: {x_ecg.shape}")
print(f"Output shape: {y_ecg.shape}")
print(f"Output sample (first batch):\n{y_ecg[0][:8]}... (truncated)")

# 2. TabularEncoder
print("\n[tabular_model.py: TabularEncoder]")
tab_model = TabularEncoder(num_features=10, hidden_dim=64, output_dim=64)
tab_model.eval()
x_tab = torch.randn(2, 10) # batch_size=2, 10 features
with torch.no_grad():
    y_tab = tab_model(x_tab)
print(f"Input shape: {x_tab.shape}")
print(f"Output shape: {y_tab.shape}")
print(f"Output sample (first batch):\n{y_tab[0][:8]}... (truncated)")

# 3. BioClinicalBERTFeatureExtractor
print("\n[text_model.py: BioClinicalBERTFeatureExtractor]")
text_model = BioClinicalBERTFeatureExtractor(output_dim=128) # Project from 768 to 128
text_model.eval()
input_ids = torch.randint(0, 1000, (2, 32)) # batch_size=2, 32 sequence length tokens
attention_mask = torch.ones(2, 32)
with torch.no_grad():
    y_text = text_model(input_ids, attention_mask)
print(f"Input shape (input_ids): {input_ids.shape}")
print(f"Output shape: {y_text.shape}")
print(f"Output sample (first batch):\n{y_text[0][:8]}... (truncated)")

# 4. GNNMultimodalFusion
print("\n[gnn_fusion.py: GNNMultimodalFusion]")
fusion_model = GNNMultimodalFusion(ecg_dim=128, text_dim=128, tab_dim=64, num_classes=2)
fusion_model.eval()
with torch.no_grad():
    y_logits = fusion_model(y_ecg, y_text, y_tab)
    y_prob = fusion_model.predict_proba(y_ecg, y_text, y_tab)
print(f"Input shapes: ECG={y_ecg.shape}, Text={y_text.shape}, Tabular={y_tab.shape}")
print(f"Output Logits shape: {y_logits.shape}")
print(f"Output Probabilities shape: {y_prob.shape}")
print(f"Logits:\n{y_logits}")
print(f"Probabilities:\n{y_prob}")

# 5. MedShieldDiagnosticNet
print("\n[full_model.py: MedShieldDiagnosticNet]")
full_model = MedShieldDiagnosticNet(
    ecg_channels=12,
    tab_features=10,
    text_output_dim=128,
    ecg_output_dim=128,
    tab_output_dim=64,
    num_classes=2
)
full_model.eval()
with torch.no_grad():
    y_full = full_model(x_ecg, input_ids, attention_mask, x_tab)
print(f"Input shapes: ECG={x_ecg.shape}, InputIDs={input_ids.shape}, Tabular={x_tab.shape}")
print(f"Output Logits shape: {y_full.shape}")
print(f"Logits:\n{y_full}")
