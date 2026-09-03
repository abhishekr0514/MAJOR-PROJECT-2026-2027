"""Preprocesses the downloaded PTB-XL/CPSC dataset.

Extracts patient IDs, converts records to physical unit (mV) by applying baseline
and gain values parsed dynamically from headers, downsamples ECGs from 500Hz to 100Hz,
assigns binary targets (Normal vs Abnormal), and compiles patient-independent splits.
"""

import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample


def parse_header_metadata(
    header_content: str,
) -> tuple[list[float], list[float], list[str], dict[str, str]]:
    """Parse WFDB header strings to retrieve gain, baseline, leads, and comments."""
    lines = [line.strip() for line in header_content.split("\n") if line.strip()]
    if not lines:
        return [], [], [], {}

    # Read first line details: e.g. "HR00001 12 500 5000 04-Jun-2020 15:11:55"
    first_line_parts = lines[0].split()
    num_leads = int(first_line_parts[1])

    gains = []
    baselines = []
    lead_names = []
    comments = {}

    # Signal description lines
    for idx in range(1, min(1 + num_leads, len(lines))):
        line = lines[idx]
        parts = line.split()
        if len(parts) >= 9:
            # Gain token: parts[2] e.g. '200/mV' or '200(0)/mV' or '200'
            gain_token = parts[2]
            gain_match = re.match(r"([0-9\.]+)(?:\((.*?)\))?", gain_token)
            gain = float(gain_match.group(1)) if gain_match else 200.0

            # Baseline token: parts[4]
            baseline = float(parts[4])
            lead_name = parts[8]

            gains.append(gain)
            baselines.append(baseline)
            lead_names.append(lead_name)

    # Comments lines
    for line in lines[1 + num_leads :]:
        if line.startswith("#"):
            comment_line = line[1:].strip()
            if ":" in comment_line:
                key, val = comment_line.split(":", 1)
                comments[key.strip()] = val.strip()

    return gains, baselines, lead_names, comments


def main() -> None:
    zip_path = "C:/Users/abhis/Downloads/archive (6).zip"
    db_csv_path = "ptbxl_database.csv"
    output_dir = "client/data"

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"ERROR: Required real dataset zip not found at: {zip_path}"
        )
    if not os.path.exists(db_csv_path):
        raise FileNotFoundError(
            f"ERROR: Required metadata CSV not found at: {db_csv_path}"
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading main database database CSV from {db_csv_path}...")
    df_db = pd.read_csv(db_csv_path)
    db_map = df_db.set_index("ecg_id").to_dict("index")

    print(f"Opening ZIP file: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        hea_files = sorted([n for n in names if n.endswith(".hea")])
        num_records = len(hea_files)
        print(f"Found {num_records} HEA/waveform records in dataset.")

        # Matrix for all signals: (21837 records, 12 leads, 1000 samples)
        all_signals = np.zeros((num_records, 12, 1000), dtype=np.float32)
        meta_rows = []

        for idx, hea_name in enumerate(hea_files):
            # Parse record index: e.g. WFDB/HR00001.hea -> 1
            filename_base = os.path.basename(hea_name).replace(".hea", "")
            ecg_id = int(re.sub(r"\D", "", filename_base))

            # Load header content
            hea_content = z.read(hea_name).decode("utf-8")
            gains, baselines, _, comments = parse_header_metadata(hea_content)

            # Load binary signal file
            mat_name = hea_name.replace(".hea", ".mat")
            mat_content = z.read(mat_name)
            mat_data = scipy.io.loadmat(io.BytesIO(mat_content))
            val = mat_data["val"]  # Shape (12, 5000)

            # Convert ADC value to physical units (mV)
            physical_val = np.zeros_like(val, dtype=np.float32)
            for ch in range(val.shape[0]):
                gain = gains[ch] if ch < len(gains) else 200.0
                baseline = baselines[ch] if ch < len(baselines) else 0.0
                physical_val[ch] = (val[ch].astype(np.float32) - baseline) / gain

            # Resample from 500 Hz (5000 samples) to 100 Hz (1000 samples)
            downsampled_val = resample(physical_val, 1000, axis=-1)
            all_signals[idx] = downsampled_val

            # Retrieve info from downloaded database CSV
            db_info = db_map.get(ecg_id, {})
            patient_id = db_info.get("patient_id", float(ecg_id))
            strat_fold = db_info.get("strat_fold", 1)

            # Retrieve Age / Sex
            age = db_info.get("age", float("nan"))
            if np.isnan(age):
                parsed_age = comments.get("Age", "")
                try:
                    age = (
                        float(parsed_age)
                        if parsed_age.lower() not in ("nan", "unknown", "")
                        else 60.0
                    )
                except ValueError:
                    age = 60.0

            sex = db_info.get("sex", -1)
            if sex == -1:
                parsed_sex = comments.get("Sex", "")
                sex = 1 if parsed_sex.lower() == "female" else 0

            # Parse diagnostic codes
            dx_str = comments.get("Dx", "")
            dx_codes = [c.strip() for c in dx_str.split(",") if c.strip()]

            # Binary Label Mapping:
            # Class 0: Normal ECG (only Sinus Rhythm code 426783006 present)
            # Class 1: Abnormal ECG (contains any diagnostic codes other than 426783006)
            has_sr = "426783006" in dx_codes
            has_others = len([c for c in dx_codes if c != "426783006"]) > 0
            label = 0 if (has_sr and not has_others) else 1

            meta_rows.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": patient_id,
                    "age": age,
                    "sex": sex,
                    "label": label,
                    "strat_fold": strat_fold,
                    "dx": ",".join(dx_codes),
                }
            )

            if (idx + 1) % 5000 == 0 or idx == num_records - 1:
                print(f"Processed {idx + 1}/{num_records} records...")

        # Save arrays and metadata
        out_ecg_path = os.path.join(output_dir, "ptbxl_ecg_100hz.npy")
        print(f"Saving signals matrix to {out_ecg_path}...")
        np.save(out_ecg_path, all_signals)

        out_meta_path = os.path.join(output_dir, "ptbxl_meta_100hz.csv")
        print(f"Saving metadata to {out_meta_path}...")
        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_csv(out_meta_path, index=False)

        # Print some summary metrics
        print("\n=== Data Preparation Summary ===")
        print(f"Signals file shape: {all_signals.shape}")
        print(f"Normal ECG records (Class 0): {len(df_meta[df_meta.label == 0])}")
        print(f"Abnormal ECG records (Class 1): {len(df_meta[df_meta.label == 1])}")
        print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
