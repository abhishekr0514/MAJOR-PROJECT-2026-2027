import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def generate_hospital_datasets(
    output_dir: str = "client/data",
    num_samples_per_hospital: list[int] | None = None,
    seed: int | None = 42,
) -> list[Path]:
    """Generate synthetic tabular, 12-lead ECG, and clinical text datasets for multi-hospital FL simulation.

    Args:
        output_dir: Target directory path for dataset files.
        num_samples_per_hospital: List of sample counts for [Hospital A, Hospital B, Hospital C].
        seed: Random seed for reproducibility.

    Returns:
        List of generated file Paths.
    """
    if seed is not None:
        np.random.seed(seed)

    if num_samples_per_hospital is None:
        num_samples_per_hospital = [150, 200, 180]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hospital_configs = [
        {"id": "hospital_a", "alias": "hospital_alpha", "label": "Hospital A"},
        {"id": "hospital_b", "alias": "hospital_beta", "label": "Hospital B"},
        {"id": "hospital_c", "alias": "hospital_gamma", "label": "Hospital C"},
    ]

    sample_names = [
        "John Doe",
        "Alice Smith",
        "Robert Johnson",
        "Emily Davis",
        "Michael Brown",
        "Sarah Wilson",
        "David Taylor",
        "Laura Martinez",
    ]
    sample_cities = [
        "New York",
        "Boston",
        "Chicago",
        "Seattle",
        "San Francisco",
        "Atlanta",
        "Dallas",
        "Miami",
    ]
    hospitals = [
        "St. Jude Hospital",
        "Mercy General Hospital",
        "City Health Medical Center",
        "Presbyterian Hospital",
    ]
    symptoms = [
        "Reports tightness in chest and shortness of breath.",
        "Experiencing acute exertional angina and mild dizziness.",
        "History of hypertension and dyspnea on exertion.",
        "Presents with atypical chest discomfort radiating to left arm.",
        "Asymptomatic routine checkup with elevated blood pressure.",
    ]

    generated_files: list[Path] = []

    for idx, config in enumerate(hospital_configs):
        h_id = config["id"]
        h_alias = config["alias"]
        h_label = config["label"]
        n_samples = num_samples_per_hospital[idx % len(num_samples_per_hospital)]

        # 1. Generate Synthetic Tabular Metrics
        patient_codes = [f"PAT-{h_id.upper()}-{i:04d}" for i in range(n_samples)]
        ages = np.random.randint(35, 80, size=n_samples)
        genders = np.random.choice(["M", "F"], size=n_samples)
        bp_sys = np.random.randint(110, 180, size=n_samples)
        bp_dia = np.random.randint(70, 110, size=n_samples)
        cholesterol = np.random.uniform(150, 320, size=n_samples).round(1)
        fasting_bs = np.random.uniform(80, 160, size=n_samples).round(1)
        chest_pain = np.random.choice([0, 1, 2, 3], size=n_samples)
        max_hr = np.random.randint(90, 195, size=n_samples)
        exercise_angina = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        diagnosis = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

        # 2. Add Synthetic Raw Clinical Text Notes with embedded PII
        raw_texts: list[str] = []
        for i in range(n_samples):
            name = np.random.choice(sample_names)
            city = np.random.choice(sample_cities)
            hosp = np.random.choice(hospitals)
            symptom = np.random.choice(symptoms)
            mrn = 100000 + i
            ssn = f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}"
            phone = f"{np.random.randint(200, 999)}-{np.random.randint(200, 999)}-{np.random.randint(1000, 9999)}"
            email = f"{name.lower().replace(' ', '.')}@example.com"
            date = (
                f"2026-0{np.random.randint(1, 9):01d}-{np.random.randint(10, 28):02d}"
            )

            text = (
                f"Patient {name} (MRN-{mrn}, SSN {ssn}) admitted at {hosp} in {city} on {date}. "
                f"Contact {email} or phone {phone}. {symptom} {h_label} record."
            )
            raw_texts.append(text)

        df = pd.DataFrame(
            {
                "patient_code": patient_codes,
                "age": ages,
                "gender": genders,
                "blood_pressure_sys": bp_sys,
                "blood_pressure_dia": bp_dia,
                "cholesterol_mg_dl": cholesterol,
                "fasting_bs_mg_dl": fasting_bs,
                "chest_pain_type": chest_pain,
                "max_heart_rate": max_hr,
                "exercise_angina": exercise_angina,
                "diagnosis": diagnosis,
                "raw_clinical_text": raw_texts,
            }
        )

        # 3. Save CSV Datasets (Primary and Alias)
        csv_path = out_path / f"{h_id}_data.csv"
        df.to_csv(csv_path, index=False)
        generated_files.append(csv_path)

        alias_csv_path = out_path / f"{h_alias}_data.csv"
        shutil.copyfile(csv_path, alias_csv_path)
        generated_files.append(alias_csv_path)

        # 4. Generate Synthetic 12-lead ECG Waveforms (n_samples, 12 leads, 1000 time-steps)
        ecg_signals = np.random.randn(n_samples, 12, 1000).astype(np.float32)
        ecg_path = out_path / f"{h_id}_ecg.npy"
        np.save(ecg_path, ecg_signals)
        generated_files.append(ecg_path)

        alias_ecg_path = out_path / f"{h_alias}_ecg.npy"
        shutil.copyfile(ecg_path, alias_ecg_path)
        generated_files.append(alias_ecg_path)

        print(
            f"✅ Generated dataset for {h_label} ({h_id} / {h_alias}): {n_samples} samples."
        )

    return generated_files


def main() -> None:
    """CLI entrypoint for generating synthetic multi-hospital dataset files."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-hospital patient datasets (tabular, ECG, text)."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="client/data",
        help="Target output directory for CSV and NPY files.",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=str,
        default="150,200,180",
        help="Comma-separated sample counts for Hospital A, B, C (e.g. 150,200,180).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data reproducibility.",
    )
    args = parser.parse_args()

    sample_counts = [int(x.strip()) for x in args.samples.split(",")]
    generate_hospital_datasets(
        output_dir=args.output_dir,
        num_samples_per_hospital=sample_counts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
