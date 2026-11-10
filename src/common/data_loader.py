from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"


def load_dataset(filename: str) -> pd.DataFrame:
    for file in RAW_DATA_PATH.iterdir():
        if file.name.lower() == filename.lower():
           return pd.read_csv(
    file,
    na_values=["?", "NA", "N/A", "null", "NULL", ""]
)

    raise FileNotFoundError(
        f"{filename} not found.\nAvailable files: {[f.name for f in RAW_DATA_PATH.iterdir()]}"
    )


def load_all_data():
    datasets = {
        "assessments": load_dataset("assessments.csv"),
        "courses": load_dataset("courses.csv"),
        "student_assessments": load_dataset("StudentAssessments.csv"),
        "student_info": load_dataset("studentInfo.csv"),
        "student_registration": load_dataset("studentRegistration.csv"),
        "student_vle": load_dataset("studentVle.csv"),
        "vle": load_dataset("vle.csv"),
    }

    return datasets