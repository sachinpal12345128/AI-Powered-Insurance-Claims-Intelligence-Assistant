"""Extract — load raw CSV and apply column mapping."""
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "PolicyNumber": "claim_id",
    "PolicyType": "policy_type",
    "AccidentArea": "customer_region",
    "VehiclePrice": "claim_amount",
    "FraudFound_P": "fraud_label",
    "PastNumberOfClaims": "customer_history",
    "Fault": "fault",
    "Make": "vehicle_make",
    "VehicleCategory": "vehicle_category",
    "Sex": "sex",
    "MaritalStatus": "marital_status",
    "Age": "age",
    "PoliceReportFiled": "police_report_filed",
    "WitnessPresent": "witness_present",
    "AgentType": "agent_type",
    "NumberOfSuppliments": "number_of_supplements",
    "AddressChange_Claim": "address_change",
    "DriverRating": "driver_rating",
    "Deductible": "deductible",
    "NumberOfCars": "number_of_cars",
    "BasePolicy": "base_policy",
    "AgeOfVehicle": "age_of_vehicle",
    "AgeOfPolicyHolder": "age_of_policy_holder",
    "Month": "incident_month",
    "Year": "incident_year",
    "DayOfWeek": "incident_day",
}

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def extract(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {p.name}")

    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    if "incident_month" in df.columns and "incident_year" in df.columns:
        df["incident_date"] = (
            df["incident_year"].astype(str) + "-"
            + df["incident_month"].map(MONTH_MAP).fillna("01") + "-01"
        )

    if "claim_id" in df.columns:
        df["claim_id"] = "POL-" + df["claim_id"].astype(str)

    if "fraud_label" in df.columns:
        df["fraud_label"] = df["fraud_label"].astype(int)

    logger.info("Extraction complete.")
    return df
