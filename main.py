from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────
app = FastAPI(
    title="Organ Matching AI Service",
    description="يستقبل بيانات المتبرع والمرضى من الـ Backend ويرجع أفضل المتطابقين.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

results_cache: dict[str, dict] = {}


# ─────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────

class DonorData(BaseModel):
    donor_id: str
    organ_type: str
    blood_type: str
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    BMI: float
    HLA_A_1: str
    HLA_A_2: str
    HLA_B_1: str
    HLA_B_2: str
    HLA_DR_1: str
    HLA_DR_2: str
    PRA: int
    CMV_status: str
    EBV_status: str
    donation_type: Optional[str] = None
    kdpi_score: Optional[float] = None


class RecipientData(BaseModel):
    recipient_id: str
    organ_needed: str
    blood_type: str
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    BMI: float
    HLA_A_1: str
    HLA_A_2: str
    HLA_B_1: str
    HLA_B_2: str
    HLA_DR_1: str
    HLA_DR_2: str
    PRA: int
    CMV_status: str
    EBV_status: str
    urgency_level: str
    waitlist_time_days: int
    dialysis_duration_days: Optional[float] = None
    MELD_score: Optional[float] = None
    lung_severity_score: Optional[float] = None


class MatchRequest(BaseModel):
    donor: DonorData
    recipients: list[RecipientData] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class MatchEntry(BaseModel):
    recipient_id: str
    score: float


class MatchResponse(BaseModel):
    donor_id: str
    top_matches: list[MatchEntry]


# ─────────────────────────────────────────
# Matching Logic
# ─────────────────────────────────────────

ABO_COMPATIBILITY = {
    "O": ["O", "A", "B", "AB"],
    "A": ["A", "AB"],
    "B": ["B", "AB"],
    "AB": ["AB"],
}

URGENCY_MAP = {
    "low": 0.1,
    "medium": 0.4,
    "high": 0.7,
    "critical": 1.0,
}


def is_abo_compatible(donor_bt: str, recip_bt: str) -> bool:
    if pd.isna(donor_bt) or pd.isna(recip_bt):
        return False

    donor_bt = str(donor_bt).strip().upper()
    recip_bt = str(recip_bt).strip().upper()

    return recip_bt in ABO_COMPATIBILITY.get(donor_bt, [])


def hla_match_score(donor: dict, recip: dict) -> float:
    matches = sum([
        donor.get("HLA_A_1") == recip.get("HLA_A_1"),
        donor.get("HLA_A_2") == recip.get("HLA_A_2"),
        donor.get("HLA_B_1") == recip.get("HLA_B_1"),
        donor.get("HLA_B_2") == recip.get("HLA_B_2"),
        donor.get("HLA_DR_1") == recip.get("HLA_DR_1"),
        donor.get("HLA_DR_2") == recip.get("HLA_DR_2"),
    ])
    return matches / 6


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)

    min_v = np.nanmin(s)
    max_v = np.nanmax(s)

    if np.isnan(min_v) or np.isnan(max_v) or max_v == min_v:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - min_v) / (max_v - min_v)


def run_matching(
    donor: DonorData,
    recipients: list[RecipientData],
    top_k: int
) -> list[dict]:

    donor_dict = donor.model_dump()
    donor_bt = donor.blood_type
    acceptable_organ = str(donor.organ_type).strip().lower()

    # 1) Exact organ match زي pro.ipynb
    same_organ = [
        r for r in recipients
        if str(r.organ_needed).strip().lower() == acceptable_organ
    ]

    if not same_organ:
        return []

    # 2) ABO filter زي pro.ipynb
    filtered = [
        r for r in same_organ
        if is_abo_compatible(donor_bt, r.blood_type)
    ]

    if not filtered:
        return []

    df = pd.DataFrame([r.model_dump() for r in filtered])

    # 3) Scores
    abo_scores = np.ones(len(df))

    hla_scores = np.array([
        hla_match_score(donor_dict, r)
        for r in df.to_dict("records")
    ])

    pra = pd.to_numeric(df["PRA"], errors="coerce").fillna(100).astype(float)
    immuno_scores = (1.0 - (pra / 100.0)).clip(0.0, 1.0).values

    urgency_scores = df["urgency_level"].apply(
        lambda u: URGENCY_MAP.get(str(u).strip().lower(), 0.0)
    ).values

    wait_norm = normalize_series(df["waitlist_time_days"]).values

    organ = acceptable_organ
    organ_specific_scores = np.zeros(len(df))

    if organ in ["kidney", "kidney_left", "kidney_right"]:
        organ_specific_scores = normalize_series(
            df["dialysis_duration_days"]
        ).values

    elif organ in ["liver", "liver_lobe"]:
        organ_specific_scores = normalize_series(
            df["MELD_score"]
        ).values

    elif "lung" in organ:
        organ_specific_scores = normalize_series(
            df["lung_severity_score"]
        ).values

    total_score = (
        0.35 * hla_scores +
        0.20 * abo_scores +
        0.10 * immuno_scores +
        0.15 * urgency_scores +
        0.10 * wait_norm +
        0.10 * organ_specific_scores
    )

    df["matching_score"] = total_score

    top = df.sort_values(
        "matching_score",
        ascending=False
    ).head(top_k)

    return [
        {
            "recipient_id": row["recipient_id"],
            "score": round(float(row["matching_score"]) * 100, 1),
        }
        for _, row in top.iterrows()
    ]


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.post(
    "/api/matching/",
    response_model=MatchResponse,
    summary="تشغيل المطابقة",
    description="يستقبل بيانات المتبرع + قائمة المرضى من الـ Backend ويرجع النتائج فوراً.",
)
def trigger_matching(req: MatchRequest):
    matches = run_matching(req.donor, req.recipients, req.top_k)

    results_cache[req.donor.donor_id] = {
        "donor_id": req.donor.donor_id,
        "top_matches": matches,
    }

    return MatchResponse(
        donor_id=req.donor.donor_id,
        top_matches=[MatchEntry(**m) for m in matches],
    )


@app.get(
    "/api/matching/{donor_id}",
    response_model=MatchResponse,
    summary="جلب آخر نتائج محفوظة",
    description="يرجع آخر نتائج مطابقة تم حسابها لمتبرع معين.",
)
def get_matching_results(donor_id: str):
    result = results_cache.get(donor_id)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for donor '{donor_id}'. Run POST /api/matching/ first.",
        )

    return MatchResponse(
        donor_id=donor_id,
        top_matches=[MatchEntry(**m) for m in result["top_matches"]],
    )


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "version": "2.0.0"
    }
