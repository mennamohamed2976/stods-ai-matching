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
    allow_origins=["*"],  # في Production: حدد الـ Django server URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# In-Memory Results Cache
# ─────────────────────────────────────────
results_cache: dict[str, dict] = {}


# ─────────────────────────────────────────
# Pydantic Schemas — بيانات المتبرع والمريض
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
    "O":  ["O", "A", "B", "AB"],
    "A":  ["A", "AB"],
    "B":  ["B", "AB"],
    "AB": ["AB"],
}

URGENCY_MAP = {
    "low": 0.1,
    "medium": 0.4,
    "high": 0.7,
    "critical": 1.0,
}

ORGAN_MAPPING = {
    "kidney":           ["kidney_left", "kidney_right"],
    "kidney_left":      ["kidney_left"],
    "kidney_right":     ["kidney_right"],
    "liver":            ["liver", "liver_lobe"],
    "liver_lobe":       ["liver_lobe", "liver"],
    "heart":            ["heart"],
    "lung_left":        ["lung_left", "lung_lobe"],
    "lung_right":       ["lung_right", "lung_lobe"],
    "lung_lobe":        ["lung_left", "lung_right", "lung_lobe"],
    "pancreas":         ["pancreas", "pancreas_segment"],
    "pancreas_segment": ["pancreas_segment", "pancreas"],
}


def is_abo_compatible(donor_bt: str, recip_bt: str) -> bool:
    return recip_bt.strip().upper() in ABO_COMPATIBILITY.get(donor_bt.strip().upper(), [])


def hla_match_score(donor: dict, recip: dict) -> float:
    matches = sum([
        donor.get("HLA_A_1")  == recip.get("HLA_A_1"),
        donor.get("HLA_A_2")  == recip.get("HLA_A_2"),
        donor.get("HLA_B_1")  == recip.get("HLA_B_1"),
        donor.get("HLA_B_2")  == recip.get("HLA_B_2"),
        donor.get("HLA_DR_1") == recip.get("HLA_DR_1"),
        donor.get("HLA_DR_2") == recip.get("HLA_DR_2"),
    ])
    return matches / 6


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    min_v, max_v = np.nanmin(s), np.nanmax(s)
    if np.isnan(min_v) or np.isnan(max_v) or max_v == min_v:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)


def run_matching(donor: DonorData, recipients: list[RecipientData], top_k: int) -> list[dict]:
    """
    نفس منطق pro.ipynb بالضبط:
    1) فلترة العضو Exact Match على donor.organ_type
    2) فلترة فصيلة الدم ABO
    3) حساب نفس الـ weights
    4) ترتيب تنازلي حسب matching_score

    ملاحظة:
    pro.ipynb يعرض matching_score كرقم من 0 إلى 1.
    الـ API يرجّعه كنسبة مئوية score من 0 إلى 100.
    """
    donor_dict = donor.model_dump()

    donor_bt = donor.blood_type
    acceptable_organ = str(donor.organ_type).lower()

    # مطابق لـ pro.ipynb:
    # subset = recipients_data[recipients_data["organ_needed"].str.lower() == acceptable_organ]
    same_organ = [
        r for r in recipients
        if str(r.organ_needed).lower() == acceptable_organ
    ]

    if not same_organ:
        return []

    # مطابق لـ pro.ipynb:
    # mask_abo = subset["blood_type"].apply(lambda bt: is_abo_compatible(donor_bt, bt))
    filtered = [
        r for r in same_organ
        if is_abo_compatible(donor_bt, r.blood_type)
    ]

    if not filtered:
        return []

    df = pd.DataFrame([r.model_dump() for r in filtered])

    # ── Scores مطابقين لـ compute_matching_scores_for_subset في pro.ipynb ──
    abo_scores = np.ones(len(df))

    hla_scores = np.array([
        hla_match_score(donor_dict, r)
        for r in df.to_dict("records")
    ])

    pra = df["PRA"].fillna(100).astype(float)
    immuno_scores = (1.0 - (pra / 100.0)).clip(0.0, 1.0).values

    urgency_scores = df["urgency_level"].apply(
        lambda u: URGENCY_MAP.get(str(u).lower(), 0.0)
    ).values

    wait_norm = normalize_series(df["waitlist_time_days"]).values

    organ = str(donor.organ_type).lower()
    organ_specific_scores = np.zeros(len(df))

    if organ in ["kidney", "kidney_left", "kidney_right"]:
        organ_specific_scores = normalize_series(
            df["dialysis_duration_days"].fillna(0)
        ).values

    elif organ in ["liver", "liver_lobe"]:
        organ_specific_scores = normalize_series(
            df["MELD_score"].fillna(0)
        ).values

    elif "lung" in organ:
        organ_specific_scores = normalize_series(
            df["lung_severity_score"].fillna(0)
        ).values

    W_HLA = 0.35
    W_ABO = 0.20
    W_IMMUNO = 0.10
    W_URGENCY = 0.15
    W_WAIT = 0.10
    W_ORGAN_SPEC = 0.10

    total_score = (
        W_HLA * hla_scores +
        W_ABO * abo_scores +
        W_IMMUNO * immuno_scores +
        W_URGENCY * urgency_scores +
        W_WAIT * wait_norm +
        W_ORGAN_SPEC * organ_specific_scores
    )

    df["matching_score"] = total_score
    top = df.sort_values("matching_score", ascending=False).head(top_k)

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

    # احفظ النتائج في الـ cache لجلبها لاحقاً بـ GET
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


# ─────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "2.0.0"}
