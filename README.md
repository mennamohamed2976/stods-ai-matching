# 🫀 Organ Matching AI Service — v2.0

الـ AI مش بيقرأ CSV — كل البيانات بتيجي من الـ Django Backend.

---

## ⚡ تشغيل الـ API

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🔄 الـ Flow الكامل

```
Flutter
  │
  ▼
POST /api/donors/          ← Django يحفظ المتبرع في الـ DB
  │
  ▼
Django يجيب كل المرضى من الـ DB
  │
  ▼
POST /api/matching/        ← Django يبعت بيانات المتبرع + المرضى للـ AI
  │                           الـ AI يحسب ويرجع النتائج فوراً (sync)
  ▼
Django يحفظ النتائج في الـ DB
  │
  ▼
GET /api/matching/{id}     ← Flutter تجيب النتائج
```

---

## 📡 الـ Endpoints

### POST `/api/matching/` — تشغيل المطابقة

**Request من الـ Django:**
```json
{
  "donor": {
    "donor_id": "D05762",
    "organ_type": "kidney_left",
    "blood_type": "O",
    "age": 61,
    "sex": "F",
    "height_cm": 163.6,
    "weight_kg": 59.7,
    "BMI": 22.3,
    "HLA_A_1": "A3", "HLA_A_2": "A1",
    "HLA_B_1": "B7", "HLA_B_2": "B35",
    "HLA_DR_1": "DR8", "HLA_DR_2": "DR11",
    "PRA": 25,
    "CMV_status": "negative",
    "EBV_status": "negative"
  },
  "recipients": [
    {
      "recipient_id": "R004146",
      "organ_needed": "kidney_left",
      "blood_type": "A",
      "age": 45,
      "sex": "M",
      "height_cm": 170.0,
      "weight_kg": 75.0,
      "BMI": 26.0,
      "HLA_A_1": "A3", "HLA_A_2": "A2",
      "HLA_B_1": "B7", "HLA_B_2": "B44",
      "HLA_DR_1": "DR8", "HLA_DR_2": "DR4",
      "PRA": 22,
      "CMV_status": "positive",
      "EBV_status": "negative",
      "urgency_level": "high",
      "waitlist_time_days": 1539,
      "dialysis_duration_days": 2325.0
    }
  ],
  "top_k": 5
}
```

**Response (200) — فوري:**
```json
{
  "donor_id": "D05762",
  "top_matches": [
    { "recipient_id": "R004146", "score": 78.6 },
    { "recipient_id": "R004571", "score": 77.2 }
  ]
}
```

---

### GET `/api/matching/{donor_id}` — جلب آخر نتائج

```json
{
  "donor_id": "D05762",
  "top_matches": [
    { "recipient_id": "R004146", "score": 78.6 }
  ]
}
```

---

## 🔧 إعداد Django

### settings.py
```python
AI_SERVICE_URL = "http://localhost:8000"  # أو URL الـ server الحقيقي
```

### استخدام الـ Service
```python
from matching.services.ai_service import run_matching_for_donor

# في الـ View أو بعد حفظ المتبرع
matches = run_matching_for_donor(donor_instance, top_k=5)
# matches = [{"recipient_id": "R001", "score": 88.5}, ...]
```

---

## 🧮 خوارزمية المطابقة

| المعيار | الوزن |
|---------|-------|
| HLA Matching (A, B, DR) | 35% |
| ABO Blood Compatibility | 20% |
| Urgency Level | 15% |
| Immunology (PRA) | 10% |
| Waitlist Time | 10% |
| Organ-Specific* | 10% |

*Organ-Specific: Kidney→dialysis days / Liver→MELD / Lung→severity score
