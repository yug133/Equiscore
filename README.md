# EquiScore: Fair and Explainable Credit Scoring in India for Applicants With No Credit History Using UPI and Financial Behaviour Data

EquiScore is a fair and explainable credit scoring system designed for thin-file applicants in India who lack traditional credit bureau records. By leveraging alternative financial behaviour data — such as UPI transaction regularity, income stability, and digital footprint signals — alongside fairness-constrained machine learning (XGBoost with Fairlearn), the system generates transparent, bias-audited credit scores (300–900). It provides SHAP-based explanations for loan officers and DiCE-powered counterfactual improvement tips for customers, enabling equitable access to credit while maintaining regulatory compliance and intersectional fairness across gender, region, and occupation subgroups.

---

## 📁 Folder Structure

```
equiscore/
├── backend/
│   ├── main.py                     # FastAPI app entry point
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Backend Docker config
│   ├── .env.example                # Environment variable template
│   ├── data/                       # Data loading, preprocessing, splitting
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── splitter.py
│   ├── features/                   # 5 engineered features + pipeline
│   │   ├── transaction_regularity.py   # TRS
│   │   ├── income_stability.py         # ISI
│   │   ├── payment_behaviour.py        # PBS
│   │   ├── digital_footprint.py        # DFS
│   │   ├── geo_income_index.py         # GII
│   │   └── feature_pipeline.py
│   ├── models/                     # ML models + evaluator
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_standard.py
│   │   ├── xgboost_fair.py
│   │   └── model_evaluator.py
│   ├── explainability/             # SHAP + DiCE explainers
│   │   ├── shap_explainer.py
│   │   ├── dice_explainer.py
│   │   └── consistency_scorer.py
│   ├── fairness/                   # Fairness auditing
│   │   ├── auditor.py
│   │   ├── intersectional.py
│   │   └── report_generator.py
│   ├── api/                        # FastAPI routes + schemas
│   │   ├── routes/
│   │   │   ├── predict.py
│   │   │   ├── audit.py
│   │   │   └── improve.py
│   │   ├── schemas.py
│   │   └── dependencies.py
│   ├── database/                   # PostgreSQL ORM + CRUD
│   │   ├── connection.py
│   │   ├── models.py
│   │   └── crud.py
│   └── utils/                      # Utilities
│       ├── score_scaler.py
│       └── logger.py
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   ├── officer/page.tsx
│   │   └── customer/page.tsx
│   ├── components/
│   │   ├── ScoreCard.tsx
│   │   ├── ShapWaterfall.tsx
│   │   ├── FairnessPanel.tsx
│   │   ├── DiceTips.tsx
│   │   ├── ApplicantForm.tsx
│   │   └── RiskBadge.tsx
│   └── lib/
│       ├── api.ts
│       └── types.ts
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 🚀 How to Run with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd equiscore

# Start all services (backend + frontend + database)
docker-compose up --build

# Services will be available at:
# - Backend API:  http://localhost:8000
# - Frontend:     http://localhost:3000
# - PostgreSQL:   localhost:5432
```

To stop all services:

```bash
docker-compose down
```

To remove all data (including database volume):

```bash
docker-compose down -v
```

---

## 🐍 How to Run Backend Locally

```bash
cd backend

# Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your PostgreSQL connection details

# Run the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API docs available at: http://localhost:8000/docs
```

---

## ⚛️ How to Run Frontend Locally

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev

# Open http://localhost:3000 in your browser
```

---

## 📡 API Endpoints

### POST `/predict`

Score a loan applicant and return credit decision with SHAP explanation.

**Request Body:**

```json
{
  "age": 35,
  "gender": "M",
  "income": 450000,
  "employment_type": "Salaried",
  "occupation_type": "Accountants",
  "education_type": "Higher education",
  "family_status": "Married",
  "housing_type": "House / apartment",
  "region_rating": 2,
  "own_car": true,
  "own_realty": true,
  "children_count": 1,
  "family_members": 3,
  "credit_amount": 500000,
  "annuity_amount": 25000,
  "goods_price": 450000,
  "ext_source_1": 0.5,
  "ext_source_2": 0.7
}
```

**Response:**

```json
{
  "application_id": "APP-20240101-001",
  "credit_score": 721,
  "default_probability": 0.12,
  "risk_level": "LOW",
  "shap_explanation": {
    "income": 0.15,
    "ext_source_2": 0.12,
    "TRS": 0.08
  },
  "top_factors": ["income", "ext_source_2", "TRS", "ISI", "PBS"]
}
```

---

### GET `/audit`

Return the latest fairness audit report.

**Response:**

```json
{
  "model_name": "xgboost_fair",
  "overall_metrics": {
    "auc_roc": 0.78,
    "gini": 0.56,
    "ks_statistic": 0.42
  },
  "fairness_metrics": {
    "dpg": { "gender_M": 0.72, "gender_F": 0.68 },
    "eod": { "gender_M": 0.05, "gender_F": 0.03 },
    "dir": { "overall": 0.94 }
  },
  "fairness_flags": []
}
```

---

### POST `/improve`

Generate counterfactual improvement tips for an applicant.

**Request Body:**

```json
{
  "application_id": "APP-20240101-001",
  "num_tips": 3
}
```

**Response:**

```json
{
  "application_id": "APP-20240101-001",
  "current_score": 480,
  "tips": [
    {
      "feature": "income",
      "current_value": 250000,
      "suggested_value": 320000,
      "impact": "Increase income by ₹70,000 to improve score"
    },
    {
      "feature": "TRS",
      "current_value": 0.4,
      "suggested_value": 0.7,
      "impact": "Improve transaction regularity through consistent UPI usage"
    }
  ],
  "potential_score": 620
}
```

---

### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

---

## 📊 Dataset Setup

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) dataset from Kaggle.

### Download Instructions

1. Sign up / log in to [Kaggle](https://www.kaggle.com)
2. Navigate to the [competition data page](https://www.kaggle.com/c/home-credit-default-risk/data)
3. Download all CSV files
4. Place them in `backend/data/raw/`:

```
backend/data/raw/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── credit_card_balance.csv
├── installments_payments.csv
├── POS_CASH_balance.csv
├── previous_application.csv
└── HomeCredit_columns_description.csv
```

---

## 👥 Team Member Responsibilities

| Member | Responsibility |
|--------|---------------|
| Member 1 | Data pipeline (loading, preprocessing, splitting), Feature engineering (TRS, ISI, PBS, DFS, GII) |
| Member 2 | Model training & evaluation (LR, RF, XGBoost, Fair XGBoost), Explainability (SHAP, DiCE, consistency scoring) |
| Member 3 | Fairness auditing (DPG, EOD, DIR, intersectional analysis), Backend API (FastAPI routes, schemas, database) |
| Member 4 | Frontend (Next.js pages, components, API integration), DevOps & documentation (Docker, CI/CD, README) |

---

## 📄 License

This project is for academic and research purposes.