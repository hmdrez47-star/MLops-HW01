# تمرین اول MLOps — پیش‌بینی ریزش مشتری (راهنمای فارسی)

<div dir="rtl">

پروژه‌ی کامل تمرین اول: داده و نسخه‌بندی، آموزش مدل، آزمایش‌های MLflow، تست،
سرو کردن مدل با Docker و در انتها drift و CI/CD.

- هدف: ستون `churn` (۰ = مانده، ۱ = رفته) با ۱۲ ویژگی عددی
- معیار اصلی: **F1** (داده نامتوازن است، حدود ۲۶٪ مثبت)
- راهنمای انگلیسی: [README.md](README.md)
- دنبال دستورها هستی؟ همه در بخش
  **[۱۱) چطور اجرا و تست کنم](#۱۱-چطور-اجرا-و-تست-کنم)** در انتهای همین فایل است.

---

## ۱) ساختار پروژه

| مسیر | کار |
| --- | --- |
| `config/config.yaml` | همه‌ی پارامترها؛ هیچ مقدار مهمی در کد hard-code نشده |
| `data/raw/` | `dataset_v1.csv` (۴۰۰ ردیف) و `dataset_v2.csv` (۴۸۰ ردیف) |
| `data_versions/` | `metadata.json` + snapshot هر نسخه |
| `src/data/load_data.py` | بارگذاری، اعتبارسنجی، تقسیم، هش، نسخه‌بندی |
| `src/features/build_features.py` | مقیاس‌بندی، انتخاب ویژگی، pipeline |
| `src/models/train_model.py` | ساخت، آموزش، ارزیابی، ذخیره و بارگذاری مدل |
| `src/monitoring/drift_detection.py` | لاگ ساختاریافته و تشخیص drift |
| `src/serving/app.py` | FastAPI: `/health`, `/predict`, `/metadata` |
| `experiments/train_experiment.py` | اجرای کامل pipeline + MLflow |
| `experiments/run_experiments.py` | پنج run + جدول مقایسه |
| `tests/` | ۶۳ تست pytest |
| `scripts/` | گزارش drift و healthcheck |
| `models/`، `reports/` | خروجی مدل و گزارش drift |
| `Dockerfile`، `.github/workflows/ci.yml` | ایمیج و CI/CD |

## ۲) داده

| نسخه | ردیف | نرخ churn | کاربرد |
| --- | --- | --- | --- |
| v1 | ۴۰۰ | ۲۶٪ | داده‌ی مرجع آموزش |
| v2 | ۴۸۰ | ۴۸٪ | توزیع جابه‌جا شده، برای سنجش drift |

هش هر نسخه در `data_versions/metadata.json` ثبت می‌شود (`v1`:
`446b605d5006…`، `v2`: `6bd52cc55b6c…`). همان هش در `models/metadata.json`
هم ذخیره می‌شود، پس هر مدل تا بایت‌های دقیق داده‌اش قابل ردیابی است و
`verify_data_version` می‌تواند صحت داده را دوباره بررسی کند.

## ۳) پیش‌پردازش

۱. جدا کردن `churn` به‌عنوان `y` و ۱۲ ستون دیگر به‌عنوان `X`
۲. تقسیم stratified با `test_size=0.2` و `random_state=42`
۳. مقیاس‌بندی `StandardScaler` (یا `none` برای مدل درختی)
۴. انتخاب ۱۰ ویژگی با `SelectKBest(f_classif)` → حذف `num_services` و `age`
۵. آموزش مدل و گزارش accuracy و F1

دو نکته‌ی مهم:

- **scaler و selector فقط روی داده‌ی آموزش fit می‌شوند** (جلوگیری از نشت داده).
- همه‌ی مدل‌ها `class_weight="balanced"` دارند. بدون آن جنگل تصادفی همه را
  «ماندن» پیش‌بینی می‌کند: accuracy حدود ۰٫۷۰ ولی F1 حدود ۰٫۰۸. همین دلیل
  اصلی انتخاب F1 به‌عنوان معیار است.

## ۴) نتایج

| نام run | داده | مدل | مقیاس | k | Accuracy | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| `lr_v2_standard` | v2 | logistic_regression | standard | 10 | 0.6979 | **0.6947** |
| `svm_v1_standard` | v1 | svm | standard | 10 | 0.6875 | 0.5098 |
| `lr_v1_k8` | v1 | logistic_regression | standard | 8 | 0.6500 | 0.5000 |
| `lr_v1_standard` | v1 | logistic_regression | standard | 10 | 0.6125 | 0.4746 |
| `rf_v1_none` | v1 | random_forest | none | 10 | 0.5250 | 0.2083 |

بالاترین F1 مربوط به run روی v2 است، ولی مقایسه منصفانه نیست چون v2 توزیع
آسان‌تری دارد. روی داده‌ی مرجع v1 ترتیب واقعی: **SVM > رگرسیون لجستیک > جنگل
تصادفی**. مدلی که سرو می‌شود رگرسیون لجستیک روی v1 است (پیش‌فرض config، بدون
نیاز به کالیبراسیون احتمال و قابل تفسیر).

## ۵) ردیابی آزمایش‌ها (MLflow)

هر run این موارد را log می‌کند: پارامترها (نسخه و هش داده، نوع مدل،
مقیاس‌بندی، k، `test_size`، `random_state` و همه‌ی hyperparameterها)، متریک‌ها
(accuracy، F1، precision، recall، زمان آموزش، متریک روی v2)، tagها و
artifactها (`model.joblib`، `preprocessor.joblib`، `metadata.json`، خلاصه‌ی
run). بک‌اند یک فایل sqlite محلی است تا پروژه آفلاین هم کامل کار کند.

## ۶) آرتیفکت مدل

`models/model.joblib` + `models/preprocessor.joblib` +
`models/metadata.json`. preprocessor همراه مدل ذخیره می‌شود، پس API همان ۱۲
ویژگی خام را می‌گیرد و دقیقاً همان مقیاس‌بندی و انتخاب ویژگی آموزش را اعمال
می‌کند.

## ۷) مانیتورینگ و Drift

لاگ هر پیش‌بینی فقط فیلدهای عملیاتی دارد؛ هر کلیدی که شبیه داده‌ی حساس باشد
قبل از نوشتن حذف می‌شود:

```json
{"timestamp": "2026-11-06T09:41:12+00:00", "latency": 0.002241,
 "prediction": 1, "confidence": 0.613, "model_version": "logistic_regression-v1-2232"}
```

سه سطح تشخیص drift: `detect_drift_ks` (آزمون KS دو نمونه‌ای)،
`detect_drift_mean_shift` (تغییر نسبی میانگین) و `detect_feature_drift`
(ترکیب هر دو؛ `drift_score` = سهم ویژگی‌های drift کرده). آستانه‌ی هشدار طبق
صورت تمرین `drift_score > 0.3` است.

نتیجه‌ی مقایسه v1 با v2: **drift رخ داده**، `drift_score = 0.3333` (۴ از ۱۲
ویژگی)، شدت **متوسط**. مهم‌ترین ویژگی‌ها: `support_tickets` (+۱۰۸٪)،
`monthly_charges` (+۲۳٪)، `total_charges` (+۳۲٪)، `credit_score` (−۳٪).
اقدام پیشنهادی: بررسی خط تولید داده برای همین ویژگی‌ها، مقایسه‌ی F1 آفلاین با
F1 سرویس و برنامه‌ریزی آموزش مجدد.

## ۸) سرویس

سه endpoint: `GET /health`، `POST /predict` (۱۲ ویژگی خام → prediction،
confidence، latency) و `GET /metadata`. تعداد ویژگی اشتباه → ۴۰۰، ورودی
غیرعددی → ۴۲۲، مدل بارگذاری‌نشده → ۵۰۳. در پاسخ هیچ‌وقت ویژگی‌های خام تکرار
نمی‌شوند.

## ۹) Docker و CI/CD

ایمیج بر پایه‌ی `python:3.12-slim` است، فقط `requirements-serve.txt` را نصب
می‌کند، با کاربر غیرروت (UID 10001) اجرا می‌شود و `HEALTHCHECK` روی `/health`
دارد. چون پوشه‌ی `models/` داخل ایمیج کپی می‌شود، قبل از build باید مدل آموزش
دیده باشد.

CI در `.github/workflows/ci.yml`: کیفیت کد (ruff) → تست (pytest) → آموزش
(`run_experiments.py`) → بررسی drift (شکست در صورت هشدار) → smoke test داکر.

## ۱۰) محدودیت‌ها

- داده فقط ۴۰۰ ردیف است؛ اختلاف F1 بین SVM و رگرسیون روی ۸۰ نمونه‌ی تست در
  محدوده‌ی نوسان آماری است.
- `SVC(probability=True)` در sklearn 1.9 یک `FutureWarning` می‌دهد (کار می‌کند).
- مرجع drift یک snapshot ثابت است؛ در محیط واقعی باید پنجره‌ی لغزان باشد.
- بک‌اند MLflow یک فایل sqlite محلی است، نه سرور مشترک.

---

## ۱۱) چطور اجرا و تست کنم

همه‌ی دستورها داخل پوشه‌ی `Project` اجرا می‌شوند. پیش‌نیاز: پایتون ۳٫۱۱ یا
بالاتر (روی ۳٫۱۴ تست شده) و برای مرحله‌ی داکر، Docker.

```bash
cd Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### قدم ۱ — آیا کد سالم است؟ (تست، بدون آموزش)

```bash
pytest tests/ -q                 # ۶۳ تست، حدود ۵ ثانیه
pytest tests/ -v                 # همان تست‌ها با نام
pytest tests/test_data.py -q     # فقط یک ماژول
```

خروجی درست: `63 passed`. تست‌ها از دو فایل CSV داخل `data/raw/` استفاده می‌کنند.

### قدم ۲ — آموزش مدل و دیدن نتیجه

```bash
python experiments/train_experiment.py
```

تنها ورودی اصلی کل pipeline است (بارگذاری → تقسیم → مقیاس‌بندی → انتخاب ویژگی
→ آموزش → ارزیابی → ذخیره → MLflow) و در پایان خلاصه را چاپ می‌کند:

```
accuracy   : 0.6125
f1_score   : 0.4746   <-- معیار اصلی
v2 eval    : accuracy=0.6188 f1=0.6839 (distribution shift)
artifact   : .../Project/models/model.joblib
mlflow id  : dff75f2800884efd9710bb003e140862
```

فایل‌های ساخته‌شده: `models/model.joblib`، `models/preprocessor.joblib`،
`models/metadata.json`، یک ورودی جدید در `data_versions/metadata.json` و یک
run در `mlflow.db`. بدون دست زدن به YAML هم می‌توان پارامترها را عوض کرد:

```bash
python experiments/train_experiment.py --model-type svm
python experiments/train_experiment.py --model-type random_forest --scaling none
python experiments/train_experiment.py --data-file dataset_v2.csv --data-version v2
python experiments/train_experiment.py --no-mlflow       # بدون ثبت در MLflow
```

### قدم ۳ — دسته‌ی آزمایش‌ها و جدول مقایسه

```bash
python experiments/run_experiments.py
```

پنج تنظیم مختلف را اجرا و در MLflow ثبت می‌کند و
`experiments/results/comparison.md` و `.json` را می‌سازد.

### قدم ۴ — دیدن runها

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db    # http://localhost:5000
```

### قدم ۵ — گزارش drift

```bash
python scripts/drift_report.py                  # reports/drift_report.json و .md
python scripts/drift_report.py --fail-on-drift  # کد خروج ۱ اگر drift > 0.3
```

### قدم ۶ — سرو کردن مدل و تست آن

```bash
uvicorn src.serving.app:app --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"features":[22.85,74.43,1570.81,1,2,28.11,12.89,12,8.43,698,46,3]}'
# {"prediction":1,"confidence":0.581,"latency":0.023}
```

هر درخواست یک خط JSON در `experiments/logs/predictions_YYYY-MM-DD.json` ثبت
می‌کند.

### قدم ۷ — Docker

```bash
docker build -t mlops-serving:hw01 .
docker run -p 8000:8000 mlops-serving:hw01
python scripts/healthcheck.py --base-url http://127.0.0.1:8000
```

</div>
