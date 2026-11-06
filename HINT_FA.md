# راهنمای کامل فارسی — تمرین اول MLOps

<div dir="rtl">

این فایل توضیح می‌دهد که در پوشه‌ی `ANSWER` دقیقاً چه چیزی ساخته شد، هر بخش
برای چه هدفی است، چطور اجرا می‌شود و نتیجه‌ی نهایی چه بود. همه‌ی مراحل تمرین
(ساختاردهی Git و Reproducibility، دیتاست و آموزش مدل، نسخه‌بندی داده و آزمایش
به‌همراه تست‌نویسی، Model Serving و Docker، و در انتها Monitoring و Drift و
CI/CD) پیاده‌سازی شده است.

---

## ۱) ساختار پروژه

ساختار خواسته‌شده در صورت تمرین عیناً پیاده شد و هر بخش یک ماژول جدا دارد:

- `config/config.yaml` — تنها محل نگه‌داری پارامترها (هیچ مقدار مهمی در کد
  hard-code نشده است).
- `data/raw/` — داده‌ی خام `dataset_v1.csv` (۴۰۰ ردیف) و `dataset_v2.csv`
  (۴۸۰ ردیف).
- `data/processed/` — محل ذخیره‌ی خروجی‌های پیش‌پردازش.
- `data_versions/` — `metadata.json` به‌همراه snapshot هر نسخه‌ی داده.
- `src/data/load_data.py` — بارگذاری، اعتبارسنجی، تقسیم داده، هش و نسخه‌بندی.
- `src/features/build_features.py` — مقیاس‌بندی و انتخاب ویژگی.
- `src/models/train_model.py` — ساخت، آموزش، ارزیابی و ذخیره‌ی مدل.
- `src/monitoring/drift_detection.py` — لاگ ساختاریافته و تشخیص drift.
- `src/serving/app.py` — سرویس FastAPI.
- `experiments/train_experiment.py` — اجرای کل pipeline بر اساس config.
- `experiments/run_experiments.py` — اجرای دسته‌ای آزمایش‌ها و ساخت جدول مقایسه.
- `tests/` — ۶۲ تست pytest.
- `models/` — خروجی مدل: `model.joblib`، `preprocessor.joblib` و
  `metadata.json`.
- `reports/` — گزارش drift به‌صورت JSON و Markdown.
- `Dockerfile` و `requirements.txt` و `requirements-serve.txt` و
  `.github/workflows/ci.yml`.

## ۲) Git و قابلیت بازتولید (Reproducibility)

- فایل `.gitignore` کامل شد: داده‌ی خام و پردازش‌شده، فایل‌های مدل
  (`*.joblib`)، محیط مجازی، `mlruns/`، دیتابیس MLflow، لاگ‌ها، گزارش‌ها و
  فایل‌های حساس (`.env`، `*.pem`، `*.key`) هیچ‌وقت commit نمی‌شوند.
- فایل `README.md` نوشته شد: توضیح ساختار، دیتاست، روش پیش‌پردازش، دستورهای
  اجرا، جدول نتایج و نحوه‌ی سرو کردن مدل.
- محیط مجازی ساخته شد و `requirements.txt` با **نسخه‌های دقیق** تولید شد
  (مثلاً `pandas==3.1.1`، `scikit-learn==1.9.1`، `mlflow==3.18.0`،
  `fastapi==0.162.0`). یک فایل جداگانه‌ی `requirements-serve.txt` هم برای
  ایمیج Docker ساخته شد تا فقط وابستگی‌های لازم برای اجرای سرویس نصب شوند.
- در `metadata.json` مدل، نسخه‌ی sklearn، نسخه‌ی پایتون، هش داده و commit گیت
  ثبت می‌شود تا هر مدل قابل ردیابی باشد.
- حداقل سه commit معنادار در Git ساخته شد (در مجموع بیش از ده commit منطقی:
  ساختار، بارگذاری داده، ویژگی‌ها، مدل، pipeline، مانیتورینگ، سرویس، تست،
  CI و مستندات).

## ۳) ماژولار بودن کد

کد اولیه که فقط `TODO` داشت به توابع مستقل و تست‌پذیر تقسیم شد:

- **بارگذاری داده:** `load_config`, `load_csv`, `validate_schema`,
  `split_data`, `compute_hash`, `create_data_version`,
  `verify_data_version`.
- **استخراج ویژگی:** `scale_features`, `select_features`,
  `build_feature_pipeline`, `transform_features`, `get_feature_scores`.
- **آموزش مدل:** `get_model`, `train_model`.
- **ارزیابی مدل:** `evaluate_model` (accuracy، F1، precision، recall).
- **ذخیره‌سازی:** `save_model`, `load_model`, `load_preprocessor`.

نکته‌ی مهم: `scaler` و `selector` فقط روی **داده‌ی آموزش** fit می‌شوند. این
کار جلوی نشت اطلاعات (data leakage) از مجموعه‌ی تست را می‌گیرد و در تست‌ها هم
صریحاً بررسی شده است.

## ۴) تنظیمات (config)

همه‌ی مقادیری که قبلاً hard-code بودند به `config/config.yaml` منتقل شدند:
`test_size=0.2`، `random_state=42`، مسیر داده، `scaling`،
`feature_selection_k=10`، نوع مدل، تمام hyperparameterها، آدرس و نام آزمایش
MLflow، آستانه‌های drift و پورت سرویس. با آرگومان‌های خط فرمان هم می‌توان
مقادیر را موقتاً تغییر داد:

```bash
python experiments/train_experiment.py --model-type svm --scaling standard --k 10
python experiments/train_experiment.py --data-file dataset_v2.csv --data-version v2
```

## ۵) دیتاست و پیش‌پردازش

- ستون هدف `churn` است: صفر یعنی مشتری مانده، یک یعنی رفته. ۱۲ ستون ویژگی
  عددی وجود دارد.
- توزیع کلاس‌ها متعادل نیست (حدود ۲۶٪ مثبت)، پس معیار اصلی **F1** انتخاب شد
  و همه‌ی مدل‌ها `class_weight="balanced"` دارند.
- تقسیم داده به‌صورت stratified انجام می‌شود تا نسبت کلاس‌ها در train و test
  یکسان بماند.
- مقیاس‌بندی با `StandardScaler` (برای رگرسیون لجستیک و SVM) و امکان
  `none` برای مدل درختی.
- انتخاب ویژگی با `SelectKBest(f_classif)` و `k=10`؛ دو ویژگی کم‌اثر
  (`num_services` و `age`) حذف می‌شوند.

### یک نکته‌ی مهم و واقعی از این دیتاست

اگر مدل جنگل تصادفی را با تنظیمات پیش‌فرض آموزش بدهید، دقت (accuracy) حدود
۰٫۷۰ می‌شود ولی F1 فقط حدود ۰٫۰۸! دلیلش این است که مدل تقریباً همه را «ماندن»
پیش‌بینی می‌کند و چون ۷۴٪ داده منفی است، دقت بالا به‌نظر می‌رسد. با اضافه کردن
`class_weight="balanced"` این مشکل تا حد زیادی کم می‌شود و F1 واقعی‌تر می‌شود.
این دقیقاً همان دلیلی است که صورت تمرین گفته معیار F1 اهمیت بیشتری دارد.

## ۶) نتایج آزمایش‌ها (MLflow)

در آزمایش `mlops_homework_01` شش run ثبت شد. هر run حداقل در یکی از سه مورد
«نسخه‌ی داده»، «نوع مدل» یا «hyperparameter» تفاوت دارد:

| نام run | داده | مدل | مقیاس‌بندی | k | Accuracy | F1 |
|---|---|---|---|---|---|---|
| `lr_v2_standard` | v2 | logistic_regression | standard | 10 | 0.6979 | **0.6947** |
| `svm_v1_standard` | v1 | svm (rbf) | standard | 10 | 0.6875 | 0.5098 |
| `lr_v1_k8` | v1 | logistic_regression | standard | 8 | 0.6500 | 0.5000 |
| `lr_v1_standard` | v1 | logistic_regression | standard | 10 | 0.6125 | 0.4746 |
| `lr_v1_deployed` | v1 | logistic_regression | standard | 10 | 0.6125 | 0.4746 |
| `rf_v1_none` | v1 | random_forest | none | 10 | 0.5250 | 0.2083 |

برای هر run این موارد log می‌شود:

- **پارامترها:** نسخه و هش داده، نوع مدل، مقیاس‌بندی، k، `test_size`،
  `random_state` و تمام hyperparameterها.
- **متریک‌ها:** accuracy، F1، precision، recall، زمان آموزش و متریک روی
  نسخه‌ی دوم داده.
- **Artifactها:** `model.joblib`، `preprocessor.joblib`، `metadata.json` و
  خلاصه‌ی run.

نکته‌ی تفسیری: بالاترین F1 مربوط به run روی داده‌ی v2 است، ولی مقایسه‌ی منصفانه
نیست چون v2 توزیع متفاوت و آسان‌تری دارد (۴۸٪ مثبت). روی داده‌ی مرجع v1
ترتیب واقعی این است: **SVM با ۰٫۵۰۹۸، رگرسیون لجستیک با ۰٫۴۷۴۶ و جنگل
تصادفی با ۰٫۲۰۸۳**.

مدل نهایی که سرو می‌شود، رگرسیون لجستیک روی v1 است، چون در config پیش‌فرض
است، به کالیبراسیون احتمال اضافه نیاز ندارد و ضرایبش قابل تفسیر است. عوض کردن
مدل سرو شده فقط یک خط تغییر در config و یک بار آموزش مجدد است.

دیدن رابط کاربری MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db
```

## ۷) نسخه‌بندی داده

تابع `create_data_version` برای هر نسخه یک ورودی در
`data_versions/metadata.json` می‌نویسد که شامل `version_id`، `hash`،
`description`، `n_rows` و `created_at` است و یک snapshot از داده هم ذخیره
می‌کند:

- نسخه‌ی `v1` (۴۰۰ ردیف): `446b605d5006…ad7a`
- نسخه‌ی `v2` (۴۸۰ ردیف): `6bd52cc55b6c…8162`

هش از SHA-256 روی CSV قانونی‌شده محاسبه می‌شود؛ پس همیشه می‌توان بررسی کرد که
داده هنوز همان داده است یا نه (`verify_data_version`). چون هش داده در
`metadata.json` مدل هم ذخیره می‌شود، هر مدل تا بایت‌های دقیق داده‌اش قابل
ردیابی است.

## ۸) تست‌نویسی با pytest

۶۲ تست نوشته شد (همه پاس می‌شوند) که این موارد را پوشش می‌دهند:

- **داده:** بارگذاری فایل موجود، خطای `FileNotFoundError` برای فایل ناموجود،
  فایل خالی، هش تکرارپذیر و حساس به تغییر داده، اندازه‌ی صحیح train/test،
  تکرارپذیری split، ساخت و اعتبارسنجی `metadata.json`.
- **ویژگی‌ها:** fit شدن فقط روی داده‌ی آموزش، درستی بازه‌ی MinMax، حالت
  `none`، تعداد ویژگی‌های انتخاب‌شده و خطا در روش نامعتبر.
- **مدل:** ساخت هر سه مدل، خطای نوع مدل نامعتبر، آموزش، متریک‌های ارزیابی،
  ذخیره و بارگذاری مدل و preprocessor.
- **مانیتورینگ:** عدم drift روی داده‌ی یکسان، تشخیص drift روی داده‌ی جابه‌جا
  شده، ساختار خروجی KS، mean shift، **حذف اطلاعات حساس از لاگ**، محتوای
  گزارش drift و پاک نشدن فیلدهای حساس.
- **سرویس:** سلامتی، متادیتا، پیش‌بینی درست، رد کردن تعداد ویژگی اشتباه
  (کد ۴۰۰) و ورودی غیرعددی (کد ۴۲۲)، ثبت لاگ بدون ویژگی خام.

اجرا:

```bash
pytest tests/ -v
```

## ۹) سرو کردن مدل (FastAPI)

مدل، preprocessor و metadata در startup بارگذاری می‌شوند. سه endpoint داریم:

- `GET /health` — وضعیت سرویس، نسخه‌ی مدل، نسخه‌ی داده و زمان بالارفتن.
- `POST /predict` — ۱۲ ویژگی خام می‌گیرد، همان مقیاس‌بندی و انتخاب ویژگی
  آموزش را اعمال می‌کند و `prediction`، `confidence` و `latency` برمی‌گرداند.
- `GET /metadata` — متادیتای مدل در حال سرو.

اجرا و آزمایش:

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features":[22.85,74.43,1570.81,1,2,28.11,12.89,12,8.43,698,46,3]}'
```

اگر length بردار اشتباه باشد کد ۴۰۰، اگر مقدار غیرعددی باشد کد ۴۲۲ و اگر مدل
بارگذاری نشده باشد کد ۵۰۳ برمی‌گردد. در پاسخ هیچ‌وقت ویژگی‌های خام تکرار
نمی‌شوند.

## ۱۰) Docker

- ایمیج بر پایه‌ی `python:3.12-slim` ساخته می‌شود و فقط
  `requirements-serve.txt` را نصب می‌کند (بدون mlflow و pytest) تا سبک
  بماند.
- سرویس با کاربر غیرروت (`appuser` با UID 10001) اجرا می‌شود.
- `HEALTHCHECK` روی `/health` تعریف شده و پورت ۸۰۰۰ expose شده است.
- فایل `.dockerignore` جلوی کپی شدن داده، تست، مدل‌های اضافی و فایل‌های حساس
  را می‌گیرد.

```bash
docker build -t mlops-serving:hw01 .
docker run -p 8000:8000 mlops-serving:hw01
python scripts/healthcheck.py --base-url http://127.0.0.1:8000
```

توجه: قبل از build باید مدل آموزش دیده باشد، چون ایمیج پوشه‌ی `models/` را
کپی می‌کند.

## ۱۱) مانیتورینگ و لاگ ساختاریافته

برای هر پیش‌بینی یک خط JSON در فایل روزانه
`experiments/logs/predictions_YYYY-MM-DD.json` نوشته می‌شود:

```json
{"timestamp": "2026-11-06T09:41:12.418+00:00", "latency": 0.002241, "prediction": 1, "confidence": 0.613, "model_version": "logistic_regression-v1-2072"}
```

فقط فیلدهای عملیاتی مجاز هستند (`SAFE_LOG_KEYS`). هر کلیدی که شبیه اطلاعات
حساس باشد (user، email، token، feature، raw و…) حتی اگر فرستنده ارسال کند،
قبل از نوشتن حذف می‌شود. این رفتار در تست هم بررسی شده است.

## ۱۲) تشخیص Drift

ماژول `src/monitoring/drift_detection.py` سه سطح ارائه می‌دهد:

1. `detect_drift_ks` — آزمون کولموگروف-اسمیرنوف دو نمونه‌ای. اگر p-value از
   آستانه (۰٫۰۵) کمتر باشد یعنی توزیع دو نمونه یکی نیست.
2. `detect_drift_mean_shift` — تغییر نسبی میانگین؛ اگر بیش از ۱۰٪ جابه‌جا شود
   هشدار می‌دهد.
3. `detect_feature_drift` — ترکیب هر دو روش برای همه‌ی ویژگی‌ها.
   `drift_score` برابر است با **سهم ویژگی‌هایی که drift کرده‌اند** (عددی بین
   ۰ و ۱) و آستانه‌ی هشدار طبق صورت تمرین ۰٫۳ است.

گزارش `reports/drift_report.json` و `reports/drift_report.md` به این سؤال‌ها
جواب می‌دهد: آیا drift رخ داده؟ کدام ویژگی‌ها تحت تأثیرند؟ شدت چقدر است؟ چه
اقدامی پیشنهاد می‌شود؟

نتیجه‌ی مقایسه‌ی v1 با v2:

| ویژگی | آماره KS | p-value | تغییر نسبی میانگین |
|---|---|---|---|
| support_tickets | 0.4029 | <0.0001 | +108.5% |
| monthly_charges | 0.4017 | <0.0001 | +23.5% |
| credit_score | 0.1883 | <0.0001 | −3.3% |
| total_charges | 0.1721 | <0.0001 | +31.6% |

خروجی: **drift تشخیص داده شد**، `drift_score = 0.3333` (۴ ویژگی از ۱۲) که از
آستانه‌ی ۰٫۳ بیشتر است و شدت آن **متوسط** است. اقدام پیشنهادی: بررسی خط
تولید داده‌ی همین چهار ویژگی، مقایسه‌ی F1 آفلاین با F1 سرویس و برنامه‌ریزی
آموزش مجدد روی توزیع فعلی.

اجرا:

```bash
python scripts/drift_report.py            # ساخت گزارش
python scripts/drift_report.py --fail-on-drift   # کد خروج ۱ در صورت هشدار
```

## ۱۳) CI/CD

فایل `.github/workflows/ci.yml` پنج job دارد:

1. **code-quality:** بررسی lint و فرمت با ruff.
2. **test:** اجرای کل تست‌های pytest.
3. **train:** اجرای `run_experiments.py` و آپلود مدل، نتایج و دیتابیس MLflow.
4. **drift-check:** ساخت گزارش drift و شکست خوردن job در صورت عبور از آستانه
   (سلامت مدل).
5. **docker-smoke:** ساخت ایمیج، اجرای کانتینر، انتظار برای `/health` و سپس
   اجرای `scripts/healthcheck.py` که قرارداد پاسخ و اعتبارسنجی ورودی را
   بررسی می‌کند.

به این ترتیب بعد از هر deploy، کیفیت کد و سلامت مدل به‌صورت خودکار بررسی
می‌شود.

## ۱۴) دستورهای اجرای کامل

```bash
cd ANSWER
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/train_experiment.py      # یک run مرجع
python experiments/run_experiments.py       # پنج run + جدول مقایسه
python scripts/drift_report.py              # گزارش drift
pytest tests/ -v                            # ۶۲ تست
uvicorn src.serving.app:app --port 8000     # سرویس
```

## ۱۵) محدودیت‌ها و نکات

- داده فقط ۴۰۰ ردیف است؛ اختلاف F1 بین SVM و رگرسیون لجستیک روی ۸۰ نمونه‌ی
  تست در محدوده‌ی نوسان آماری است.
- `SVC(probability=True)` در scikit-learn نسخه‌ی ۱٫۹ یک `FutureWarning`
  می‌دهد (به نفع `CalibratedClassifierCV` منسوخ شده). کار می‌کند و نسخه‌ی
  کتابخانه pin شده است.
- مرجع drift یک snapshot ثابت است؛ در محیط واقعی باید پنجره‌ی لغزان روی
  آخرین نمونه‌های سرو شده باشد.
- بک‌اند MLflow یک فایل sqlite محلی است؛ برای بازتولید عالی است ولی جای
  سرور مشترک را نمی‌گیرد.
- فایل‌های مدل و داده طبق خواسته‌ی تمرین commit نمی‌شوند؛ با اجرای
  `python experiments/train_experiment.py` دوباره ساخته می‌شوند.
- برای ارسال تمرین، طبق صورت سؤال نام فایل را به شکل
  «نام تمرین + نام و نام خانوادگی» تغییر دهید.

</div>
