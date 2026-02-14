# بيانات تدريب كاشف التصيد

## تنسيق الملف (CSV)

- **عمود النص:** اسم العمود أحد: `text`, `content`, `body`, `email`, `message` (أو أول عمود إن لم يوجد).
- **عمود التصنيف:** اسم العمود أحد: `label`, `is_phishing`, `phishing`, `class`, `category` (أو ثاني عمود).

قيم التصنيف:
- **تصيد (1):** `1`, `phishing`, `phish`, `malicious`, `yes`, `true`
- **شرعي (0):** أي قيمة أخرى (مثل `0`, `legitimate`, `safe`, `no`).

## مثال

```csv
text,label
"Verify your account now. Click here.",1
"Your order has shipped. Track below.",0
```

## إضافة بيانات أكثر

- ضع ملفات CSV إضافية في هذا المجلد وادمجها، أو استبدل `train.csv` بملف أكبر.
- يمكنك استخدام مجموعات معروفة (مثل PhishTank، أو datasets من Kaggle) بعد تحويلها للصيغة أعلاه.
- كلما زاد عدد رسائل التصيد والشرعية المتنوعة، تحسّن أداء النموذج.

## تشغيل التدريب

من مجلد `backend/backend`:

```bash
python train_model.py
```

أو مع ملف مخصص:

```bash
python train_model.py --data data/my_data.csv
```

النموذج والمُجهّز يُحفظان في نفس المجلد: `model.pkl`, `preprocessor.pkl`.
