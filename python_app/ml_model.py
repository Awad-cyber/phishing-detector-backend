import os
import re
import string
from typing import Dict, List, Tuple

import joblib


class ModelNotLoadedError(RuntimeError):
    """Raised when the ML model or vectorizer failed to load."""


# Model paths (compatible with Render deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")


model = None
vectorizer = None
_model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(PREPROCESSOR_PATH)
except Exception as exc:  # noqa: BLE001
    # We keep the error and raise a clear exception later if prediction is called.
    _model_load_error = exc


def _ensure_model_is_loaded() -> None:
    """
    Ensure that the model and vectorizer are loaded.

    This avoids NameError at prediction time and gives a clear message instead.
    """
    if model is None or vectorizer is None:
        raise ModelNotLoadedError(f"Model or preprocessor could not be loaded: {_model_load_error}")


KNOWN_LEGIT_DOMAINS = [
    "google.com",
    "gmail.com",
    "outlook.com",
    "microsoft.com",
    "live.com",
    "yahoo.com",
    "apple.com",
    "icloud.com",
    "paypal.com",
    "amazon.com",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "bankofamerica.com",
    "chase.com",
    "hsbc.com",
]


def _extract_domain(url: str) -> str:
    """
    Extract a simple domain (registrable part) from a URL string.

    This is not a full public suffix parser, but is sufficient for demonstration:
    it keeps the last two labels (e.g., "paypal.com", "google.com").
    """
    # Remove protocol if still present
    without_proto = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    # Remove path/query/fragment
    host = without_proto.split("/")[0]
    # Remove potential credentials and port
    if "@" in host:
        host = host.split("@", 1)[1]
    if ":" in host:
        host = host.split(":", 1)[0]

    # Very simple IPv4 detection
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        return host

    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _levenshtein(a: str, b: str) -> int:
    """
    Simple Levenshtein distance implementation (edit distance).
    Good enough for short domain names in a university project.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[-1][-1]


def analyse_url(url: str) -> Dict[str, str]:
    """
    Analyse a single URL and compare its domain with a list of known legitimate domains.

    Returns a small explanation that can be shown directly in the UI.
    """
    domain = _extract_domain(url)

    # 0) Check for HTTP links first - classify as unsafe
    if url.startswith('http://') and not url.startswith('https://'):
        return {
            "url": url,
            "domain": domain,
            "category": "unsafe",
            "reason": f"هذا الرابط يستخدم اتصال HTTP غير مشفر - غير آمن لمشاركة البيانات الحساسة",
        }

    # 1) Exact match with a known legitimate domain
    if domain in KNOWN_LEGIT_DOMAINS:
        return {
            "url": url,
            "domain": domain,
            "category": "legitimate",
            "reason": f"The URL domain '{domain}' matches a known legitimate service.",
        }

    # 2) Domain contains a known brand but is not exactly equal
    for legit in KNOWN_LEGIT_DOMAINS:
        brand = legit.split(".")[0]
        if brand in domain and domain != legit:
            return {
                "url": url,
                "domain": domain,
                "category": "suspicious_lookalike",
                "reason": f"The domain '{domain}' contains the brand '{brand}' "
                f"but does not exactly match the official domain '{legit}'.",
            }

    # 3) Domain has small edit distance to a known legitimate domain
    min_distance = None
    closest_ref = None
    for legit in KNOWN_LEGIT_DOMAINS:
        d = _levenshtein(domain, legit)
        if min_distance is None or d < min_distance:
            min_distance = d
            closest_ref = legit

    if min_distance is not None and min_distance <= 2 and closest_ref:
        return {
            "url": url,
            "domain": domain,
            "category": "suspicious_lookalike",
            "reason": f"The domain '{domain}' looks very similar to '{closest_ref}' "
            "but is not identical. This is a common phishing technique (typosquatting).",
        }

    # 4) IP address domains are also suspicious in many scenarios
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain):
        return {
            "url": url,
            "domain": domain,
            "category": "suspicious_ip",
            "reason": f"The URL uses a raw IP address '{domain}' instead of a normal domain name.",
        }

    # 5) Unknown domain (neither clearly safe nor clearly fake)
    return {
        "url": url,
        "domain": domain,
        "category": "unknown",
        "reason": f"The domain '{domain}' is not in the list of known services. Treat it with caution.",
    }


def clean_text(text: str) -> str:
    """
    Basic text normalisation for the TF-IDF + Logistic Regression pipeline.

    - Lowercasing
    - Removing URLs (replaced with token URL)
    - Replacing digits with NUM
    - Removing punctuation
    """
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def extract_rule_based_features(text: str) -> Dict[str, object]:
    """
    Lightweight, rule-based heuristics to complement the ML model.

    These are intentionally simple, interpretable, and academic:
    - Suspicious URLs
    - Urgent / threatening language
    - Credential / financial keywords
    - Character-level patterns (length, uppercase, punctuation)
    """
    text_lower = text.lower()

    url_pattern = r"https?://[^\s]+"
    urls = re.findall(url_pattern, text_lower)

    urgency_phrases = [
        # English urgency phrases
        "urgent",
        "immediately",
        "asap",
        "limited time",
        "hurry",
        "action required",
        "final notice",
        "act now",
        "verify now",
        "confirm now",
        "expires soon",
        "time sensitive",
        "within 24 hours",
        "last chance",
        "don't wait",
        "act immediately",
        "urgent action",
        "immediate attention",
        "time is running out",
        "before it's too late",
        "quick action required",
        "urgent response needed",
        # Arabic urgency phrases - expanded and more comprehensive
        "أسرع",
        "فوراً",
        "الآن",
        "خلال 24 ساعة",
        "مهم جداً",
        "يُنفذ فوراً",
        "عاجل",
        "مستعجل",
        "بسرعة",
        "فوري",
        "فورا",
        "حالاً",
        "حالا",
        "بشكل عاجل",
        "فورا دون تردد",
        "لا تتأخر",
        "الوقت محدود",
        "فرصة أخيرة",
        "آخر فرصة",
        "ينتهي قريباً",
        "سينتهي قريبا",
        "قبل فوات الأوان",
        "يجب عليك الآن",
        "مطلوب فورا",
        "إجراء فوري",
        "استجابة عاجلة",
        "تنفيذ فوري",
        "تصرف فورا",
        "تصرف الآن",
        "لا تنتظر",
        "سرعان ما يمكن",
        "في أسرع وقت",
        "بأسرع ما يمكن",
        "فورا وقبل فوات الأوان",
        "الوقت ينفد",
        "الوقت يقترب من النهاية",
        "مهل قليلة",
        "وقت قصير",
        "فرصة لا تعوض",
        "لن تتكرر",
        "لن تتكرر هذه الفرصة",
        "عاجل جدا",
        "عاجل جداً",
        "مستعجل جدا",
        "مستعجل جداً",
        "الأهمية قصوى",
        "أولوية قصوى",
        "أمر عاجل",
        "أمر مستعجل",
        "تنفيذ حالي",
        "التنفيذ الفوري",
        "الإجراء الفوري",
        "التصرف الفوري",
        "الحل الفوري",
        "الحل العاجل",
        "القرار الفوري",
        "القرار العاجل",
        "الرد الفوري",
        "الرد العاجل",
        "التأكيد الفوري",
        "التأكيد العاجل",
        "الموافقة الفورية",
        "الموافقة العاجلة",
        "القبول الفوري",
        "القبول العاجل",
        "الرفض الفوري",
        "الرفض العاجل",
        "الإلغاء الفوري",
        "الإلغاء العاجل",
        "التغيير الفوري",
        "التغيير العاجل",
        "التعديل الفوري",
        "التعديل العاجل",
        "التحديث الفوري",
        "التحديث العاجل",
        "التفعيل الفوري",
        "التفعيل العاجل",
        "إلغاء التفعيل الفوري",
        "إلغاء التفعيل العاجل",
        "تسجيل الخروج الفوري",
        "تسجيل الخروج العاجل",
        "تسجيل الدخول الفوري",
        "تسجيل الدخول العاجل",
        "تغيير كلمة المرور فورا",
        "تغيير كلمة المرور عاجل",
        "تحديث البيانات فورا",
        "تحديث البيانات عاجل",
        "تأكيد البريد الإلكتروني فورا",
        "تأكيد البريد الإلكتروني عاجل",
        "تفعيل الحساب فورا",
        "تفعيل الحساب عاجل",
        "إغلاق الحساب فورا",
        "إغلاق الحساب عاجل",
        "حذف الحساب فورا",
        "حذف الحساب عاجل",
        "تعليق الحساب فورا",
        "تعليق الحساب عاجل",
        "إيقاف الحساب فورا",
        "إيقاف الحساب عاجل",
        "حظر الحساب فورا",
        "حظر الحساب عاجل",
        "رفع الحظر فورا",
        "رفع الحظر عاجل",
        "استعادة الحساب فورا",
        "استعادة الحساب عاجل",
        "إعادة تعيين كلمة المرور فورا",
        "إعادة تعيين كلمة المرور عاجل",
        "تغيير رقم الهاتف فورا",
        "تغيير رقم الهاتف عاجل",
        "تحديث رقم الهاتف فورا",
        "تحديث رقم الهاتف عاجل",
        "تأكيد رقم الهاتف فورا",
        "تأكيد رقم الهاتف عاجل",
        "تفعيل رقم الهاتف فورا",
        "تفعيل رقم الهاتف عاجل",
        "إلغاء رقم الهاتف فورا",
        "إلغاء رقم الهاتف عاجل",
        "حذف رقم الهاتف فورا",
        "حذف رقم الهاتف عاجل",
        "إضافة رقم الهاتف فورا",
        "إضافة رقم الهاتف عاجل",
        "تغيير العنوان فورا",
        "تغيير العنوان عاجل",
        "تحديث العنوان فورا",
        "تحديث العنوان عاجل",
        "تأكيد العنوان فورا",
        "تأكيد العنوان عاجل",
        "إضافة العنوان فورا",
        "إضافة العنوان عاجل",
        "حذف العنوان فورا",
        "حذف العنوان عاجل",
        "تعديل العنوان فورا",
        "تعديل العنوان عاجل",
        "استلام العنوان فورا",
        "استلام العنوان عاجل",
        "شحن العنوان فورا",
        "شحن العنوان عاجل",
        "توصيل العنوان فورا",
        "توصيل العنوان عاجل",
        "إرسال العنوان فورا",
        "إرسال العنوان عاجل",
        "استلام الطلب فورا",
        "استلام الطلب عاجل",
        "شحن الطلب فورا",
        "شحن الطلب عاجل",
        "توصيل الطلب فورا",
        "توصيل الطلب عاجل",
        "إرسال الطلب فورا",
        "إرسال الطلب عاجل",
        "إلغاء الطلب فورا",
        "إلغاء الطلب عاجل",
        "تعديل الطلب فورا",
        "تعديل الطلب عاجل",
        "تحديث الطلب فورا",
        "تحديث الطلب عاجل",
        "تأكيد الطلب فورا",
        "تأكيد الطلب عاجل",
        "استلام المنتج فورا",
        "استلام المنتج عاجل",
        "شحن المنتج فورا",
        "شحن المنتج عاجل",
        "توصيل المنتج فورا",
        "توصيل المنتج عاجل",
        "إرسال المنتج فورا",
        "إرسال المنتج عاجل",
        "إلغاء المنتج فورا",
        "إلغاء المنتج عاجل",
        "تعديل المنتج فورا",
        "تعديل المنتج عاجل",
        "تحديث المنتج فورا",
        "تحديث المنتج عاجل",
        "تأكيد المنتج فورا",
        "تأكيد المنتج عاجل",
        "استلام الخدمة فورا",
        "استلام الخدمة عاجل",
        "تقديم الخدمة فورا",
        "تقديم الخدمة عاجل",
        "إلغاء الخدمة فورا",
        "إلغاء الخدمة عاجل",
        "تعديل الخدمة فورا",
        "تعديل الخدمة عاجل",
        "تحديث الخدمة فورا",
        "تحديث الخدمة عاجل",
        "تأكيد الخدمة فورا",
        "تأكيد الخدمة عاجل",
        "استلام البطاقة فورا",
        "استلام البطاقة عاجل",
        "شحن البطاقة فورا",
        "شحن البطاقة عاجل",
        "توصيل البطاقة فورا",
        "توصيل البطاقة عاجل",
        "إرسال البطاقة فورا",
        "إرسال البطاقة عاجل",
        "إلغاء البطاقة فورا",
        "إلغاء البطاقة عاجل",
        "تعديل البطاقة فورا",
        "تعديل البطاقة عاجل",
        "تحديث البطاقة فورا",
        "تحديث البطاقة عاجل",
        "تأكيد البطاقة فورا",
        "تأكيد البطاقة عاجل",
        "استلام الحساب فورا",
        "استلام الحساب عاجل",
        "شحن الحساب فورا",
        "شحن الحساب عاجل",
        "توصيل الحساب فورا",
        "توصيل الحساب عاجل",
        "إرسال الحساب فورا",
        "إرسال الحساب عاجل",
        "إلغاء الحساب فورا",
        "إلغاء الحساب عاجل",
        "تعديل الحساب فورا",
        "تعديل الحساب عاجل",
        "تحديث الحساب فورا",
        "تحديث الحساب عاجل",
        "تأكيد الحساب فورا",
        "تأكيد الحساب عاجل",
    ]
    threat_phrases = [
        "suspended",
        "blocked",
        "deleted",
        "unauthorized",
        "legal action",
        "police",
        "penalty",
        "account locked",
        "verify or suspend",
        "will be closed",
        "أوقف",
        "سيتم إغلاق",
        "حسابك موقوف",
        "سيتم حذف",
        "انتهت صلاحية",
    ]
    credential_phrases = [
        # English credential / personal-data phrases
        "verify your account",
        "confirm your password",
        "login now",
        "log in now",
        "update your information",
        "update your account",
        "update your details",
        "bank account",
        "credit card",
        "debit card",
        "security code",
        "one-time password",
        "otp code",
        "enter your password",
        "enter your personal data",
        "enter your personal information",
        "enter your credentials",
        "confirm your identity",
        "click to login",
        "sign in",
        "secure your account",
        "click the link below",
        "click here to verify",
        "follow this link",
        # Arabic credential / personal-data phrases
        "ادخل بياناتك",
        "أدخل بياناتك",
        "ادخل معلوماتك الشخصية",
        "أدخل معلوماتك الشخصية",
        "تحديث بياناتك",
        "تحديث معلوماتك",
        "ادخل كلمة السر",
        "أدخل كلمة السر",
        "ادخل كلمة المرور",
        "أدخل كلمة المرور",
        "كلمة السر",
        "كلمة المرور",
        "ادخل رقم البطاقة",
        "أدخل رقم البطاقة",
        "رقم البطاقة",
        "رقم الحساب",
        "بيانات الحساب",
        "تأكيد الحساب",
        "تأكيد هويتك",
        "تسجيل الدخول بحسابك",
        "ادخل بيانات بطاقتك",
        "أدخل بيانات بطاقتك",
        "اضغط على الرابط",
        "استخدم الرابط",
        "رابط التحقق",
        "الرابط أدناه",
        "تفعيل الحساب",
    ]

    spam_phrases = [
        # English "too good to be true" / pressure phrases
        "you have won",
        "you won",
        "congratulations",
        "claim your prize",
        "claim your reward",
        "limited offer",
        "exclusive offer",
        "free gift",
        "free prize",
        "click here to claim",
        "click here",
        "click below",
        # Arabic equivalents
        "لقد ربحت",
        "مبروك",
        "تهانينا",
        "جائزة مجانية",
        "جائزة نقدية",
        "عرض محدود",
        "اضغط هنا للحصول على الجائزة",
        "اضغط هنا لاستلام الجائزة",
        "اضغط هنا",
    ]

    def _count_hits(phrases: List[str]) -> int:
        return sum(1 for p in phrases if p in text_lower)

    urgency_hits = _count_hits(urgency_phrases)
    threat_hits = _count_hits(threat_phrases)
    credential_hits = _count_hits(credential_phrases)
    spam_hits = _count_hits(spam_phrases)

    # --- character-level features ---
    raw = text.strip()
    length_chars = len(raw)
    num_exclamations = raw.count("!")

    # ratio of uppercase letters to all alphabetic characters
    letters = [ch for ch in raw if ch.isalpha()]
    upper_letters = [ch for ch in raw if ch.isupper()]
    uppercase_ratio = (len(upper_letters) / len(letters)) if letters else 0.0

    # ratio of non-ASCII characters (often appears in obfuscation / mixed scripts)
    non_ascii_chars = [ch for ch in raw if ord(ch) > 127]
    non_ascii_ratio = (len(non_ascii_chars) / max(length_chars, 1)) if length_chars else 0.0

    return {
        "has_url": bool(urls),
        "num_urls": len(urls),
        "urgency_hits": urgency_hits,
        "threat_hits": threat_hits,
        "credential_hits": credential_hits,
        "spam_hits": spam_hits,
        "length_chars": length_chars,
        "num_exclamations": num_exclamations,
        "uppercase_ratio": uppercase_ratio,
        "non_ascii_ratio": non_ascii_ratio,
        "urls": urls,
    }


def compute_heuristic_score(features: Dict[str, object]) -> float:
    """
    Map rule-based features to a simple risk score in [0, 1].

    This is intentionally transparent and documented for academic purposes.
    """
    score = 0.0

    if features["has_url"]:
        score += 0.2
        if features["num_urls"] >= 3:
            score += 0.1

    if features["urgency_hits"] > 0:
        score += 0.2

    if features["threat_hits"] > 0:
        score += 0.25

    if features["credential_hits"] > 0:
        # Asking for credentials / personal data is a strong phishing signal
        score += 0.35
        if features["credential_hits"] >= 2:
            score += 0.1

    # "Too good to be true" / prize language
    if features.get("spam_hits", 0) > 0:
        score += 0.15

    # Character-level contributions
    # Many exclamation marks → often used in phishing / spam
    if features["num_exclamations"] >= 3:
        score += 0.1

    # Very high uppercase ratio (shouting style)
    if features["uppercase_ratio"] >= 0.5 and features["length_chars"] >= 30:
        score += 0.1

    # Significant amount of non-ASCII characters in a mostly Latin email
    if features["non_ascii_ratio"] >= 0.3 and features["length_chars"] >= 30:
        score += 0.05

    # Cap at 1.0 to keep it interpretable
    return min(score, 1.0)


def build_reasons(text: str, ml_probability: float, features: Dict[str, object]) -> List[str]:
    """
    Convert features and model output into human-readable explanations.
    """
    reasons: List[str] = []

    if features["has_url"]:
        reasons.append("At least one URL was detected in the email body.")
        if features["num_urls"] > 1:
            reasons.append("Multiple URLs were found, which is common in phishing or promotional emails.")

    if features["urgency_hits"] > 0:
        reasons.append("Urgent language used (e.g., 'urgent', 'immediately', 'action required').")

    if features["threat_hits"] > 0:
        reasons.append(
            "Threatening language detected (e.g., 'account suspended', 'legal action', 'penalty')."
        )

    if features["credential_hits"] > 0:
        reasons.append(
            "The email asks for sensitive information such as passwords, bank details, or account verification."
        )
        reasons.append(
            "هذه الرسالة تطلب منك إدخال بيانات شخصية أو معلومات حساب، "
            "وهذا سلوك شائع في رسائل التصيّد الاحتيالي."
        )

    if features.get("spam_hits", 0) > 0:
        reasons.append(
            "The email contains prize/offer language (e.g., 'you have won', 'free gift'), "
            "which is often used to attract victims in phishing campaigns."
        )

    # Character-level reasons
    if features["num_exclamations"] >= 3:
        reasons.append("The email uses many exclamation marks, which is common in aggressive phishing messages.")

    if features["uppercase_ratio"] >= 0.5 and features["length_chars"] >= 30:
        reasons.append("A large portion of the text is in CAPITAL letters, which can indicate pressure or shouting.")

    if features["non_ascii_ratio"] >= 0.3 and features["length_chars"] >= 30:
        reasons.append("The email contains many unusual or non-standard characters, which may hide malicious content.")

    # Model-based reasons
    if ml_probability >= 0.8:
        reasons.append("The statistical model strongly matches known phishing email patterns.")
    elif ml_probability <= 0.2:
        reasons.append("The statistical model is not similar to known phishing emails.")
    else:
        reasons.append("The statistical model indicates a moderate similarity to phishing emails.")

    if not reasons:
        reasons.append("No strong phishing indicators were detected in the content.")

    return reasons


def predict(text: str) -> Dict[str, object]:
    """
    Main prediction function used by the Flask API.

    Returns:
        {
            "label": "phishing" | "suspicious" | "safe",
            "risk_level": "High" | "Medium" | "Low",
            "confidence_score": float,
            "ml_probability": float,
            "linguistic_analysis": { ... },
            "reasons": [ ... ]
        }
    """
    _ensure_model_is_loaded()

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    # 1. Statistical probability from the Logistic Regression model
    ml_probability = float(model.predict_proba(vec)[0][1])

    # 2. Simple rule-based heuristics (content-level)
    features = extract_rule_based_features(text)
    heuristic_score = compute_heuristic_score(features)

    # 2b. Detailed URL analysis based on domain comparison
    url_analysis = []
    if features["has_url"]:
        for url in features.get("urls", []):
            url_analysis.append(analyse_url(url))

    # 2c. Let URL categories influence the heuristic score
    #     - suspicious_lookalike and suspicious_ip should increase risk
    #     - clearly legitimate domains can slightly decrease heuristic risk
    for item in url_analysis:
        category = item.get("category")
        if category == "suspicious_lookalike":
            heuristic_score = min(1.0, heuristic_score + 0.3)
        elif category == "suspicious_ip":
            heuristic_score = min(1.0, heuristic_score + 0.2)
        elif category == "legitimate":
            heuristic_score = max(0.0, heuristic_score - 0.1)

    # 3. Combine both signals — وزن أكبر للقواعد لتحسين الحساسية وتقليل "آمن" الخاطئ
    #    35% ML + 65% قواعد (القواعد تلتقط عبارات التصيد حتى لو النموذج المدرب محافظ)
    final_score = (0.35 * ml_probability) + (0.65 * heuristic_score)

    # 3b. رسائل تطلب بيانات أو تحتوي رابط بدون دومين موثوق → على الأقل مشبوه
    if features["credential_hits"] > 0 and not features["has_url"]:
        final_score = max(final_score, 0.6)
    if features["has_url"] and heuristic_score >= 0.25:
        # وجود رابط مع أي إشارات مشبوهة → لا نعطي "آمن" بسهولة
        final_score = max(final_score, 0.45)

    # 3c. إذا تحليل الروابط وجد دومين مشبوه (تقليد أو IP) → رفع الخطورة
    has_suspicious_url = any(
        item.get("category") in ("suspicious_lookalike", "suspicious_ip")
        for item in url_analysis
    )
    if has_suspicious_url:
        final_score = max(final_score, 0.7)

    # 4. عتبات أقل لتصنيف "خطير" و "مشبوه" — أقل أماناً خاطئاً (False Negative)
    if final_score >= 0.65:
        label = "phishing"
        risk_level = "High"
    elif final_score >= 0.35:
        label = "suspicious"
        risk_level = "Medium"
    else:
        label = "safe"
        risk_level = "Low"

    reasons = build_reasons(text, ml_probability, features)

    return {
        "label": label,
        "risk_level": risk_level,
        "confidence_score": round(final_score, 3),
        "ml_probability": round(ml_probability, 3),
        "heuristic_score": round(heuristic_score, 3),
        "final_score": round(final_score, 3),
        "linguistic_analysis": {
            "has_url": features["has_url"],
            "num_urls": features["num_urls"],
            "urgency_hits": features["urgency_hits"],
            "threat_hits": features["threat_hits"],
            "credential_hits": features["credential_hits"],
            "spam_hits": features.get("spam_hits", 0),
            "length_chars": features["length_chars"],
            "num_exclamations": features["num_exclamations"],
            "uppercase_ratio": round(features["uppercase_ratio"], 3),
            "non_ascii_ratio": round(features["non_ascii_ratio"], 3),
            # Kept for compatibility with the original response schema
            "sentiment": "neutral",
        },
        "reasons": reasons,
        "url_analysis": url_analysis,
    }
