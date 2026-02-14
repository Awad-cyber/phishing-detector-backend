// 1. استدعاء الحزم المطلوبة
const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config({ path: './api.env' });

// 2. إعدادات الخادم
const app = express();
const PORT = process.env.PORT || 10000;
const AI_API_URL = 'http://127.0.0.1:5001/predict';

// 3. تفعيل الإضافات (Middlewares)
app.use(cors());
app.use(express.json());

/** فحص أن الخادم شغال (للتأكد من أن Node لا يغلق فوراً) */
app.get('/', (req, res) => {
    res.json({ status: 'ok', message: 'خادم Node شغال', port: PORT });
});

/**
 * نقطة النهاية: فحص البريد الإلكتروني المطور (Advanced Email Scan)
 */
app.post('/scan-email', async (req, res) => {
    const { email_content } = req.body;

    if (!email_content) {
        return res.status(400).json({ error: 'محتوى البريد الإلكتروني مطلوب' });
    }

    try {
        // أ. استخراج الروابط من النص
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        const urls = email_content.match(urlRegex) || [];

        // ب. إرسال النص لخادم التحليل (Flask)
        let prediction = null;
        try {
            const aiResponse = await axios.post(AI_API_URL, { text: email_content });
            const data = aiResponse.data;
            if (data && data.success && data.prediction) {
                prediction = data.prediction;
            }
        } catch (aiError) {
            console.error("خطأ في التواصل مع خادم التحليل:", aiError.message);
        }

        // ج. فحص الروابط عبر VirusTotal — ننتظر اكتمال التحليل (polling) ثم نقرأ stats
        let vtResults = [];
        if (urls.length > 0 && process.env.VIRUSTOTAL_API_KEY) {
            const apiKey = process.env.VIRUSTOTAL_API_KEY;
            const vtScanUrl = 'https://www.virustotal.com/api/v3/urls';
            const vtHeaders = { 'x-apikey': apiKey };
            const pollIntervalMs = 5000;
            const maxWaitMs = 60000;

            for (let i = 0; i < Math.min(urls.length, 2); i++) {
                try {
                    const scanRes = await axios.post(vtScanUrl, `url=${encodeURIComponent(urls[i])}`, {
                        headers: { ...vtHeaders, 'Content-Type': 'application/x-www-form-urlencoded' }
                    });
                    const analysisId = scanRes.data.data.id;
                    let attrs = { status: 'queued', stats: {} };
                    const deadline = Date.now() + maxWaitMs;
                    while (Date.now() < deadline) {
                        const reportRes = await axios.get(`https://www.virustotal.com/api/v3/analyses/${analysisId}`, { headers: vtHeaders });
                        attrs = reportRes.data.data.attributes || {};
                        if (attrs.status === 'completed') break;
                        await new Promise(r => setTimeout(r, pollIntervalMs));
                    }
                    const stats = attrs.stats || {};
                    vtResults.push({
                        url: urls[i],
                        status: attrs.status || 'unknown',
                        stats: {
                            malicious: stats.malicious != null ? Number(stats.malicious) : 0,
                            suspicious: stats.suspicious != null ? Number(stats.suspicious) : 0,
                            harmless: stats.harmless != null ? Number(stats.harmless) : 0,
                            undetected: stats.undetected != null ? Number(stats.undetected) : 0
                        }
                    });
                } catch (vtError) {
                    console.error(`خطأ في فحص الرابط ${urls[i]}:`, vtError.message);
                    vtResults.push({ url: urls[i], status: 'error', stats: {}, error: vtError.message });
                }
            }
        }

        // د. الحكم الأولي من تحليل النص (Flask)
        let verdict = prediction ? {
            label: prediction.label || "unknown",
            risk_level: prediction.risk_level || "Unknown",
            confidence: prediction.confidence_score != null ? prediction.confidence_score : 0
        } : { label: "unknown", risk_level: "Unknown", confidence: 0 };

        // هـ. دمج نتائج VirusTotal في الحكم النهائي — إذا الرابط خبيث عند VT لا نعطي "آمن"
        let reasons = (prediction && prediction.reasons) ? [...prediction.reasons] : [];
        const maxMalicious = vtResults.reduce((max, r) => {
            const m = (r.stats && r.stats.malicious) ? r.stats.malicious : 0;
            return m > max ? m : max;
        }, 0);

        if (maxMalicious >= 5) {
            verdict = { label: "phishing", risk_level: "High", confidence: Math.max(verdict.confidence, 0.92) };
            reasons.unshift(`VirusTotal: ${maxMalicious} محركاً صنّفوا الرابط كخبيث — خطورة عالية.`);
        } else if (maxMalicious >= 1) {
            if (verdict.risk_level === "Low" || verdict.confidence < 0.55) {
                verdict = { label: "suspicious", risk_level: "High", confidence: Math.max(verdict.confidence, 0.78) };
            } else {
                verdict.confidence = Math.max(verdict.confidence, 0.7);
                verdict.risk_level = verdict.confidence >= 0.8 ? "High" : "Medium";
                verdict.label = verdict.risk_level === "High" ? "phishing" : "suspicious";
            }
            reasons.unshift(`VirusTotal: ${maxMalicious} محركاً صنّفوا الرابط كخبيث — لا تعتبر الرسالة آمنة.`);
        }

        const security_recommendation = verdict.risk_level === "High" ?
            "تحذير: هذه الرسالة خطيرة جداً. لا تقم بالضغط على أي روابط أو إدخال بياناتك." :
            verdict.risk_level === "Medium" ?
            "تنبيه: الرسالة تحتوي على مؤشرات مريبة. يرجى التحقق من هوية المرسل." :
            "الرسالة تبدو آمنة، ولكن يرجى الحذر دائماً.";

        const finalReport = {
            timestamp: new Date().toISOString(),
            verdict,
            ai_deep_analysis: prediction ? {
                ml_probability: prediction.ml_probability,
                linguistic_metrics: prediction.linguistic_analysis || {},
                reasons
            } : { ml_probability: null, linguistic_metrics: {}, reasons },
            url_analysis: (prediction && prediction.url_analysis) ? prediction.url_analysis : [],
            external_intelligence: {
                total_urls_found: urls.length,
                scanned_urls: vtResults
            },
            security_recommendation
        };

        res.json(finalReport);

    } catch (error) {
        console.error("خطأ عام في معالجة البريد:", error);
        res.status(500).json({ error: 'حدث خطأ أثناء معالجة البريد الإلكتروني' });
    }
});
app.options('/scan-email', cors());
// تشغيل الخادم
const server = app.listen(PORT, () => {
    console.log(`الخادم المطور يعمل على المنفذ http://localhost:${PORT}`);
    console.log('لا تغلق هذه النافذة — أوقف الخادم بـ Ctrl+C');
});
server.on('error', (err) => {
    console.error('خطأ في تشغيل الخادم:', err.message);
});
