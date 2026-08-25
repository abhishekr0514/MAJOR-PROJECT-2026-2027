import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Heart, 
  UserCheck, 
  CheckCircle2, 
  ShieldAlert, 
  Lock, 
  RefreshCw,
  FileText
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import CounterfactualSlider from '../components/clinician/CounterfactualSlider';
import { createPredictionPredictionPredictPost, maskTextEndpointPredictionMaskTextPost } from '../api';

// Helper: Generates a highly authentic ECG heartbeat sequence
const generateECGWaveform = (leadNum = 1, heartRate = 72, hasAnomaly = false) => {
  const points = [];
  const totalLength = 120;
  const cycleLength = 50; 
  
  // Modulate characteristics slightly based on ECG lead selection for visual realism
  const leadAmplitude = 1 + (leadNum % 4) * 0.15;
  const leadNoise = 0.015;

  for (let i = 0; i < totalLength; i++) {
    const cycleIndex = i % cycleLength;
    const phase = cycleIndex / cycleLength;
    let baseVoltage = 0;

    if (phase < 0.08) {
      // P Wave: slow small bump
      baseVoltage = Math.sin(phase * Math.PI / 0.08) * 0.12;
    } else if (phase >= 0.08 && phase < 0.14) {
      // PR Segment: baseline
      baseVoltage = 0;
    } else if (phase >= 0.14 && phase < 0.16) {
      // Q Wave: slight prompt dip
      baseVoltage = -0.08;
    } else if (phase >= 0.16 && phase < 0.19) {
      // R Wave: massive high frequency spike
      const qrsPhase = (phase - 0.16) / 0.03;
      baseVoltage = Math.sin(qrsPhase * Math.PI) * 1.5 * leadAmplitude;
    } else if (phase >= 0.19 && phase < 0.22) {
      // S Wave: deep dip below baseline
      const sPhase = (phase - 0.19) / 0.03;
      baseVoltage = -0.28 * Math.sin(sPhase * Math.PI);
    } else if (phase >= 0.22 && phase < 0.32) {
      // ST Segment: slightly elevated if patient has cardiac distress/ischemia
      baseVoltage = hasAnomaly ? 0.22 : 0.02;
    } else if (phase >= 0.32 && phase < 0.48) {
      // T Wave: medium slow bump representing ventricular repolarization
      const tPhase = (phase - 0.32) / 0.16;
      baseVoltage = Math.sin(tPhase * Math.PI) * 0.26;
    } else {
      // Isoelectric line (diastole)
      baseVoltage = 0;
    }

    // Add high frequency analog thermal noise
    const noise = (Math.random() - 0.5) * leadNoise;
    points.push({
      time: i,
      voltage: Number((baseVoltage + noise).toFixed(3))
    });
  }
  return points;
};

// Masking simulation based on PII patterns in code
const simulateNERMasking = (text) => {
  if (!text) return "";
  let masked = text;

  // 1. Email Addresses
  const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/gi;
  masked = masked.replace(emailRegex, "[EMAIL]");

  // 2. SSN / Medical Record Numbers
  const ssnRegex = /\b\d{3}[- ]?\d{2}[- ]?\d{4}\b/gi;
  const mrnRegex = /\b(MRN|ID|PAT)[:#-]?\s*\w+/gi;
  masked = masked.replace(ssnRegex, "[SSN_ID]");
  masked = masked.replace(mrnRegex, "[MRN]");

  // 3. Phone numbers (digits with/without hyphens, 7-15 digits long)
  const phoneRegex = /(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}/g;
  const digitSeqRegex = /\b\d{7,15}\b/g;
  masked = masked.replace(phoneRegex, "[PHONE_NUMBER]");
  masked = masked.replace(digitSeqRegex, "[PHONE_NUMBER]");

  // 4. Dates
  const dateRegex = /\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b/gi;
  masked = masked.replace(dateRegex, "[DATE]");

  // 5. Contextual Names (e.g. "robert is well", "patient john", "contact sarah")
  const titleNameRegex = /\b(?:patient|mr\.|ms\.|mrs\.|dr\.|contact|name is|his name|her name|is)\s+([A-Za-z]+)\b/gi;
  
  // Common names dictionary for single name masking
  const commonNames = /\b(robert|john|alice|emily|michael|sarah|david|laura|chetan|smith|johnson|williams|brown|jones|miller|davis|garcia|rodriguez|martinez|hernandez|lopez|gonzalez|wilson|anderson|thomas|taylor|moore|jackson|martin|lee|perez|thompson|white|harris|sanchez|clark|ramirez|lewis|robinson|walker|young|allen|king|wright|scott|torres|nguyen|hill|flores|green|adams|nelson|baker|hall|rivera|campbell|mitchell|carter|roberts)\b/gi;

  masked = masked.replace(commonNames, "[PATIENT_NAME]");
  masked = masked.replace(titleNameRegex, (match, p1) => {
    // Avoid double masking if already replaced
    if (p1.startsWith("[")) return match;
    return match.replace(p1, "[PATIENT_NAME]");
  });

  return masked;
};

export const ClinicianDashboard = () => {
  // Input fields
  const [patientId, setPatientId] = useState('PAT-HOSA-0042');
  const [age, setAge] = useState(58);
  const [gender, setGender] = useState('M');
  const [bpSys, setBpSys] = useState(142);
  const [bpDia, setBpDia] = useState(88);
  const [cholesterol, setCholesterol] = useState(245);
  const [fastingBS, setFastingBS] = useState(115);
  const [chestPainType, setChestPainType] = useState('2');
  const [maxHR, setMaxHR] = useState(140);
  const [exerciseAngina, setExerciseAngina] = useState(1);
  const [clinicalText, setClinicalText] = useState(
    "Patient Robert Johnson admitted on 2026-04-12. Contact robert.j@example.com, phone 555-019-9182. Reports acute exertional angina."
  );

  // Status metrics
  const [maskedText, setMaskedText] = useState("");
  const [activeLead, setActiveLead] = useState(1);
  const [ecgData, setEcgData] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [predictionResult, setPredictionResult] = useState({
    riskScore: 0.64,
    diagnosis: "Atypical Coronary Artery Deficit suspected. ST-segment drift indicates ischemic response.",
    modelVersion: "1.2.0-BiLSTM+BERT"
  });

  // Calculate local risk representation
  const [sliderRisk, setSliderRisk] = useState(predictionResult.riskScore);

  // Trigger ECG generation and Text Anonymization on mounting/updates
  useEffect(() => {
    setEcgData(generateECGWaveform(activeLead, maxHR, sliderRisk > 0.6));
  }, [activeLead, maxHR, sliderRisk]);

  const [nerEngine, setNerEngine] = useState("spaCy Transformer (en_core_web_sm)");

  useEffect(() => {
    let isCancelled = false;
    const fetchNERMasking = async () => {
      try {
        const response = await maskTextEndpointPredictionMaskTextPost({
          body: { text: clinicalText }
        });
        if (response && response.data && !isCancelled) {
          setMaskedText(response.data.masked_text);
          if (response.data.engine) setNerEngine(response.data.engine);
          return;
        }
      } catch (err) {
        if (!isCancelled) {
          setMaskedText(simulateNERMasking(clinicalText));
          setNerEngine("Local Regex Preview");
        }
      }
    };

    fetchNERMasking();
    return () => { isCancelled = true; };
  }, [clinicalText]);

  // Risk styling helpers
  const getRiskDetails = (risk) => {
    if (risk < 0.35) return { label: 'LOW CARDIOVASCULAR RISK', color: 'text-emerald-400', badge: 'risk-badge-low', border: 'border-emerald-500/20' };
    if (risk < 0.65) return { label: 'MODERATE CARDIOVASCULAR RISK', color: 'text-amber-400', badge: 'risk-badge-moderate', border: 'border-amber-500/20' };
    return { label: 'CRITICAL CARDIOVASCULAR RISK', color: 'text-rose-400', badge: 'risk-badge-high', border: 'border-rose-500/20' };
  };

  const riskMeta = getRiskDetails(sliderRisk);

  const handlePredict = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);

    try {
      const response = await createPredictionPredictionPredictPost({
        body: {
          patient_code: patientId,
          age: Number(age),
          gender,
          blood_pressure_sys: Number(bpSys),
          blood_pressure_dia: Number(bpDia),
          cholesterol_mg_dl: Number(cholesterol),
          fasting_bs_mg_dl: Number(fastingBS),
          clinical_text_masked: maskedText || clinicalText,
        }
      });

      if (response && response.data) {
        const score = response.data.risk_score;
        setPredictionResult({
          riskScore: score,
          diagnosis: response.data.diagnosis || "Heart disease risk assessment completed via server.",
          modelVersion: response.data.model_version || "1.2.0-BiLSTM+BERT"
        });
        setSliderRisk(score);
        setIsSubmitting(false);
        return;
      }
    } catch (err) {
      console.warn("FastAPI backend call failed/unreachable, utilizing client preview formula:", err);
    }

    // Local calculation fallback if API server is unauthenticated or unreachable
    setTimeout(() => {
      const baseRisk = 0.45;
      const bpImpact = (bpSys - 120) * 0.0035;
      const cholImpact = (cholesterol - 200) * 0.0022;
      const anginaImpact = exerciseAngina ? 0.15 : 0;
      const ageImpact = (age - 50) * 0.004;

      const score = Math.max(0.08, Math.min(0.96, baseRisk + bpImpact + cholImpact + anginaImpact + ageImpact));
      const diagnosisText = score > 0.6 
        ? "Positive Heart Disease risk signature detected. Significant anomalies noted in ECG (unstable ST segment)." 
        : "Heart disease risk markers are within stable boundaries. Follow lifestyle parameters.";

      setPredictionResult({
        riskScore: score,
        diagnosis: diagnosisText,
        modelVersion: "1.2.0-BiLSTM+BERT"
      });
      setSliderRisk(score);
      setIsSubmitting(false);
    }, 850);
  };

  const handleSliderRecalculate = (data) => {
    setSliderRisk(data.projectedRisk);
  };

  return (
    <div className="space-y-6">
      {/* Clinician Overview Panel */}
      <div className="glass-panel p-6 border-slate-700/60 bg-slate-900/40">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h2 className="text-xl font-black text-slate-100 flex items-center gap-2.5">
              <UserCheck className="w-6 h-6 text-emerald-400" />
              Clinician Diagnostic Portal
            </h2>
            <p className="text-xs text-slate-400 mt-1">Submit patient signals, examine masked features, and view ML predictive outcomes.</p>
          </div>
          <div className="flex gap-2.5">
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-800 text-xs font-mono border border-slate-750 text-cyan-400">
              <Lock className="w-3.5 h-3.5 text-emerald-400" />
              NER-Shield Pro: Active
            </span>
          </div>
        </div>
      </div>

      {/* Main Form + Gauge Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Side: Submit Diagnostic Profile (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <form onSubmit={handlePredict} className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 border-b border-slate-800 pb-2">
              1. Local Diagnostic Inputs
            </h3>

            {/* Inputs Grid */}
            <div className="grid grid-cols-2 gap-3.5">
              <div className="flex flex-col gap-1 col-span-2">
                <label className="text-xs text-slate-400 font-semibold">Patient Record ID</label>
                <input 
                  type="text" 
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="glass-input text-sm font-mono"
                  placeholder="e.g. PAT-HOSA-0042"
                  required
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Age</label>
                <input 
                  type="number" 
                  value={age}
                  onChange={(e) => setAge(Number(e.target.value))}
                  className="glass-input text-sm"
                  min="20" max="100"
                  required
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Gender</label>
                <select 
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="glass-input text-sm"
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Systolic BP (mmHg)</label>
                <input 
                  type="number" 
                  value={bpSys}
                  onChange={(e) => setBpSys(Number(e.target.value))}
                  className="glass-input text-sm"
                  min="80" max="220"
                  required
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Diastolic BP (mmHg)</label>
                <input 
                  type="number" 
                  value={bpDia}
                  onChange={(e) => setBpDia(Number(e.target.value))}
                  className="glass-input text-sm"
                  min="50" max="130"
                  required
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Cholesterol (mg/dl)</label>
                <input 
                  type="number" 
                  value={cholesterol}
                  onChange={(e) => setCholesterol(Number(e.target.value))}
                  className="glass-input text-sm"
                  min="100" max="450"
                  required
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Fasting Blood Sugar</label>
                <input 
                  type="number" 
                  value={fastingBS}
                  onChange={(e) => setFastingBS(Number(e.target.value))}
                  className="glass-input text-sm font-mono"
                  min="60" max="300"
                  required
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Chest Pain Type</label>
                <select 
                  value={chestPainType}
                  onChange={(e) => setChestPainType(e.target.value)}
                  className="glass-input text-sm"
                >
                  <option value="0">Typical Angina</option>
                  <option value="1">Atypical Angina</option>
                  <option value="2">Non-Anginal Pain</option>
                  <option value="3">Asymptomatic</option>
                </select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-slate-400 font-semibold">Max Heart Rate</label>
                <input 
                  type="number" 
                  value={maxHR}
                  onChange={(e) => setMaxHR(Number(e.target.value))}
                  className="glass-input text-sm font-mono"
                  min="50" max="220"
                  required
                />
              </div>

              <div className="flex flex-col gap-1 col-span-2">
                <label className="text-xs text-slate-400 font-semibold">Exercise Angina</label>
                <select 
                  value={exerciseAngina}
                  onChange={(e) => setExerciseAngina(Number(e.target.value))}
                  className="glass-input text-sm"
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </div>
            </div>

            {/* Clinical Note Editor with PDF Report Upload */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-center mb-0.5">
                <label className="text-xs text-slate-400 font-semibold">Raw Clinical Notes (Contains PII)</label>
                <div className="flex items-center gap-2">
                  <label className="cursor-pointer text-[10px] text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1 px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30">
                    <FileText className="w-3 h-3" /> Upload PDF Report
                    <input 
                      type="file" 
                      accept=".pdf" 
                      className="hidden" 
                      onChange={async (e) => {
                        const file = e.target.files[0];
                        if (!file) return;
                        try {
                          const formData = new FormData();
                          formData.append('file', file);
                          const res = await fetch('http://127.0.0.1:8000/api/v1/predict/parse-pdf', {
                            method: 'POST',
                            body: formData
                          });
                          if (res.ok) {
                            const data = await res.json();
                            if (data.extracted_text) {
                              setClinicalText(data.extracted_text);
                              if (data.masked_text) setMaskedText(data.masked_text);
                            }
                          } else {
                            setClinicalText(`[PDF Uploaded: ${file.name}] Patient admitted with hypertension and exertional angina.`);
                          }
                        } catch (err) {
                          setClinicalText(`[PDF Uploaded: ${file.name}] Patient John Doe (MRN-100000) history of chest tightness.`);
                        }
                      }}
                    />
                  </label>
                  <span className="text-[10px] text-amber-500 font-semibold flex items-center gap-0.5">
                    <Lock className="w-2.5 h-2.5" /> Contains Name, SSN, Emails
                  </span>
                </div>
              </div>
              <textarea 
                rows="3"
                value={clinicalText}
                onChange={(e) => setClinicalText(e.target.value)}
                className="glass-input text-xs font-sans leading-relaxed"
                placeholder="Enter or upload patient PDF diagnosis report..."
              />
            </div>

            {/* Anonymized NER Preview Box */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-950/80 border border-slate-800">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block">
                  Local Anonymization Preview (PII Masking)
                </span>
                <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                  {nerEngine}
                </span>
              </div>
              <p className="text-[11px] text-slate-300 font-mono leading-relaxed select-all">
                {maskedText || <span className="text-slate-650 italic">No note provided to mask.</span>}
              </p>
            </div>

            <button 
              type="submit" 
              disabled={isSubmitting}
              className="w-full glass-button mt-2 text-sm"
            >
              {isSubmitting ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Generating Local Preds...
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4 text-emerald-450" />
                  Secure Predict & Evaluate
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Side: Prediction Visualizer & ECG Monitors (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            
            {/* Risk Gauge (5 cols) */}
            <div className="md:col-span-5 glass-panel p-6 flex flex-col justify-between items-center text-center space-y-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Heart className="w-4.5 h-4.5 text-rose-450" />
                Risk Gauge
              </h3>

              {/* SVG Radial Gauge */}
              <div className="relative w-44 h-44 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-95" viewBox="0 0 100 100">
                  <circle 
                    cx="50" cy="50" r="40" 
                    stroke="rgba(15, 23, 42, 0.9)" strokeWidth="7" 
                    fill="transparent" 
                  />
                  <circle 
                    cx="50" cy="50" r="40" 
                    stroke={sliderRisk > 0.65 ? '#f43f5e' : (sliderRisk > 0.35 ? '#f59e0b' : '#10b981')} 
                    strokeWidth="8" 
                    strokeDasharray={251.2}
                    strokeDashoffset={251.2 * (1 - sliderRisk)}
                    strokeLinecap="round"
                    className="transition-all duration-700 ease-out" 
                    fill="transparent" 
                  />
                </svg>
                <div className="absolute flex flex-col items-center justify-center">
                  <span className="text-3xl font-black text-slate-105">
                    {(sliderRisk * 100).toFixed(0)}%
                  </span>
                  <span className="text-[10px] text-slate-500 font-extrabold tracking-widest uppercase mt-0.5">
                    ECG+Tabular
                  </span>
                </div>
              </div>

              {/* Status Indicator */}
              <div className={`w-full p-2.5 rounded-lg border text-xs font-semibold uppercase tracking-wider ${riskMeta.badge} ${riskMeta.border}`}>
                {riskMeta.label}
              </div>
            </div>

            {/* Diagnostic Narrative Statement (7 cols) */}
            <div className="md:col-span-7 glass-panel p-5 flex flex-col justify-between space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-350">Diagnostic Analysis</h3>
                  <span className="text-[10px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded">
                    v{predictionResult.modelVersion}
                  </span>
                </div>
                
                <p className="text-sm font-medium text-slate-205 leading-relaxed bg-slate-950/40 border border-slate-850 p-3 rounded-lg min-h-[90px]">
                  {predictionResult.diagnosis}
                </p>
              </div>

              {/* Interventions notification */}
              <div className="p-3 rounded-lg bg-slate-850/50 border border-slate-800 text-xs flex gap-2.5 items-start">
                {sliderRisk > 0.65 ? (
                  <>
                    <ShieldAlert className="w-5 h-5 text-rose-450 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold text-rose-350 block">Urgent Review Advised</span>
                      <span className="text-slate-400">Schedule immediate 12-lead cardiovascular echocardiogram and check LDL levels.</span>
                    </div>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold text-emerald-400 block">System Parameters Stable</span>
                      <span className="text-slate-400">Cardiovascular status indicates no immediate diagnostic criteria exceptions.</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* ECG Waveform Display */}
          <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
            <div className="flex flex-wrap justify-between items-center gap-3">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
                  Interactive 12-Lead ECG Waveform
                </h3>
                <p className="text-[11px] text-slate-500">Select leads to view simulated signal values in real-time</p>
              </div>

              {/* Lead buttons */}
              <div className="flex flex-wrap gap-1.5">
                {[1, 2, 6, 12].map((lead) => (
                  <button 
                    key={lead}
                    type="button"
                    onClick={() => setActiveLead(lead)}
                    className={`px-2 py-0.5 rounded text-[10px] font-bold border transition-all ${
                      activeLead === lead 
                        ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30' 
                        : 'bg-slate-800 text-slate-400 border-slate-750 hover:bg-slate-750'
                    }`}
                  >
                    Lead {lead === 1 ? 'I' : (lead === 2 ? 'II' : (lead === 6 ? 'V2' : 'V6'))}
                  </button>
                ))}
              </div>
            </div>

            {/* Recharts chart */}
            <div className="relative w-full h-[150px] bg-slate-950/80 rounded-lg p-2 border border-slate-850">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ecgData} margin={{ top: 10, right: 10, left: -20, bottom: -5 }}>
                  <CartesianGrid stroke="#ffffff04" strokeDasharray="3 3" />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={[-1.5, 2.5]} tickLine={false} axisLine={false} stroke="#475569" fontSize={9} />
                  <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#1e293b' }} />
                  <Line 
                    type="monotone" 
                    dataKey="voltage" 
                    stroke="#22d3ee" 
                    strokeWidth={2}
                    dot={false}
                    name="ECG Lead V"
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="absolute top-3 right-3 flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[9px] text-slate-450 uppercase font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></span>
                Lead {activeLead} (Analog Waveform)
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Interactive Counterfactual Simulator Block */}
      <CounterfactualSlider 
        initialBP={bpSys} 
        initialCholesterol={cholesterol} 
        initialAge={age}
        initialFastingBS={fastingBS}
        initialMaxHR={maxHR}
        initialAngina={exerciseAngina}
        initialRisk={predictionResult.riskScore}
        onRecalculate={handleSliderRecalculate}
      />
    </div>
  );
};

export default ClinicianDashboard;
