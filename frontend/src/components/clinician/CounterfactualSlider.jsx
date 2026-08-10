import React, { useState, useEffect } from 'react';
import { Sliders, RefreshCw, TrendingDown } from 'lucide-react';

/**
 * CounterfactualSlider component
 * Allows clinicians to adjust patient cardiovascular metrics in real-time
 * to simulate hypotheticals ("What-If" scenarios) and visualize risk reduction.
 */
export const CounterfactualSlider = ({ 
  initialBP = 142, 
  initialCholesterol = 245, 
  initialAge = 58,
  initialFastingBS = 115,
  initialMaxHR = 140,
  initialAngina = 1,
  initialRisk = 0.64, 
  onRecalculate 
}) => {
  const [systolicBP, setSystolicBP] = useState(initialBP);
  const [cholesterol, setCholesterol] = useState(initialCholesterol);
  const [age, setAge] = useState(initialAge);
  const [fastingBS, setFastingBS] = useState(initialFastingBS);
  const [maxHR, setMaxHR] = useState(initialMaxHR);
  const [exerciseAngina, setExerciseAngina] = useState(initialAngina);
  const [projectedRisk, setProjectedRisk] = useState(initialRisk);

  // Sync state with props if customer/patient profile changes
  useEffect(() => {
    setSystolicBP(initialBP);
    setCholesterol(initialCholesterol);
    setAge(initialAge);
    setFastingBS(initialFastingBS);
    setMaxHR(initialMaxHR);
    setExerciseAngina(initialAngina);
    setProjectedRisk(initialRisk);
  }, [initialBP, initialCholesterol, initialAge, initialFastingBS, initialMaxHR, initialAngina, initialRisk]);

  // Recalculate heart disease risk with simplified coefficient weights
  useEffect(() => {
    // Local calculation logic (mimics structural API predictions)
    const baseRisk = 0.45;
    const bpImpact = (systolicBP - 120) * 0.0038;
    const cholImpact = (cholesterol - 200) * 0.0022;
    const ageImpact = (age - 50) * 0.0045;
    const fbsImpact = (fastingBS - 100) * 0.0018;
    const hrImpact = (150 - maxHR) * 0.0025; // Lower max HR increases risk
    const anginaImpact = exerciseAngina ? 0.15 : -0.05;

    const raw = baseRisk + bpImpact + cholImpact + ageImpact + fbsImpact + hrImpact + anginaImpact;
    const finalRisk = Math.max(0.05, Math.min(0.98, raw));
    
    setProjectedRisk(finalRisk);

    if (onRecalculate) {
      onRecalculate({
        systolicBP,
        cholesterol,
        age,
        fastingBS,
        maxHR,
        exerciseAngina,
        projectedRisk: finalRisk
      });
    }
  }, [systolicBP, cholesterol, age, fastingBS, maxHR, exerciseAngina, onRecalculate]);

  const handleReset = () => {
    setSystolicBP(initialBP);
    setCholesterol(initialCholesterol);
    setAge(initialAge);
    setFastingBS(initialFastingBS);
    setMaxHR(initialMaxHR);
    setExerciseAngina(initialAngina);
    setProjectedRisk(initialRisk);
  };

  const riskDelta = projectedRisk - initialRisk;
  const riskReduced = riskDelta < 0;

  const getRiskColor = (risk) => {
    if (risk < 0.35) return 'text-emerald-400';
    if (risk < 0.65) return 'text-amber-400';
    return 'text-rose-400';
  };

  return (
    <div className="glass-panel p-6 border border-slate-700/60 bg-slate-900/60 backdrop-blur-md space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center border-b border-slate-800/80 pb-4">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-emerald-400" />
          <div>
            <h3 className="text-lg font-bold text-slate-100">Counterfactual "What-If" Simulator</h3>
            <p className="text-xs text-slate-400">Simulate parameters locally to achieve target health profiles</p>
          </div>
        </div>
        <button 
          onClick={handleReset}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-slate-700/60 text-xs font-semibold text-slate-350 hover:text-slate-100 transition-all duration-200"
          title="Reset to initial patient values"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Reset Profile
        </button>
      </div>

      {/* Sliders Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-5">
        {/* Systolic BP */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-350 font-medium">Systolic Blood Pressure</span>
            <span className="font-mono font-bold text-slate-100">{systolicBP} <span className="text-xs text-slate-500 font-normal">mmHg</span></span>
          </div>
          <input 
            type="range" min="90" max="190" value={systolicBP} 
            onChange={(e) => setSystolicBP(Number(e.target.value))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-slate-500 font-mono">
            <span>90 (Optimal)</span>
            <span>190 (Severe Crisis)</span>
          </div>
        </div>

        {/* Serum Cholesterol */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-350 font-medium">Serum Cholesterol</span>
            <span className="font-mono font-bold text-slate-100">{cholesterol} <span className="text-xs text-slate-500 font-normal">mg/dL</span></span>
          </div>
          <input 
            type="range" min="130" max="340" value={cholesterol} 
            onChange={(e) => setCholesterol(Number(e.target.value))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-slate-500 font-mono">
            <span>130 (Desirable)</span>
            <span>340 (High Risk)</span>
          </div>
        </div>

        {/* Fasting Blood Sugar */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-350 font-medium">Fasting Blood Sugar</span>
            <span className="font-mono font-bold text-slate-100">{fastingBS} <span className="text-xs text-slate-500 font-normal">mg/dL</span></span>
          </div>
          <input 
            type="range" min="70" max="220" value={fastingBS} 
            onChange={(e) => setFastingBS(Number(e.target.value))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-slate-500 font-mono">
            <span>70 (Normal)</span>
            <span>220 (Elevated Diabetes)</span>
          </div>
        </div>

        {/* Max Heart Rate */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-350 font-medium">Max Heart Rate Achieved</span>
            <span className="font-mono font-bold text-slate-100">{maxHR} <span className="text-xs text-slate-500 font-normal">bpm</span></span>
          </div>
          <input 
            type="range" min="80" max="200" value={maxHR} 
            onChange={(e) => setMaxHR(Number(e.target.value))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400"
          />
          <div className="flex justify-between text-[10px] text-slate-500 font-mono">
            <span>80 (Poor cardiovascular conditioning)</span>
            <span>200 (Excellent)</span>
          </div>
        </div>

        {/* Age Selector */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-350 font-medium">Age Shift Simulator</span>
            <span className="font-mono font-bold text-slate-100">{age} <span className="text-xs text-slate-500 font-normal">years</span></span>
          </div>
          <input 
            type="range" min="30" max="85" value={age} 
            onChange={(e) => setAge(Number(e.target.value))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 hover:accent-emerald-400"
          />
        </div>

        {/* Exercise Angina Selection Toggle */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-slate-850/60 border border-slate-800/80">
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-slate-200">Exercise Induced Angina</span>
            <span className="text-xs text-slate-400">Exertion causes immediate chest pain</span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setExerciseAngina(0)}
              className={`px-3 py-1 text-xs font-bold rounded-lg border transition-all ${
                exerciseAngina === 0 
                  ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50 shadow-inner' 
                  : 'bg-slate-800 text-slate-405 border-slate-700 hover:bg-slate-700/80'
              }`}
            >
              No
            </button>
            <button
              onClick={() => setExerciseAngina(1)}
              className={`px-3 py-1 text-xs font-bold rounded-lg border transition-all ${
                exerciseAngina === 1 
                  ? 'bg-rose-500/25 text-rose-450 border-rose-500/50 shadow-inner' 
                  : 'bg-slate-800 text-slate-405 border-slate-700 hover:bg-slate-700/80'
              }`}
            >
              Yes
            </button>
          </div>
        </div>
      </div>

      {/* Results Comparison Block */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col sm:flex-row justify-between items-center gap-4">
        <div>
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Estimated Diagnostic Outcome</span>
          <p className="text-xs text-slate-500 mt-0.5">Recalculated locally with multimodal weights coefficients</p>
        </div>

        <div className="flex items-center gap-4 border-l border-slate-850 pl-0 sm:pl-6 w-full sm:w-auto justify-between sm:justify-start">
          <div className="text-center sm:text-left">
            <span className="text-[10px] text-slate-500 block uppercase font-medium">Baseline</span>
            <span className="text-sm font-medium text-slate-400">{(initialRisk * 100).toFixed(1)}%</span>
          </div>

          <div className="text-2xl text-slate-600 font-light">→</div>

          <div className="text-center sm:text-left">
            <span className="text-[10px] text-slate-500 block uppercase font-medium">Counterfactual</span>
            <span className={`text-2xl font-black ${getRiskColor(projectedRisk)} transition-all duration-300`}>
              {(projectedRisk * 100).toFixed(1)}%
            </span>
          </div>

          {/* Delta calculation badge */}
          {Math.abs(riskDelta) >= 0.001 && (
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${
              riskReduced
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                : 'bg-rose-500/15 text-rose-400 border border-rose-500/20'
            }`}>
              {riskReduced ? <TrendingDown className="w-3.5 h-3.5" /> : null}
              {riskReduced ? '' : '+'}
              {(riskDelta * 100).toFixed(1)}%
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CounterfactualSlider;
