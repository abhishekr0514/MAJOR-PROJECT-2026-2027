import React, { useState } from 'react';
import { 
  Heart, 
  Lock, 
  ChevronRight, 
  Sparkles, 
  ShieldCheck, 
  Apple, 
  Dumbbell, 
  CheckCircle
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// Historical patient measurements for Recharts
const mockRiskHistory = [
  { month: 'Jan', risk: 0.76 },
  { month: 'Feb', risk: 0.72 },
  { month: 'Mar', risk: 0.68 },
  { month: 'Apr', risk: 0.64 }
];

export const PatientPortal = () => {
  const [patientCode] = useState('PAT-HOSA-0042');
  const [age] = useState(58);
  const [gender] = useState('M');
  const [systolicBP] = useState(142);
  const [diastolicBP] = useState(88);
  const [cholesterol] = useState(245);
  const [fastingBS] = useState(115);
  const [maxHR] = useState(140);
  const [exerciseAngina] = useState(1);
  const [currentRisk] = useState(0.64);

  // Recommendations data generator based on parameters
  const getCustomRecommendations = () => {
    const list = [];
    if (cholesterol > 200) {
      list.push({
        id: 'chol',
        category: 'Nutrition',
        title: 'Lower Serum Cholesterol',
        description: 'Increase soluble fiber (oat bran, legumes) and take 2g Omega-3 daily. Restrict saturated fats to <7% of daily calories.',
        icon: <Apple className="w-5 h-5 text-emerald-400" />,
        action: 'Track dietary fat inputs weekly'
      });
    }
    if (systolicBP > 130) {
      list.push({
        id: 'bp',
        category: 'Cardiovascular',
        title: 'Manage Blood Pressure (DASH)',
        description: 'Reduce sodium intake to under 1,500mg/day. Prioritize magnesium-rich spinach, avocados, and potassium-balanced items.',
        icon: <Dumbbell className="w-5 h-5 text-cyan-400" />,
        action: 'Measure resting sys BP daily'
      });
    }
    if (fastingBS > 100) {
      list.push({
        id: 'fbs',
        category: 'Metabolic',
        title: 'Fasting Blood Sugar Guarding',
        description: 'Reduce glycemic load. Walk for 15 minutes immediately post major dinners to improve insulin sensitivity.',
        icon: <Sparkles className="w-5 h-5 text-amber-400" />,
        action: 'Review carb counts in meals'
      });
    }
    return list;
  };

  const recommendations = getCustomRecommendations();

  const getRiskLabel = (risk) => {
    if (risk < 0.35) return { text: 'Low Susceptibility', color: 'text-emerald-400', badge: 'risk-badge-low' };
    if (risk < 0.65) return { text: 'Moderate Susceptibility', color: 'text-amber-400', badge: 'risk-badge-moderate' };
    return { text: 'High Susceptibility', color: 'text-rose-400', badge: 'risk-badge-high' };
  };

  const riskDetails = getRiskLabel(currentRisk);

  return (
    <div className="space-y-6">
      
      {/* Welcome Banner */}
      <div className="glass-panel p-6 border-slate-700/60 bg-gradient-to-r from-indigo-950/20 via-slate-900/40 to-cyan-950/20">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider">Patient Portal Account</span>
              <span className="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 font-mono">{patientCode}</span>
            </div>
            <h2 className="text-xl font-black text-slate-100 flex items-center gap-2">
              <ShieldCheck className="w-6 h-6 text-emerald-400" />
              Your Health Summary
            </h2>
            <p className="text-xs text-slate-400 mt-1">Review your cardiovascular diagnostic metrics, recovery targets, and privacy information.</p>
          </div>

          <div className="flex items-center gap-2.5 bg-slate-950/70 p-2.5 rounded-xl border border-slate-850">
            <Lock className="w-4 h-4 text-emerald-400" />
            <div className="text-left">
              <span className="text-[10px] text-emerald-400 font-semibold block uppercase">Zero PII Leakage Guarantee</span>
              <span className="text-[9px] text-slate-500">Identifiers are masked locally prior to model evaluation rounds.</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Side: Summary and Risk Trend History (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          
          {/* Anonymized Health Profile Summary */}
          <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-350 border-b border-slate-800 pb-2 flex items-center justify-between">
              <span>Cardiovascular Attributes Table</span>
              <span className="text-[10px] text-slate-500 font-normal uppercase">Last Synced: Today</span>
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3.5">
              <div className="glass-card p-3">
                <span className="text-[10px] text-slate-450 block font-semibold mb-0.5">Demographics</span>
                <span className="text-sm font-bold text-slate-100">{age} yrs • {gender === 'M' ? 'Male' : 'Female'}</span>
              </div>

              <div className="glass-card p-3">
                <span className="text-[10px] text-slate-450 block font-semibold mb-0.5">Blood Pressure</span>
                <span className="text-sm font-bold text-slate-100">{systolicBP} / {diastolicBP} <span className="text-[10px] text-slate-500">mmHg</span></span>
              </div>

              <div className="glass-card p-3">
                <span className="text-[10px] text-slate-450 block font-semibold mb-0.5">Cholesterol Level</span>
                <span className="text-sm font-bold text-slate-100">{cholesterol} <span className="text-[10px] text-slate-500">mg/dL</span></span>
              </div>

              <div className="glass-card p-3">
                <span className="text-[10px] text-slate-450 block font-semibold mb-0.5">Fasting Blood Sugar</span>
                <span className="text-sm font-bold text-slate-100">{fastingBS} <span className="text-[10px] text-slate-500">mg/dL</span></span>
              </div>

              <div className="glass-card p-3">
                <span className="text-[10px] text-slate-450 block font-semibold mb-0.5">Max Heart Rate Achieved</span>
                <span className="text-sm font-bold text-slate-100">{maxHR} <span className="text-[10px] text-slate-500">bpm</span></span>
              </div>

              <div className="glass-card p-3">
                <span className="text-[10px] text-slate-450 block font-semibold mb-0.5">Exercise Angina</span>
                <span className="text-sm font-bold text-slate-100">{exerciseAngina ? 'Yes (Positive)' : 'No'}</span>
              </div>
            </div>
          </div>

          {/* Risk Progression History */}
          <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
            <div>
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
                Cardiovascular Stability Indicator Track
              </h3>
              <p className="text-[11px] text-slate-500 mt-0.5">Historical risk projections calculated across monthly updates</p>
            </div>

            <div className="relative w-full h-[180px] bg-slate-950/60 rounded-xl p-3 border border-slate-850">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={mockRiskHistory} margin={{ top: 10, right: 10, left: -25, bottom: -5 }}>
                  <XAxis dataKey="month" stroke="#475569" fontSize={10} tickLine={false} />
                  <YAxis domain={[0, 1.0]} stroke="#475569" fontSize={10} tickLine={false} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                  <defs>
                    <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <Area 
                    type="monotone" 
                    dataKey="risk" 
                    stroke="#f59e0b" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#riskGrad)"
                    name="Risk Probability"
                  />
                </AreaChart>
              </ResponsiveContainer>
              <div className="absolute top-3 right-3 flex items-center gap-1 bg-slate-900 border border-slate-800 text-[9px] text-slate-450 uppercase font-mono px-2 py-0.5 rounded">
                Trend: -12% Decrease (Improving)
              </div>
            </div>
          </div>

        </div>

        {/* Right Side: Recommendations & Privacy Panel (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          
          {/* Active Risk Susceptibility Panel */}
          <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 text-center space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-350 flex justify-center items-center gap-1.5">
              <Heart className="w-4.5 h-4.5 text-rose-500" />
              Calculated Susceptibility Index
            </h3>

            <div className="relative w-36 h-36 mx-auto flex items-center justify-center">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" stroke="rgba(15,23,42,0.9)" strokeWidth="8" fill="transparent" />
                <circle 
                  cx="50" cy="50" r="40" 
                  stroke="#f59e0b" 
                  strokeWidth="8" 
                  strokeDasharray={251.2}
                  strokeDashoffset={251.2 * (1 - currentRisk)}
                  strokeLinecap="round"
                  fill="transparent" 
                />
              </svg>
              <div className="absolute text-center flex flex-col justify-center items-center">
                <span className={`text-3xl font-black ${riskDetails.color}`}>{(currentRisk * 100).toFixed(0)}%</span>
                <span className="text-[9px] text-slate-500 font-extrabold tracking-wider uppercase mt-0.5">Moderate Check</span>
              </div>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed max-w-sm mx-auto">
              Your profile indicates a <span className={`font-semibold ${riskDetails.color}`}>{riskDetails.text}</span> risk level. Following targets below can reduce this index level.
            </p>
          </div>

          {/* Actionable Lifestyle Targets */}
          <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
              Personalized Lifestyle Interventions
            </h3>

            <div className="space-y-3.5">
              {recommendations.map((rec) => (
                <div key={rec.id} className="p-3.5 rounded-lg bg-slate-950/65 border border-slate-850 space-y-2 hover:border-slate-800 transition-all duration-200">
                  <div className="flex justify-between items-start gap-2.5">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded-lg bg-slate-900 border border-slate-800">
                        {rec.icon}
                      </div>
                      <div>
                        <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wide block">{rec.category}</span>
                        <h4 className="text-xs font-bold text-slate-200">{rec.title}</h4>
                      </div>
                    </div>
                    <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-slate-905 border border-slate-850 text-slate-400 uppercase">Recommended</span>
                  </div>

                  <p className="text-xs text-slate-400 leading-relaxed pl-1">
                    {rec.description}
                  </p>

                  <div className="pt-1.5 border-t border-slate-900 flex justify-between items-center text-[10px] text-emerald-450 font-bold select-none">
                    <span className="flex items-center gap-1 pl-1">
                      <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                      Target Action: {rec.action}
                    </span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};

export default PatientPortal;
