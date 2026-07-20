# MedShield FL — Frontend React.js Dashboard Blueprint (`Phase 7`)

This document outlines the UI component tree, state management, design system, and interactive features for the **React.js Dashboard Application** (`/frontend/`).

---

## 🎨 Design System & Visual Principles

* **Aesthetic**: Modern dark mode with glassmorphism panels, dynamic health gauges, vibrant gradient accents (emerald green to coral red for risk level), and responsive layouts.
* **Libraries**: React 18, Recharts / Chart.js (for ECG signals & FL convergence graphs), Lucide-React (icons), Axios (API client).

---

## 🧩 Component Architecture (`/frontend/src/`)

```
src/
├── assets/                  # CSS styles, glassmorphism utilities, fonts
├── components/
│   ├── common/              # Navbar, Sidebar, Card, Button, Modal
│   ├── clinician/
│   │   ├── DiagnosticForm.jsx   # ECG upload, Masked text, Tabular inputs
│   │   ├── RiskGauge.jsx        # Animated radial health gauge (0-100%)
│   │   ├── ECGWaveform.jsx      # Interactive 12-lead time-series graph
│   │   └── CounterfactualSlider.jsx # "What-If" interactive parameter sliders
│   ├── patient/
│   │   ├── HealthSummary.jsx    # Anonymized report summary
│   │   └── RecommendationCard.jsx # Actionable lifestyle recommendations
│   └── admin/
│       ├── HospitalNodesTable.jsx # Federated hospital status & license keys
│       └── FLTrainingMonitor.jsx  # Real-time round progress bar & global loss curve
├── pages/
│   ├── LoginPage.jsx
│   ├── ClinicianDashboard.jsx
│   ├── PatientPortal.jsx
│   └── AdminDashboard.jsx
└── services/
    ├── api.js               # Axios instance with Bearer token interceptor
    └── auth.js              # Auth context provider & role guard
```

---

## 🖥️ Core Views Specification

### 1. Clinician Diagnostic View (`ClinicianDashboard.jsx`)

#### A. Interactive Counterfactual Slider Component (`CounterfactualSlider.jsx`)
Allows clinicians to adjust patient lifestyle metrics in real-time to visualize predicted risk reduction:

```jsx
import React, { useState } from 'react';

export const CounterfactualSlider = ({ initialBP, initialCholesterol, initialRisk, onRecalculate }) => {
  const [systolicBP, setSystolicBP] = useState(initialBP);
  const [cholesterol, setCholesterol] = useState(initialCholesterol);
  const [projectedRisk, setProjectedRisk] = useState(initialRisk);

  const handleBPChange = (e) => {
    const newBP = Number(e.target.value);
    setSystolicBP(newBP);
    // Simulate real-time model recalculation call
    const deltaRisk = ((newBP - 120) / 100) * 0.25;
    setProjectedRisk(Math.max(0.15, Math.min(0.95, initialRisk + deltaRisk)));
  };

  return (
    <div className="glass-panel p-6 rounded-xl border border-slate-700 bg-slate-900/60 backdrop-blur-md">
      <h3 className="text-xl font-bold text-slate-100 mb-4">Counterfactual "What-If" Target Simulator</h3>
      
      <div className="mb-4">
        <label className="block text-sm text-slate-300 mb-2">Target Systolic Blood Pressure: {systolicBP} mmHg</label>
        <input 
          type="range" min="90" max="190" value={systolicBP} onChange={handleBPChange}
          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
        />
      </div>

      <div className="mt-6 p-4 rounded-lg bg-slate-800/80 border border-slate-700 flex justify-between items-center">
        <span className="text-slate-300 font-medium">Projected Heart Disease Risk:</span>
        <span className={`text-2xl font-black ${projectedRisk > 0.6 ? 'text-red-400' : 'text-emerald-400'}`}>
          {(projectedRisk * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
};
```

---

### 2. Federated Learning Monitor View (`AdminDashboard.jsx`)

Displays real-time training convergence across connected hospital clients:

```jsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export const FLTrainingMonitor = ({ flRoundsData }) => {
  return (
    <div className="glass-panel p-6 rounded-xl bg-slate-900/60">
      <h2 className="text-xl font-bold text-slate-100 mb-4">Federated Model Training Convergence</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={flRoundsData}>
          <XAxis dataKey="round" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
          <Line type="monotone" dataKey="global_accuracy" stroke="#10b981" strokeWidth={3} name="Global Accuracy" />
          <Line type="monotone" dataKey="global_loss" stroke="#f43f5e" strokeWidth={2} name="Global Loss" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
```

---

## 🛠️ App Setup & Initialization Commands

To create and run the React frontend:

```bash
# Initialize Vite React app in /frontend
npx -y create-vite@latest frontend --template react

# Install UI dependencies
cd frontend
npm install axios recharts lucide-react clsx tailwindcss

# Start Vite dev server (Port 5173)
npm run dev
```

---

## ✅ Phase 7 Verification Checklist
- [ ] React 18 application initialized in `frontend/`
- [ ] Clinician View includes `DiagnosticForm`, `RiskGauge`, and `ECGWaveform`
- [ ] Interactive `CounterfactualSlider` enables real-time "What-If" parameter simulation
- [ ] Admin View displays `FLTrainingMonitor` global accuracy graph
- [ ] Axios services connect cleanly to FastAPI `/prediction` and `/federation` endpoints
