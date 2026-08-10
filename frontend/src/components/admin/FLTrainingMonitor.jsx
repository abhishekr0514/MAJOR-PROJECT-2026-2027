import React from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { Cpu, Award, TrendingDown, Layers, Zap } from 'lucide-react';

/**
 * Custom dark tooltip for Recharts curves
 */
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-panel p-3 bg-slate-950/90 border border-slate-700/80 shadow-2xl rounded-lg font-sans text-xs">
        <p className="font-bold text-slate-100 mb-1.5 font-mono text-[11px] uppercase tracking-wide">
          Round #{label} Results
        </p>
        {payload.map((entry, idx) => (
          <div key={idx} className="flex items-center gap-2 mt-1">
            <span 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: entry.stroke }}
            />
            <span className="text-slate-400 font-medium">{entry.name}:</span>
            <span className="font-bold font-mono text-slate-200">
              {entry.name.includes('Accuracy') 
                ? `${(entry.value * 100).toFixed(1)}%` 
                : entry.value.toFixed(4)
              }
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

/**
 * FLTrainingMonitor Component
 * Render federated deep learning convergence statistics
 */
export const FLTrainingMonitor = ({ flRoundsData = [] }) => {
  // Compute latest round stats
  const latestRound = flRoundsData[flRoundsData.length - 1] || {
    round: 0,
    global_accuracy: 0,
    global_loss: 0,
    active_nodes: 0,
    aggregation_strategy: 'FedAvg'
  };

  // Find accuracy/loss improvements
  const showImprovements = flRoundsData.length > 1;
  const prevRound = showImprovements ? flRoundsData[flRoundsData.length - 2] : null;
  const accuracyDelta = prevRound ? (latestRound.global_accuracy - prevRound.global_accuracy) : 0;
  const lossDelta = prevRound ? (latestRound.global_loss - prevRound.global_loss) : 0;

  return (
    <div className="glass-panel p-6 border border-slate-700/60 bg-slate-900/60 backdrop-blur-md space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b border-slate-800/80 pb-4 gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-100">Global Training Convergence</h2>
            <p className="text-xs text-slate-400 font-medium">Real-time parameters synchronization across hospital nodes</p>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-slate-850 px-3 py-1.5 rounded-full border border-slate-800 text-xs">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
          <span className="text-slate-350 font-semibold uppercase tracking-wider text-[10px]">
            Aggregation: {latestRound.aggregation_strategy || 'FedAvg (SecAgg)'}
          </span>
        </div>
      </div>

      {/* Grid of latest metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Round card */}
        <div className="glass-card p-3.5 flex items-center justify-between">
          <div>
            <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">FL Round</span>
            <span className="text-xl font-bold font-mono text-slate-100">#{latestRound.round}</span>
          </div>
          <Layers className="w-8 h-8 text-indigo-500/40" />
        </div>

        {/* Accuracy Card */}
        <div className="glass-card p-3.5 flex items-center justify-between">
          <div>
            <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Global Accuracy</span>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xl font-bold font-mono text-emerald-400">
                {(latestRound.global_accuracy * 100).toFixed(1)}%
              </span>
              {showImprovements && accuracyDelta !== 0 && (
                <span className={`text-[10px] font-bold ${accuracyDelta >= 0 ? 'text-emerald-400' : 'text-rose-450'}`}>
                  {accuracyDelta >= 0 ? '+' : ''}{(accuracyDelta * 100).toFixed(1)}%
                </span>
              )}
            </div>
          </div>
          <Award className="w-8 h-8 text-emerald-500/40" />
        </div>

        {/* Loss Card */}
        <div className="glass-card p-3.5 flex items-center justify-between">
          <div>
            <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Global Loss</span>
            <div className="flex items-baseline gap-1.5">
              <span className="text-xl font-bold font-mono text-rose-400">
                {latestRound.global_loss.toFixed(4)}
              </span>
              {showImprovements && lossDelta !== 0 && (
                <span className={`text-[10px] font-bold ${lossDelta <= 0 ? 'text-emerald-400' : 'text-rose-450'}`}>
                  {lossDelta <= 0 ? '' : '+'}{(lossDelta).toFixed(3)}
                </span>
              )}
            </div>
          </div>
          <TrendingDown className="w-8 h-8 text-rose-500/40" />
        </div>

        {/* Sync Nodes Card */}
        <div className="glass-card p-3.5 flex items-center justify-between">
          <div>
            <span className="text-[10px] text-slate-500 block uppercase font-bold tracking-wider">Active Nodes</span>
            <span className="text-xl font-bold font-mono text-slate-100">
              {latestRound.active_nodes || 3} <span className="text-xs text-emerald-500 font-normal">Online</span>
            </span>
          </div>
          <Zap className="w-8 h-8 text-amber-500/40" />
        </div>
      </div>

      {/* Line Chart */}
      <div className="relative w-full rounded-xl bg-slate-950/60 p-4 border border-slate-800/80">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart 
            data={flRoundsData} 
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <CartesianGrid stroke="#1e293b/40" strokeDasharray="3 3" vertical={false} />
            <XAxis 
              dataKey="round" 
              stroke="#64748b" 
              fontSize={10}
              tickLine={false} 
              axisLine={false}
              padding={{ left: 10, right: 10 }}
            />
            <YAxis 
              stroke="#64748b" 
              fontSize={10}
              tickLine={false} 
              axisLine={false}
              domain={[0, 1.0]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              verticalAlign="top" 
              height={36} 
              iconType="circle" 
              iconSize={8}
              wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }}
            />
            <Line 
              type="monotone" 
              dataKey="global_accuracy" 
              stroke="#10b981" 
              strokeWidth={3} 
              activeDot={{ r: 6, stroke: '#090d16', strokeWidth: 2 }}
              name="Global Accuracy" 
            />
            <Line 
              type="monotone" 
              dataKey="global_loss" 
              stroke="#f43f5e" 
              strokeWidth={2.5} 
              activeDot={{ r: 5, stroke: '#090d16', strokeWidth: 2 }}
              name="Global Loss" 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Logs/Aggregator detail */}
      <div className="border-t border-slate-850 pt-4 flex flex-col md:flex-row justify-between text-xs text-slate-400 gap-2">
        <span>Aggregation status: <span className="text-emerald-400 font-semibold">SUCCESS</span></span>
        <span>Secure Secure Aggregation Protocol: <span className="font-mono text-cyan-400 font-semibold">Active</span></span>
        <span>Model Version: <span className="font-mono text-slate-300 font-semibold">1.2.0-BiLSTM+BERT</span></span>
      </div>
    </div>
  );
};

export default FLTrainingMonitor;
