import React, { useState } from 'react';
import { 
  Database, 
  Cpu, 
  Plus, 
  RefreshCw, 
  Key, 
  Sliders, 
  Play,
  Trash2
} from 'lucide-react';
import FLTrainingMonitor from '../components/admin/FLTrainingMonitor';

// Default FL training rounds mock list
const defaultRounds = [
  { round: 1, global_accuracy: 0.58, global_loss: 0.65, active_nodes: 3, aggregation_strategy: 'FedAvg' },
  { round: 2, global_accuracy: 0.67, global_loss: 0.48, active_nodes: 3, aggregation_strategy: 'FedAvg' },
  { round: 3, global_accuracy: 0.74, global_loss: 0.38, active_nodes: 3, aggregation_strategy: 'FedAvg' },
  { round: 4, global_accuracy: 0.81, global_loss: 0.31, active_nodes: 3, aggregation_strategy: 'FedAvg' },
  { round: 5, global_accuracy: 0.84, global_loss: 0.28, active_nodes: 3, aggregation_strategy: 'FedAvg' }
];

export const AdminDashboard = () => {
  // FL rounds state
  const [roundsData, setRoundsData] = useState(defaultRounds);
  const [activeTab, setActiveTab] = useState('nodes'); // nodes, settings
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [strategy, setStrategy] = useState('FedAvg');
  const [epochs, setEpochs] = useState(5);
  const [minClients, setMinClients] = useState(3);

  // Nodes list state
  const [nodes, setNodes] = useState([
    { id: 'hosa', name: 'Hospital Alpha (St. Jude)', records: 150, status: 'Online', lastActive: '2m ago', key: 'MSFL-A91X-8801' },
    { id: 'hosb', name: 'Hospital Beta (Mercy General)', records: 200, status: 'Online', lastActive: '5m ago', key: 'MSFL-B77Y-4982' },
    { id: 'hosc', name: 'Hospital Gamma (City Health)', records: 180, status: 'Online', lastActive: '12m ago', key: 'MSFL-G34Z-2804' }
  ]);

  // Modal / Form input state for node registration
  const [showAddForm, setShowAddForm] = useState(false);
  const [newNodeName, setNewNodeName] = useState('');
  const [newNodeRecords, setNewNodeRecords] = useState(120);

  // Handle FL Round execution simulation
  const handleTriggerRound = () => {
    if (isTraining) return;
    setIsTraining(true);
    setTrainingProgress(0);

    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          
          // Complete training: Add new round to Recharts list
          setRoundsData((curr) => {
            const nextRoundNum = curr.length + 1;
            // Convergence curve: accuracy increases, loss degrades
            const lastRound = curr[curr.length - 1];
            const accGrowth = (0.95 - lastRound.global_accuracy) * 0.22;
            const lossDecay = lastRound.global_loss * 0.15;
            
            return [
              ...curr,
              {
                round: nextRoundNum,
                global_accuracy: Number((lastRound.global_accuracy + accGrowth).toFixed(3)),
                global_loss: Number((lastRound.global_loss - lossDecay).toFixed(4)),
                active_nodes: nodes.filter(n => n.status === 'Online').length,
                aggregation_strategy: strategy
              }
            ];
          });
          
          setIsTraining(false);
          return 100;
        }
        return prev + 10;
      });
    }, 150);
  };

  // Node registration
  const handleRegisterNode = (e) => {
    e.preventDefault();
    if (!newNodeName) return;

    // Generate simulated license key
    const hex = Math.floor(Math.random() * 9000 + 1000);
    const generatedKey = `MSFL-${newNodeName.substring(0, 3).toUpperCase()}-${hex}`;

    const newNodeObj = {
      id: `hos-${Date.now()}`,
      name: newNodeName,
      records: Number(newNodeRecords),
      status: 'Online',
      lastActive: 'Just now',
      key: generatedKey
    };

    setNodes([...nodes, newNodeObj]);
    setNewNodeName('');
    setNewNodeRecords(120);
    setShowAddForm(false);
  };

  const handleDeRegisterNode = (id) => {
    setNodes(nodes.filter(n => n.id !== id));
  };

  const handleToggleStatus = (id) => {
    setNodes(nodes.map(n => {
      if (n.id === id) {
        return {
          ...n,
          status: n.status === 'Online' ? 'Offline' : 'Online',
          lastActive: n.status === 'Online' ? '1h ago' : 'Just now'
        };
      }
      return n;
    }));
  };

  return (
    <div className="space-y-6">
      
      {/* Admin Panel Header */}
      <div className="glass-panel p-6 border-slate-700/60 bg-slate-900/40">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h2 className="text-xl font-black text-slate-100 flex items-center gap-2.5">
              <Cpu className="w-6 h-6 text-indigo-400" />
              Federated Console & Security Coordinator
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Configure federated learning protocols, aggregate localized matrices weights, and monitor client node keys.
            </p>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setActiveTab('nodes')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition-all ${
                activeTab === 'nodes' 
                  ? 'bg-indigo-500/20 text-indigo-350 border-indigo-500/30' 
                  : 'bg-slate-800 text-slate-400 border-slate-750 hover:bg-slate-700'
              }`}
            >
              Control Nodes
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition-all ${
                activeTab === 'settings' 
                  ? 'bg-indigo-500/20 text-indigo-350 border-indigo-500/30' 
                  : 'bg-slate-800 text-slate-400 border-slate-750 hover:bg-slate-700'
              }`}
            >
              FL Parameters
            </button>
          </div>
        </div>
      </div>

      {/* Primary Side-by-Side: Convergence and controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Side: Recharts FL Curves & Control buttons (8 Cols) */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* Training Chart Monitor */}
          <FLTrainingMonitor flRoundsData={roundsData} />

          {/* Trigger Round Control Panel */}
          <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-350">
                  Federated Optimization Actions
                </h3>
                <p className="text-[11px] text-slate-500">Aggregate new weights metrics from connected hospitals</p>
              </div>

              {!isTraining ? (
                <button 
                  onClick={handleTriggerRound}
                  className="glass-button text-xs gap-1.5 px-4 font-bold"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  Trigger FL Round #{roundsData.length + 1}
                </button>
              ) : (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-400 font-bold uppercase">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  SecAgg aggregation in progress ({trainingProgress}%)
                </div>
              )}
            </div>

            {/* Aggregation progress animator bar */}
            {isTraining && (
              <div className="space-y-1.5">
                <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-850">
                  <div 
                    className="h-full bg-gradient-to-r from-emerald-500 via-indigo-500 to-cyan-400 transition-all duration-300"
                    style={{ width: `${trainingProgress}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>Contacting hospital APIs...</span>
                  <span>Verifying homomorphic verification matrix...</span>
                  <span>Completed</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Hospital node list list table (4 Cols) */}
        <div className="lg:col-span-4 space-y-6">
          {activeTab === 'nodes' ? (
            <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800 pb-2.5">
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                  <Database className="w-4.5 h-4.5 text-indigo-400" />
                  Active Hospital Nodes
                </h3>
                <button 
                  onClick={() => setShowAddForm(!showAddForm)}
                  className="p-1 rounded bg-slate-800 border border-slate-700 text-slate-300 hover:text-slate-100 transition"
                  title="Register new Node"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>

              {/* Add form */}
              {showAddForm && (
                <form onSubmit={handleRegisterNode} className="p-3.5 rounded-lg bg-slate-950/70 border border-slate-800 space-y-3">
                  <h4 className="text-[11px] font-bold text-slate-350 uppercase">Register Local Client API</h4>
                  <div className="space-y-2">
                    <input 
                      type="text" 
                      placeholder="Hospital Alias Name" 
                      value={newNodeName}
                      onChange={(e) => setNewNodeName(e.target.value)}
                      className="glass-input text-xs w-full"
                      required
                    />
                    <input 
                      type="number" 
                      placeholder="Records Size Count" 
                      value={newNodeRecords}
                      onChange={(e) => setNewNodeRecords(e.target.value)}
                      className="glass-input text-xs w-full"
                      required
                    />
                  </div>
                  <div className="flex gap-2">
                    <button type="submit" className="glass-button text-[10px] py-1 px-3">Generate Auth Key</button>
                    <button type="button" onClick={() => setShowAddForm(false)} className="text-[10px] text-slate-500 font-semibold px-2">Cancel</button>
                  </div>
                </form>
              )}

              {/* Node status indicators list */}
              <div className="space-y-3">
                {nodes.map((node) => (
                  <div key={node.id} className="p-3 rounded-lg glass-card space-y-2.5">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="text-xs font-bold text-slate-200">{node.name}</h4>
                        <div className="flex gap-2 mt-1">
                          <span className="text-[9px] text-slate-500 font-medium font-mono">{node.records} records</span>
                          <span className="text-[9px] text-slate-600 font-medium font-mono">•</span>
                          <span className="text-[9px] text-slate-500 font-medium font-mono">{node.lastActive}</span>
                        </div>
                      </div>
                      
                      <button
                        onClick={() => handleToggleStatus(node.id)}
                        className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${
                          node.status === 'Online'
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : 'bg-slate-800 text-slate-500 border border-slate-700'
                        }`}
                      >
                        {node.status}
                      </button>
                    </div>

                    {/* Security Token credentials check */}
                    <div className="pt-2 border-t border-slate-900 flex justify-between items-center text-[10px]">
                      <div className="flex items-center gap-1 font-mono text-[9px] text-slate-400">
                        <Key className="w-3 h-3 text-cyan-400" />
                        Credentials: {node.key}
                      </div>

                      <button 
                        onClick={() => handleDeRegisterNode(node.id)}
                        className="text-slate-600 hover:text-rose-400 p-1 transition"
                        title="Deregister node connection"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="glass-panel p-5 border-slate-700/60 bg-slate-900/60 space-y-4">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Sliders className="w-4.5 h-4.5 text-indigo-400" />
                FL Parameters Configuration
              </h3>

              <div className="space-y-4">
                {/* Aggregator Strategy Choice */}
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-450 font-bold block uppercase">Aggregation Strategy</label>
                  <select 
                    value={strategy}
                    onChange={(e) => setStrategy(e.target.value)}
                    className="glass-input text-xs w-full"
                  >
                    <option value="FedAvg">FedAvg (Federated Averaging)</option>
                    <option value="FedProx">FedProx (Federate Proximal - handles non-IID)</option>
                    <option value="FedOpt">FedOpt (Federated Optimization)</option>
                  </select>
                </div>

                {/* Local Training Epochs count */}
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-450 font-bold block uppercase">Local Epochs</label>
                  <input 
                    type="number" 
                    value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                    className="glass-input text-xs w-full font-mono"
                    min="1" max="50"
                  />
                  <span className="text-[10px] text-slate-550 italic">Specifies local gradient steps per client</span>
                </div>

                {/* Client participants count */}
                <div className="space-y-1.5">
                  <label className="text-xs text-slate-450 font-bold block uppercase">Minimum Connected Clients</label>
                  <input 
                    type="number" 
                    value={minClients}
                    onChange={(e) => setMinClients(Number(e.target.value))}
                    className="glass-input text-xs w-full font-mono"
                    min="1" max="100"
                  />
                </div>

                <div className="p-3.5 rounded-lg bg-indigo-950/20 border border-indigo-500/20 text-[11px] text-indigo-350 leading-relaxed font-sans">
                  The coordinator handles homomorphically obscured gradients using SecAgg protocol. Key exchanges are executed locally without exposing weights vectors.
                </div>
              </div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
};

export default AdminDashboard;
