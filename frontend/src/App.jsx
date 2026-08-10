import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  Lock, 
  Cpu, 
  User, 
  Activity, 
  Database,
  ExternalLink,
  LogIn,
  LogOut,
  CheckCircle2
} from 'lucide-react';
import ClinicianDashboard from './pages/ClinicianDashboard';
import PatientPortal from './pages/PatientPortal';
import AdminDashboard from './pages/AdminDashboard';
import LoginModal from './components/auth/LoginModal';

export default function App() {
  const [currentRole, setCurrentRole] = useState('clinician'); // clinician, patient, admin
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [userEmail, setUserEmail] = useState(localStorage.getItem('user_email') || '');
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('access_token'));

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    const email = localStorage.getItem('user_email');
    if (token) {
      setIsAuthenticated(true);
      if (email) setUserEmail(email);
    }
  }, []);

  const handleLoginSuccess = (email) => {
    setIsAuthenticated(true);
    setUserEmail(email);
  };

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_email');
    setIsAuthenticated(false);
    setUserEmail('');
  };

  const renderActiveView = () => {
    switch (currentRole) {
      case 'clinician':
        return <ClinicianDashboard />;
      case 'patient':
        return <PatientPortal />;
      case 'admin':
        return <AdminDashboard />;
      default:
        return <ClinicianDashboard />;
    }
  };

  return (
    <div className="min-h-screen p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-6">
      
      {/* Top Main Navigation Header */}
      <header className="glass-panel p-4 flex flex-col md:flex-row justify-between items-center gap-4 transition-all duration-300">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
            <ShieldCheck className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 via-teal-300 to-indigo-400">
              MedShield FL
            </h1>
            <p className="text-xs text-slate-400 font-medium tracking-wide">
              Privacy-Preserving Multimodal Federated Diagnostic Engine
            </p>
          </div>
        </div>

        {/* Global connection telemetry indicators & Auth status */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-950/40 border border-slate-850 text-xs text-slate-350 select-none">
            <Lock className="w-3.5 h-3.5 text-emerald-400" />
            <span>SecAgg Active</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-950/40 border border-slate-850 text-xs text-slate-350 select-none">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>FL Convergence Round #5</span>
          </div>

          {/* Authentication Badge */}
          {isAuthenticated ? (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span className="font-semibold">{userEmail || 'Connected'}</span>
              <button 
                onClick={handleLogout}
                title="Log out"
                className="ml-1 p-1 hover:bg-emerald-500/20 rounded-full transition-colors"
              >
                <LogOut className="w-3.5 h-3.5 text-emerald-400 hover:text-rose-400" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setIsLoginModalOpen(true)}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/40 text-xs font-bold text-indigo-300 hover:text-indigo-200 transition-all duration-200 shadow-lg shadow-indigo-500/10"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign In (JWT)</span>
            </button>
          )}
        </div>
      </header>

      {/* Role Selector Tabs (Glass segment controller) */}
      <div className="flex justify-center">
        <div className="inline-flex p-1.5 rounded-xl bg-slate-950/60 border border-slate-850/80 shadow-2xl backdrop-blur-lg">
          <button
            onClick={() => setCurrentRole('clinician')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-extrabold uppercase tracking-wider rounded-lg transition-all duration-200 ${
              currentRole === 'clinician'
                ? 'bg-gradient-to-r from-emerald-500/10 to-emerald-500/20 text-emerald-400 border border-emerald-500/25 shadow-inner'
                : 'text-slate-400 hover:text-slate-205 border border-transparent'
            }`}
          >
            <Activity className="w-4 h-4" />
            Clinician Dashboard
          </button>
          
          <button
            onClick={() => setCurrentRole('patient')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-extrabold uppercase tracking-wider rounded-lg transition-all duration-200 ${
              currentRole === 'patient'
                ? 'bg-gradient-to-r from-amber-500/10 to-amber-500/20 text-amber-400 border border-amber-500/25 shadow-inner'
                : 'text-slate-400 hover:text-slate-205 border border-transparent'
            }`}
          >
            <User className="w-4 h-4" />
            Patient Portal
          </button>
          
          <button
            onClick={() => setCurrentRole('admin')}
            className={`flex items-center gap-2 px-4 py-2 text-xs font-extrabold uppercase tracking-wider rounded-lg transition-all duration-200 ${
              currentRole === 'admin'
                ? 'bg-gradient-to-r from-indigo-500/10 to-indigo-500/20 text-indigo-400 border border-indigo-500/25 shadow-inner'
                : 'text-slate-400 hover:text-slate-205 border border-transparent'
            }`}
          >
            <Database className="w-4 h-4" />
            Admin Console
          </button>
        </div>
      </div>

      {/* Main Page View Shell */}
      <main className="transition-all duration-300">
        {renderActiveView()}
      </main>

      {/* Authentication Modal */}
      <LoginModal 
        isOpen={isLoginModalOpen}
        onClose={() => setIsLoginModalOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />

      {/* Footer Branding */}
      <footer className="pt-6 border-t border-slate-900 flex flex-col sm:flex-row justify-between items-center text-[10px] text-slate-500 gap-2">
        <p>© 2026-2027 MedShield FL Consortium. All rights and metrics reserved.</p>
        <div className="flex gap-4">
          <a href="#docs" className="hover:text-slate-300 flex items-center gap-1">
            Technical Specification <ExternalLink className="w-2.5 h-2.5" />
          </a>
          <span>•</span>
          <a href="#pii" className="hover:text-slate-300">Zero Leakage Policy</a>
        </div>
      </footer>

    </div>
  );
}
