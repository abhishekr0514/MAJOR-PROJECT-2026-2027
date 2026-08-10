import React, { useState } from 'react';
import { LogIn, X, Lock, Mail, AlertCircle, CheckCircle2 } from 'lucide-react';
import { loginAuthLoginPost } from '../../api';

export default function LoginModal({ isOpen, onClose, onLoginSuccess }) {
  const [email, setEmail] = useState('admin@heart.com');
  const [password, setPassword] = useState('change-me-immediately');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMsg('');
    setSuccessMsg('');

    try {
      const response = await loginAuthLoginPost({
        body: {
          username: email,
          password: password,
        },
      });

      if (response && response.data && response.data.access_token) {
        const token = response.data.access_token;
        localStorage.setItem('access_token', token);
        localStorage.setItem('user_email', email);
        setSuccessMsg('Successfully authenticated with MedShield backend!');
        setTimeout(() => {
          if (onLoginSuccess) onLoginSuccess(email, token);
          onClose();
        }, 800);
      } else {
        setErrorMsg('Authentication failed: Invalid credentials received.');
      }
    } catch (err) {
      console.error('Login error:', err);
      setErrorMsg('Login failed. Please check credentials or verify FastAPI backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-md glass-panel border border-slate-800 p-6 rounded-2xl shadow-2xl space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
              <Lock className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-100">Backend Authentication</h3>
              <p className="text-xs text-slate-400">Sign in to acquire JWT token from FastAPI</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-slate-800/50 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Error / Success Notifications */}
        {errorMsg && (
          <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{errorMsg}</span>
          </div>
        )}

        {successMsg && (
          <div className="flex items-center gap-2 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Account Email
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="admin@heart.com"
                className="w-full pl-10 pr-3 py-2 bg-slate-900/60 border border-slate-800 rounded-xl text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full pl-10 pr-3 py-2 bg-slate-900/60 border border-slate-800 rounded-xl text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 transition-colors"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-gradient-to-r from-indigo-500 via-indigo-600 to-emerald-600 hover:from-indigo-600 hover:to-emerald-700 text-white font-bold text-xs rounded-xl shadow-lg shadow-indigo-500/20 transition-all duration-200 disabled:opacity-50"
          >
            {isLoading ? (
              <span>Authenticating...</span>
            ) : (
              <>
                <LogIn className="w-4 h-4" />
                <span>Sign In & Connect API</span>
              </>
            )}
          </button>
        </form>

        <div className="p-3 bg-slate-900/40 border border-slate-800/60 rounded-xl text-[11px] text-slate-400 space-y-1">
          <p className="font-semibold text-slate-300">Default Super Admin Credentials:</p>
          <p>• Email: <code className="text-indigo-300">admin@heart.com</code></p>
          <p>• Password: <code className="text-indigo-300">change-me-immediately</code></p>
        </div>
      </div>
    </div>
  );
}
