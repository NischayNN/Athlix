import { useNavigate } from "react-router-dom";
import { useState, useEffect, useRef } from "react";
import { api } from "../services/api";
import FormCorrectionPreview from "../components/FormCorrectionPreview";

// ─── Utility: Animated number counter ────────────────────────────
function useCounter(target, duration = 1200) {
  const [count, setCount] = useState(0);
  const started = useRef(false);

  const start = () => {
    if (started.current) return;
    started.current = true;
    let current = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      current += step;
      if (current >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, 16);
  };

  return [count, start];
}

// ─── Utility: Fade-in on scroll (Smooth Apple-style) ─────────────
function FadeSection({ children, delay = 0, className = "" }) {
  const ref = useRef(null);
  const [vis, setVis] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVis(true); obs.disconnect(); } },
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`transition-all duration-[800ms] ease-[cubic-bezier(0.16,1,0.3,1)] ${vis ? "opacity-100 translate-y-0 scale-100" : "opacity-0 translate-y-6 scale-[0.98]"
        } ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

// ─── Activity Ring (Apple Fitness Style) ─────────────────────────
function ActivityRing({ score }) {
  const size = 240;
  const strokeWidth = 22;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const [offset, setOffset] = useState(circumference);
  const [displayScore, scoreStart] = useCounter(score);
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          // Add a slight delay for the smooth "pop" effect
          setTimeout(() => setOffset(circumference - (score / 100) * circumference), 150);
          scoreStart();
          obs.disconnect();
        }
      },
      { threshold: 0.3 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [score, circumference, scoreStart]);

  return (
    <div ref={ref} className="relative flex items-center justify-center mx-auto" style={{ width: size, height: size }}>
      <svg className="w-full h-full -rotate-90 drop-shadow-2xl" viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id="ringGradient" x1="100%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#ff1493" /> {/* Neon Pink */}
            <stop offset="50%" stopColor="#ff4500" /> {/* Neon Orange */}
            <stop offset="100%" stopColor="#ffea00" /> {/* Neon Yellow */}
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {/* Background Track */}
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#2c2c2e" strokeWidth={strokeWidth} />
        {/* Main Gradient Ring */}
        <circle
          cx={size / 2} cy={size / 2} r={radius} fill="none"
          stroke="url(#ringGradient)" strokeWidth={strokeWidth} strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-[2000ms] ease-[cubic-bezier(0.16,1,0.3,1)]"
        />
      </svg>
      {/* Center content */}
      <div className="absolute flex flex-col items-center justify-center">
        <span className="text-6xl font-bold tracking-tight text-white">{displayScore}</span>
        <span className="text-sm font-semibold text-zinc-400 mt-1 uppercase tracking-wider">Score</span>
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════
// APPLE FITNESS STYLE RESULTS PAGE
// ═════════════════════════════════════════════════════════════════
function Results() {
  const navigate = useNavigate();
  const [mockData, setMockData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const data = await api.getAnalysisResults();
      setMockData(data);
    };
    fetchData();
  }, []);

  if (!mockData) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center font-sans">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-[#2c2c2e] rounded-full border-t-[#32ade6] animate-spin" />
        </div>
      </div>
    );
  }

  const maxScore = Math.max(...mockData.decayData.map(d => d.score));
  const dropIdx = mockData.decayData.findIndex((d, i) => i > 0 && mockData.decayData[i - 1].score - d.score > 10);
  const dropRep = dropIdx >= 0 ? mockData.decayData[dropIdx].rep : 5;

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-[#32ade6] selection:text-white pb-24">

      {/* ─── Top Navigation ────────────────────────────────────── */}
      <div className="sticky top-0 z-50 bg-black/70 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <button
            onClick={() => navigate('/dashboard')}
            className="text-[#32ade6] font-semibold flex items-center gap-1 hover:opacity-80 transition"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Summary
          </button>
          <div className="font-semibold text-base tracking-tight">Analysis</div>
          <button
            onClick={() => navigate('/upload')}
            className="text-[#32ade6] font-semibold hover:opacity-80 transition"
          >
            New
          </button>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 pt-8 space-y-6">

        {/* ─── Hero Overview Card ──────────────────────────────── */}
        <FadeSection>
          <div className="bg-[#1c1c1e] rounded-[32px] p-8 md:p-10 flex flex-col items-center relative overflow-hidden">
            <h2 className="text-3xl font-bold tracking-tight mb-2">Workout Complete</h2>
            <p className="text-zinc-400 font-medium mb-12">{mockData.movement} • {mockData.decayData.length} Reps</p>

            <ActivityRing score={mockData.score} />

            <div className="mt-12 w-full grid grid-cols-2 gap-4 text-center">
              <div className="bg-black/40 rounded-2xl p-4">
                <div className="text-zinc-400 text-xs font-semibold uppercase tracking-wider mb-1">Reps Logged</div>
                <div className="text-xl font-bold text-[#ff9f0a]">{mockData.decayData.length} Total</div>
              </div>
              <div className="bg-black/40 rounded-2xl p-4">
                <div className="text-zinc-400 text-xs font-semibold uppercase tracking-wider mb-1">Velocity Profile</div>
                <div className="text-xl font-bold text-[#32ade6]">{mockData.velocityClassification || 'Moderate'}</div>
              </div>
            </div>
          </div>
        </FadeSection>

        {/* ─── Movement Risk Index Card (NEW) ──────────────────── */}
        <FadeSection delay={50}>
          <div className="bg-gradient-to-br from-[#1c1c1e] to-[#0a0a0c] rounded-[32px] p-8 md:p-10 border border-white/10 relative overflow-hidden shadow-2xl">
            <div className="absolute top-0 right-0 p-8 opacity-20 pointer-events-none text-9xl font-black text-white">
              {mockData.movementRiskIndex}
            </div>
            
            <h3 className="text-2xl font-bold tracking-tight mb-2 relative z-10">Movement Risk Index</h3>
            <p className="text-zinc-400 text-sm font-medium leading-relaxed mb-8 max-w-sm relative z-10">
              Context-aware synthesis of mechanical deviations, session intensity, recovery state, and injury history.
            </p>

            <div className="flex flex-col md:flex-row items-center gap-8 relative z-10">
              <div className="flex flex-col items-center justify-center bg-black/60 rounded-[32px] w-48 h-48 border border-white/5 shadow-inner">
                <span className={`text-[10px] font-bold uppercase tracking-widest px-3 py-1 mb-2 rounded-full ${
                  mockData.riskLabel === 'High' ? 'text-[#ff3b30] bg-[#ff3b30]/10' :
                  mockData.riskLabel === 'Moderate' ? 'text-[#ff9f0a] bg-[#ff9f0a]/10' :
                  'text-[#34c759] bg-[#34c759]/10'
                }`}>
                  {mockData.riskLabel} RISK
                </span>
                <span className="text-7xl font-black text-white tracking-tighter">{mockData.movementRiskIndex}</span>
                <span className="text-xs text-zinc-500 uppercase font-bold tracking-widest mt-1">/ 100</span>
              </div>
              
              <div className="flex-1 bg-black/40 rounded-[24px] p-6 border border-white/5 w-full">
                <p className="text-sm text-zinc-300 font-medium leading-relaxed">
                  <span className="text-white font-bold block mb-2 break-words">Insight:</span>
                  {mockData.explanationInsight}
                </p>
              </div>
            </div>

            {/* Explainability Breakdown */}
            <div className="mt-8 pt-6 border-t border-white/5 relative z-10">
              <h4 className="text-xs uppercase font-bold text-zinc-500 tracking-widest mb-4">Risk Contribution Breakdown</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {mockData.riskBreakdown?.slice(0, 4).map((rb, idx) => (
                  <div key={idx} className="bg-black/30 rounded-xl p-4">
                    <div className="text-[10px] text-zinc-400 font-bold uppercase truncate mb-1" title={rb.issue}>{rb.issue}</div>
                    <div className="text-lg font-bold text-white mb-2">+{rb.contribution}</div>
                    <div className="flex flex-col gap-1 text-[9px] text-zinc-600 font-mono">
                      <span>Sev: {rb.flawSeverity}</span>
                      <span>Int: x{rb.intensityMult}</span>
                      <span>Rec: x{rb.recoveryMult}</span>
                      <span>His: x{rb.historyMult}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </FadeSection>

        {/* ─── Form Quality Over Time (Apple Health Chart) ─────── */}
        <FadeSection delay={100}>
          <div className="bg-[#1c1c1e] rounded-[32px] p-8 relative overflow-hidden">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h3 className="text-xl font-bold tracking-tight">Form Decay</h3>
                <p className="text-zinc-400 text-sm font-medium mt-1">Rep-by-rep kinematic score</p>
              </div>
              <div className="h-10 w-10 bg-[#32ade6]/20 rounded-full flex items-center justify-center">
                <svg className="w-5 h-5 text-[#32ade6]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
              </div>
            </div>

            <div className="h-[220px] flex items-end justify-between gap-2 relative mt-4">
              {/* Threshold Line */}
              <div
                className="absolute left-0 right-0 border-t-2 border-dashed border-[#ff3b30]/40 z-0"
                style={{ bottom: `${(80 / maxScore) * 100}%` }}
              >
                <span className="absolute -top-6 right-0 text-xs font-bold text-[#ff3b30] bg-[#1c1c1e] pl-2">
                  Failure Threshold (80)
                </span>
              </div>

              {mockData.decayData.map((data, idx) => {
                const pct = (data.score / maxScore) * 100;
                const failing = data.score < 80;
                const isBreakdown = data.rep === dropRep;

                return (
                  <div key={idx} className="relative flex flex-col items-center w-full h-full justify-end z-10 group">
                    {isBreakdown && (
                      <div className="absolute -top-8 text-[#ff3b30] text-[10px] uppercase font-bold tracking-wider bg-[#ff3b30]/10 px-2 py-1 rounded-md animate-pulse whitespace-nowrap">
                        Breakdown
                      </div>
                    )}
                    {/* The rounded bar */}
                    <div
                      className={`w-full max-w-[40px] rounded-full transition-all duration-[1200ms] ease-out ${failing ? 'bg-[#ff3b30]' : 'bg-[#32ade6]'
                        } group-hover:opacity-80 cursor-default`}
                      style={{ height: `${pct}%` }}
                    />
                    <div className="mt-3 text-sm font-semibold text-zinc-400">
                      {data.rep}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-6 p-4 rounded-2xl bg-black/30">
              <p className="text-sm text-zinc-300 font-medium leading-relaxed">
                <span className="text-white font-bold">Insight: </span>
                Kinematic integrity dropped sharply at Rep {dropRep}. To induce optimal adaptation without injury risk, recommend capping this load at {dropRep - 1} reps.
              </p>
            </div>
          </div>
        </FadeSection>

        {/* ─── Personalized Intensity Load (Derived) ───────────── */}
        <FadeSection delay={150}>
          <div className="bg-[#1c1c1e] rounded-[32px] p-8 flex flex-col gap-6 relative overflow-hidden border border-white/5">
            <div className="absolute top-0 left-0 w-2 h-full bg-gradient-to-b from-[#32ade6] to-[#af52ff]" />
            
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <h3 className="text-xl font-bold tracking-tight mb-1">Intensity Profile</h3>
                <p className="text-zinc-400 text-sm font-medium">Kinematic load multiplied by estimated velocity.</p>
              </div>
              <div className="px-4 py-2 bg-black/40 rounded-full border border-white/5 text-xs font-bold text-zinc-300">
                <span className="opacity-50 mr-2">Est. Velocity</span> {mockData.movementVelocity} m/s
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-black/40 rounded-2xl p-5 border border-white/5">
                <div className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-1">Relative Load</div>
                <div className="text-3xl font-black text-white flex items-baseline gap-1">
                  {mockData.relativePct} <span className="text-sm text-zinc-500 uppercase">%</span>
                </div>
                <div className="text-xs text-zinc-400 mt-1 font-medium">{mockData.weightUsed}kg of {mockData.maxPR}kg Max</div>
              </div>

              <div className="bg-black/40 rounded-2xl p-5 border border-white/5">
                <div className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-1">Movement Class</div>
                <div className={`text-2xl font-bold ${
                  mockData.velocityClassification === 'Strength' ? 'text-[#ff3b30]' :
                  mockData.velocityClassification === 'Power' ? 'text-[#ffea00]' : 'text-[#32ade6]'
                }`}>
                  {mockData.velocityClassification}
                </div>
                <div className="text-xs text-zinc-400 mt-1 font-medium">Adaptation Focus</div>
              </div>

              <div className="bg-black/40 rounded-2xl p-5 border border-white/5 relative overflow-hidden">
                <div className="absolute -bottom-4 right-0 opacity-10">
                  <svg className="w-24 h-24" fill="currentColor" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                </div>
                <div className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-1">Load Score</div>
                <div className="text-4xl font-black text-white">{mockData.loadScore}</div>
                <div className="text-xs text-zinc-400 mt-1 font-medium">Cumulative Stress</div>
              </div>
            </div>
            
          </div>
        </FadeSection>

        {/* ─── Highlights / Structural Flaws (List layout) ───────── */}
        <FadeSection delay={200}>
          <div className="bg-[#1c1c1e] rounded-[32px] p-8">
            <h3 className="text-xl font-bold tracking-tight mb-6">Highlights</h3>
            <div className="space-y-0">
              {mockData.keyIssues.map((issue, idx) => {
                const isLast = idx === mockData.keyIssues.length - 1;
                const isHigh = issue.severity === 'High';
                const isMed = issue.severity === 'Medium';
                const iconColor = isHigh ? 'text-[#ff3b30] bg-[#ff3b30]/20' : isMed ? 'text-[#ff9f0a] bg-[#ff9f0a]/20' : 'text-[#32ade6] bg-[#32ade6]/20';

                return (
                  <div key={issue.id} className={`flex items-start gap-4 py-5 ${!isLast ? 'border-b border-white/5' : ''}`}>
                    <div className={`mt-1 w-10 h-10 rounded-full flex items-center justify-center shrink-0 ${iconColor}`}>
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="font-bold text-lg">{issue.issue}</h4>
                        <span className={`text-xs font-bold uppercase tracking-wider px-2.5 py-1 rounded-full ${isHigh ? 'text-[#ff3b30] bg-[#ff3b30]/10' :
                          isMed ? 'text-[#ff9f0a] bg-[#ff9f0a]/10' :
                            'text-[#32ade6] bg-[#32ade6]/10'
                          }`}>
                          {issue.severity}
                        </span>
                      </div>
                      <p className="text-zinc-400 text-sm font-medium leading-relaxed">
                        {issue.detail}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </FadeSection>

        {/* ─── Visual Proof / Frame Comparison ──────────────────── */}
        <FadeSection delay={300}>
          <div className="grid sm:grid-cols-2 gap-6">
            <div className="bg-[#1c1c1e] rounded-[32px] p-6 flex flex-col h-full">
              <h3 className="text-lg font-bold tracking-tight mb-1">Best Execution</h3>
              <p className="text-sm font-medium text-zinc-400 tracking-tight mb-4">Rep 01</p>

              <div className="w-full aspect-square bg-black rounded-2xl relative overflow-hidden flex items-center justify-center border border-white/5 shadow-inner">
                {/* Glowing Skeleton */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1/3 h-1/2">
                  <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-[#32ade6] rounded-full shadow-[0_0_15px_#32ade6] z-10" />
                  <div className="absolute top-[40%] right-[10%] w-2.5 h-2.5 bg-[#32ade6] rounded-full shadow-[0_0_10px_#32ade6] z-10" />
                  <div className="absolute bottom-[10%] left-[20%] w-2.5 h-2.5 bg-[#32ade6] rounded-full shadow-[0_0_10px_#32ade6] z-10" />
                  <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <line x1="50" y1="5" x2="85" y2="45" stroke="#32ade6" strokeWidth="2.5" opacity="0.6" strokeLinecap="round" />
                    <line x1="85" y1="45" x2="25" y2="90" stroke="#32ade6" strokeWidth="2.5" opacity="0.6" strokeLinecap="round" />
                  </svg>
                </div>
                <div className="absolute bottom-3 right-3 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full">
                  <span className="text-xs font-bold text-white">Score: 92</span>
                </div>
              </div>
            </div>

            <div className="bg-[#1c1c1e] rounded-[32px] p-6 flex flex-col h-full">
              <h3 className="text-lg font-bold tracking-tight mb-1">Structural Failure</h3>
              <p className="text-sm font-medium text-zinc-400 tracking-tight mb-4">Rep {dropRep}</p>

              <div className="w-full aspect-square bg-black rounded-2xl relative overflow-hidden flex items-center justify-center border border-[#ff3b30]/20 shadow-inner">
                {/* Red Distorted Skeleton */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1/3 h-1/2">
                  <div className="absolute top-[10%] left-[20%] w-3 h-3 bg-[#ff3b30] rounded-full shadow-[0_0_20px_#ff3b30] z-10" />
                  <div className="absolute top-[45%] right-[5%] w-2.5 h-2.5 bg-[#ff3b30] rounded-full shadow-[0_0_15px_#ff3b30] z-10" />
                  <div className="absolute bottom-[5%] left-[30%] w-2.5 h-2.5 bg-white rounded-full z-10" />
                  <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <line x1="25" y1="12" x2="90" y2="48" stroke="#ff3b30" strokeWidth="3" opacity="0.8" strokeLinecap="round" />
                    <line x1="90" y1="48" x2="35" y2="95" stroke="#ff3b30" strokeWidth="3" opacity="0.8" strokeLinecap="round" />
                  </svg>
                  <div className="absolute top-[35%] -right-12 bg-[#ff3b30] text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-lg">
                    DEV 12°
                  </div>
                </div>
                <div className="absolute bottom-3 right-3 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-[#ff3b30]/30">
                  <span className="text-xs font-bold text-[#ff3b30]">Score: 72</span>
                </div>
              </div>
            </div>
          </div>
        </FadeSection>

        {/* ─── Form Correction Preview ─────────────────────────── */}
        <FadeSection delay={350}>
          <FormCorrectionPreview
            formFlags={{
              knee_valgus: mockData.keyIssues?.some(i => i.issue === 'Knee Valgus'),
              incomplete_depth: mockData.keyIssues?.some(i => i.issue === 'Incomplete Depth'),
              excessive_forward_lean: mockData.keyIssues?.some(i => i.issue === 'Excessive Forward Lean'),
            }}
          />
        </FadeSection>

        {/* ─── Coaching Recommendations (App Store Style Cards) ──── */}
        <FadeSection delay={400}>
          <div>
            <h3 className="text-2xl font-bold tracking-tight mb-6 px-2">Coaching Plan</h3>
            <div className="grid sm:grid-cols-2 gap-4">
              {mockData.coachingTips.map((tip, idx) => {
                // Different bright gradients for the cards
                const gradients = [
                  "from-[#ff2a6d] to-[#ff7e5f]", // Pink to Peach
                  "from-[#0575E6] to-[#00F260]", // Deep Blue to Green
                  "from-[#8E2DE2] to-[#4A00E0]", // Purples
                ];
                const g = gradients[idx % gradients.length];

                return (
                  <div key={tip.id} className="relative rounded-[32px] overflow-hidden bg-[#1c1c1e] p-6 h-full flex flex-col hover:scale-[1.02] transition-transform duration-300 shadow-xl cursor-pointer">
                    <div className={`absolute top-0 left-0 w-full h-2 bg-gradient-to-r ${g}`} />

                    <div className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 mb-2">Priority {idx + 1}</div>
                    <h4 className="text-xl font-bold text-white mb-2 leading-tight">{tip.action}</h4>
                    <p className="text-sm text-zinc-400 font-medium mb-6 flex-1 leading-relaxed">{tip.cue}</p>

                    <div className="flex items-center justify-between mt-auto pt-4 border-t border-white/5">
                      <span className="text-xs font-bold text-zinc-300 bg-white/10 px-3 py-1.5 rounded-lg">
                        {tip.target}
                      </span>
                      <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center text-white">
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </FadeSection>

      </div>
    </div>
  );
}

export default Results;
