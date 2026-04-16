import { useNavigate } from "react-router-dom";

function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6 md:p-12">
      <div className="max-w-7xl mx-auto">
        
        {/* Welcome Header */}
        <header className="flex flex-col md:flex-row md:justify-between md:items-end border-b border-slate-800 pb-8 mb-10 gap-6">
          <div>
            <h1 className="text-4xl font-black text-white tracking-tight mb-2">Welcome to your <span className="text-cyan-400">Workspace</span></h1>
            <p className="text-lg text-slate-400">Select a movement to analyze or view recent feedback. Upload a video to get started.</p>
          </div>
          <button 
            onClick={() => navigate('/upload')}
            className="flex items-center gap-2 px-6 py-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl transition shadow-[0_0_20px_rgba(34,211,238,0.3)] hover:shadow-[0_0_30px_rgba(34,211,238,0.5)] transform hover:-translate-y-1 w-full md:w-auto justify-center"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Upload New Video
          </button>
        </header>

        <main className="grid lg:grid-cols-3 gap-10">
          
          {/* Main Content Area (Movements) */}
          <div className="lg:col-span-2 space-y-8">
            <div>
              <h2 className="text-2xl font-bold text-slate-200 mb-6 flex items-center">
                <div className="w-2 h-6 bg-cyan-500 rounded-sm mr-3"></div>
                Active Analysis Modules
              </h2>
              
              <div className="grid sm:grid-cols-2 gap-6">
                {/* Squat Card */}
                <div 
                  className="group relative bg-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500 transition-all cursor-pointer shadow-lg hover:shadow-cyan-500/20" 
                  onClick={() => navigate('/upload')}
                >
                  <div className="absolute top-4 right-4 flex items-center">
                    <span className="flex h-2 w-2 rounded-full bg-green-400 animate-pulse mr-2"></span>
                    <span className="text-green-400 text-xs font-bold uppercase tracking-wider">Available</span>
                  </div>
                  <div className="h-14 w-14 bg-slate-950 text-cyan-400 rounded-xl flex items-center justify-center mb-5 border border-slate-800 group-hover:bg-cyan-500/10 transition shadow-inner">
                    <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-400 transition">Squat Analysis</h3>
                  <p className="text-slate-400 mb-5 text-sm leading-relaxed">
                    Full-body biomechanical tracking. Detect depth, forward lean, and form decay.
                  </p>
                  <div className="flex items-center text-cyan-400 font-semibold group-hover:translate-x-2 transition-transform text-sm">
                    Start Analysis <span className="ml-2">→</span>
                  </div>
                </div>

                {/* Bowling Card */}
                <div className="relative bg-slate-900/40 border border-slate-800 border-dashed rounded-2xl p-6 cursor-default">
                  <div className="absolute top-4 right-4 flex items-center">
                    <span className="text-amber-500/80 text-xs font-bold uppercase tracking-wider bg-amber-500/10 px-2 py-1 rounded">Beta</span>
                  </div>
                  <div className="h-14 w-14 bg-slate-950/50 text-slate-600 rounded-xl flex items-center justify-center mb-5 border border-slate-800">
                    <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"></path>
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-slate-400 mb-2">Cricket Bowling</h3>
                  <p className="text-slate-500 mb-5 text-sm leading-relaxed">
                    Pace and spin kinematics. Run-up momentum, release angle, and front-foot contact.
                  </p>
                  <div className="flex items-center text-slate-600 font-medium text-sm">
                    Coming Soon <span className="ml-2">⏳</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Short Info Banner */}
            <div className="bg-gradient-to-r from-slate-900 text-slate-300 p-6 rounded-2xl border border-slate-800 text-sm leading-relaxed relative overflow-hidden hidden sm:block">
              <div className="absolute right-0 top-0 w-32 h-32 bg-cyan-500/5 rounded-full blur-2xl"></div>
              Athlix processes video at 60 frames per second using proprietary vision models. Keep your camera stable, ensure the full body is visible, and shoot in good lighting for the most accurate kinematic data.
            </div>

          </div>

          {/* Right Sidebar Area (Features Summary) */}
          <div className="space-y-6">
            <h2 className="text-xl font-bold text-slate-200 mb-2 flex items-center border-b border-slate-800 pb-4">
              Intelligence Features
            </h2>
            
            <div className="space-y-4">
              {/* Feature 1 */}
              <div className="flex items-start bg-slate-900 border border-slate-800 p-4 rounded-xl">
                <div className="mt-1 bg-slate-950 p-2 rounded-lg border border-slate-800 mr-4">
                  <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold text-sm">Pose Flaw Detection</h4>
                  <p className="text-slate-500 text-xs mt-1">Automatic detection of severe posture mistakes.</p>
                </div>
              </div>

              {/* Feature 2 */}
              <div className="flex items-start bg-slate-900 border border-slate-800 p-4 rounded-xl">
                <div className="mt-1 bg-slate-950 p-2 rounded-lg border border-slate-800 mr-4">
                  <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold text-sm">Form Decay Tracking</h4>
                  <p className="text-slate-500 text-xs mt-1">Monitor form degradation across all repetitions.</p>
                </div>
              </div>

              {/* Feature 3 */}
              <div className="flex items-start bg-slate-900 border border-slate-800 p-4 rounded-xl">
                <div className="mt-1 bg-slate-950 p-2 rounded-lg border border-slate-800 mr-4">
                  <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold text-sm">Explainable Injury Risk</h4>
                  <p className="text-slate-500 text-xs mt-1">Correlate kinematic data with potential strain injuries.</p>
                </div>
              </div>

              {/* Feature 4 */}
              <div className="flex items-start bg-slate-900 border border-slate-800 p-4 rounded-xl">
                <div className="mt-1 bg-slate-950 p-2 rounded-lg border border-slate-800 mr-4">
                   <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>
                </div>
                <div>
                  <h4 className="text-white font-semibold text-sm">Coaching Feedback</h4>
                  <p className="text-slate-500 text-xs mt-1">Get precise cues instantly to fix your form.</p>
                </div>
              </div>
            </div>
            
          </div>
        </main>
      </div>
    </div>
  );
}

export default Dashboard;
