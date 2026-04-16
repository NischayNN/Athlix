import { useNavigate } from "react-router-dom";

function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6 md:p-12">
      <div className="max-w-6xl mx-auto">
        <header className="flex justify-between items-center border-b border-slate-800 pb-6 mb-10">
          <div>
            <h1 className="text-3xl font-bold text-white tracking-tight">Movement <span className="text-cyan-400">Hub</span></h1>
            <p className="mt-2 text-slate-400">Select a movement to analyze or view recent sessions.</p>
          </div>
          <button 
            onClick={() => navigate('/upload')}
            className="hidden md:block px-6 py-3 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl transition shadow-[0_0_15px_rgba(34,211,238,0.4)]"
          >
            Upload New Video
          </button>
        </header>

        <main>
          <div className="flex justify-between items-end mb-6">
            <h2 className="text-xl font-semibold text-slate-200">Available Modules</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Squat Card */}
            <div 
              className="group relative bg-slate-900 border border-slate-700 rounded-2xl p-8 hover:border-cyan-500 transition-colors cursor-pointer shadow-lg hover:shadow-cyan-500/20" 
              onClick={() => navigate('/upload')}
            >
              <div className="absolute top-6 right-6 px-3 py-1 bg-green-500/20 text-green-400 text-xs font-bold rounded-full uppercase tracking-wider">
                Active
              </div>
              <div className="h-16 w-16 bg-slate-800 text-cyan-400 rounded-2xl flex items-center justify-center mb-6 border border-slate-700 group-hover:bg-cyan-500/10 transition">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-cyan-400 transition">Squat Analysis</h3>
              <p className="text-slate-400 mb-6 line-clamp-2">
                Full-body biomechanical analysis. Detect stance width, depth, forward lean, and track fatigue decay across reps.
              </p>
              <div className="flex items-center text-cyan-400 font-semibold group-hover:translate-x-2 transition-transform">
                Start Analysis <span className="ml-2">→</span>
              </div>
            </div>

            {/* Bowling Card */}
            <div className="relative bg-slate-900/50 border border-slate-800 rounded-2xl p-8 opacity-80 cursor-not-allowed shadow-inner">
              <div className="absolute top-6 right-6 px-3 py-1 bg-amber-500/20 text-amber-400 text-xs font-bold rounded-full uppercase tracking-wider">
                Beta / Demo
              </div>
              <div className="h-16 w-16 bg-slate-800/50 text-slate-500 rounded-2xl flex items-center justify-center mb-6 border border-slate-800">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"></path>
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-slate-300 mb-3">Cricket Bowling</h3>
              <p className="text-slate-500 mb-6">
                Pace and spin biomechanics. Track front foot contact, release angle, and body rotation. Limited feature preview.
              </p>
              <div className="flex items-center text-slate-500 font-medium">
                Coming Soon <span className="ml-2">⏳</span>
              </div>
            </div>
          </div>
          
          <div className="mt-12 md:hidden">
             <button 
                onClick={() => navigate('/upload')}
                className="w-full px-6 py-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl transition shadow-[0_0_15px_rgba(34,211,238,0.4)]"
              >
                Upload New Video
              </button>
          </div>
        </main>
      </div>
    </div>
  );
}

export default Dashboard;
