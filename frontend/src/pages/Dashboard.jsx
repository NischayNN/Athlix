import { useNavigate } from "react-router-dom";

function Dashboard() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-black text-white font-sans selection:bg-zinc-700 flex flex-col md:flex-row">
      {/* Sidebar Navigation */}
      <aside className="w-full md:w-72 border-b md:border-b-0 md:border-r border-zinc-900 p-8 flex flex-col justify-between">
        <div>
          <div
            className="flex items-center gap-3 mb-16 cursor-pointer"
            onClick={() => navigate('/')}
          >
            <div className="w-6 h-6 bg-white text-black flex items-center justify-center rounded-sm">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
            </div>
            <span className="font-bold text-xl tracking-tight uppercase">Athlix</span>
          </div>

          <nav className="space-y-6">
            <div>
              <p className="text-xs font-bold tracking-[0.2em] text-zinc-600 uppercase mb-6">Workspace</p>
              <ul className="space-y-4 font-medium text-sm tracking-wide">
                <li className="text-white flex items-center gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-white"></span> Overview
                </li>
                <li className="text-zinc-600 hover:text-zinc-400 cursor-not-allowed transition flex items-center gap-3">
                  <span className="w-1.5 h-1.5 rounded-full bg-transparent"></span> History
                </li>
                <li
                  onClick={() => navigate('/profile')}
                  className="text-white hover:text-zinc-300 cursor-pointer transition flex items-center gap-3"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-zinc-600"></span> Athlete Profile
                </li>
              </ul>
            </div>
          </nav>
        </div>

        <div className="mt-12 hidden md:block border-t border-zinc-900 pt-8">
          <p className="text-xs font-bold tracking-[0.2em] uppercase text-zinc-600">System Secure</p>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 p-8 md:p-16">
        <header className="flex flex-col md:flex-row md:justify-between md:items-end border-b border-zinc-900 pb-10 mb-12 gap-6">
          <div>
            <h1 className="text-5xl font-black tracking-tighter uppercase mb-4">Dashboard</h1>
            <p className="text-lg text-zinc-500 font-light tracking-wide">Select a movement protocol to begin precision tracking.</p>
          </div>
          <button
            onClick={() => navigate('/upload')}
            className="px-10 py-5 bg-white text-black font-bold uppercase tracking-[0.2em] text-xs hover:bg-zinc-200 transition shadow-2xl"
          >
            New Analysis
          </button>
        </header>

        <section className="grid lg:grid-cols-3 gap-16">
          {/* Active Modules Map */}
          <div className="lg:col-span-2 space-y-12">
            <div>
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-xs font-bold tracking-[0.2em] text-white uppercase">Movement Architecture</h2>
              </div>

              <div className="grid sm:grid-cols-2 gap-6">
                {/* Protocol: Squat */}
                <div
                  className="group relative bg-[#111] border border-zinc-800 p-8 hover:border-zinc-500 transition-colors cursor-pointer flex flex-col justify-between min-h-[320px]"
                  onClick={() => navigate('/upload')}
                >
                  <div className="flex justify-between items-start mb-6">
                    <span className="text-[10px] font-bold tracking-widest border border-white/20 px-3 py-1 uppercase text-zinc-400">Protocol 01</span>
                    <span className="flex items-center text-white text-[10px] font-bold uppercase tracking-wider">
                      <span className="flex h-1.5 w-1.5 rounded-full bg-white mr-2 animate-pulse"></span>
                      Active
                    </span>
                  </div>

                  <div>
                    <h3 className="text-3xl font-black text-white uppercase tracking-tighter mb-4 group-hover:text-zinc-300 transition line-clamp-2">Gym:<br />Squat</h3>
                    <p className="text-zinc-500 text-sm font-light leading-relaxed mb-8">
                      Full-body biomechanical tracking. Detecting depth, forward lean, and varus collapse under load.
                    </p>
                    <div className="flex items-center text-white font-bold text-xs uppercase tracking-[0.15em] group-hover:translate-x-2 transition-transform">
                      Initiate <span className="ml-2">→</span>
                    </div>
                  </div>
                </div>

                {/* Protocol: Bowling */}
                <div className="relative bg-[#050505] border border-zinc-900 p-8 flex flex-col justify-between min-h-[320px] overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-transparent to-white/[0.02]"></div>
                  <div className="relative z-10 flex justify-between items-start mb-6">
                    <span className="text-[10px] font-bold tracking-widest border border-zinc-800 px-3 py-1 uppercase text-zinc-700">Protocol 02</span>
                    <span className="text-zinc-700 text-[10px] font-bold uppercase tracking-wider">
                      Locked
                    </span>
                  </div>

                  <div className="relative z-10">
                    <h3 className="text-3xl font-black text-zinc-700 uppercase tracking-tighter mb-4 line-clamp-2">Sports:<br />Fast Bowler</h3>
                    <p className="text-zinc-700 text-sm font-light leading-relaxed mb-8">
                      Pace and spin kinematics. Run-up momentum, release angle, and front-foot contact forces.
                    </p>
                    <div className="flex items-center text-zinc-700 font-bold text-xs uppercase tracking-[0.15em]">
                      Demo Preview
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Upload Banner */}
            <div
              onClick={() => navigate('/upload')}
              className="bg-black border border-zinc-800 p-8 flex flex-col sm:flex-row items-center justify-between cursor-pointer hover:border-zinc-500 transition"
            >
              <div>
                <h3 className="text-white font-bold uppercase tracking-[0.15em] text-sm mb-2">Direct Upload</h3>
                <p className="text-zinc-500 text-xs font-light tracking-wide">Bypass protocol selection. Parse generic movement data instantly.</p>
              </div>
              <div className="mt-6 sm:mt-0 px-8 py-4 bg-[#111] border border-zinc-700 text-white font-bold uppercase text-[10px] tracking-widest hover:bg-white hover:text-black transition">
                Select File
              </div>
            </div>

          </div>

          {/* Side Capabilities Panel */}
          <div className="space-y-12">
            <div>
              <h2 className="text-xs font-bold tracking-[0.2em] text-white uppercase mb-8 border-b border-zinc-900 pb-4">
                Engine Status
              </h2>

              <div className="space-y-10">

                <div className="flex items-start gap-5">
                  <div className="mt-1 font-mono text-zinc-700 text-sm">01</div>
                  <div>
                    <h4 className="text-white font-bold text-xs tracking-[0.15em] uppercase mb-2">Pose Tracking</h4>
                    <p className="text-zinc-500 text-sm font-light leading-relaxed">High-fidelity 3D extraction. Stable camera required for optimal joint accuracy.</p>
                  </div>
                </div>

                <div className="flex items-start gap-5">
                  <div className="mt-1 font-mono text-zinc-700 text-sm">02</div>
                  <div>
                    <h4 className="text-white font-bold text-xs tracking-[0.15em] uppercase mb-2">Decay Modeling</h4>
                    <p className="text-zinc-500 text-sm font-light leading-relaxed">Status active. Monitoring technique degradation against initial baseline stability.</p>
                  </div>
                </div>

                <div className="flex items-start gap-5">
                  <div className="mt-1 font-mono text-zinc-700 text-sm">03</div>
                  <div>
                    <h4 className="text-white font-bold text-xs tracking-[0.15em] uppercase mb-2">Risk Heuristics</h4>
                    <p className="text-zinc-500 text-sm font-light leading-relaxed">Correlating persistent angular deviations with statistical probability of strain.</p>
                  </div>
                </div>

              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Dashboard;
