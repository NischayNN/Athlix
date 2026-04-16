import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-950 text-white px-6 py-12">

      {/* Hero Section */}
      <div className="max-w-4xl mx-auto text-center">
        <h1 className="text-5xl font-bold text-cyan-400">
          Athlix
        </h1>

        <p className="mt-4 text-lg text-slate-300">
          AI-powered squat video analysis to detect form flaws, track fatigue, and prevent injuries.
        </p>

        <div className="mt-6 flex justify-center gap-4">
          <button
            onClick={() => navigate('/dashboard')}
            className="px-6 py-3 bg-cyan-500 hover:bg-cyan-400 text-black font-semibold rounded-xl transition"
          >
            Explore Dashboard
          </button>

          <button
            onClick={() => navigate('/upload')}
            className="px-6 py-3 border border-slate-600 hover:border-cyan-400 text-white rounded-xl transition"
          >
            Upload Video
          </button>
        </div>
      </div>

      {/* Problem Section */}
      <div className="mt-20 max-w-5xl mx-auto text-center">
        <h2 className="text-2xl font-semibold text-white">
          Why Athletes Get Injured
        </h2>
        <p className="mt-3 text-slate-400">
          Most injuries occur due to poor form and fatigue. Athletes often don’t realize their posture is degrading until it’s too late.
        </p>
      </div>

      {/* Features Section */}
      <div className="mt-16 grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">

        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-lg">
          <h3 className="text-xl font-semibold text-cyan-400">
            Pose Flaw Detection
          </h3>
          <p className="mt-2 text-slate-400">
            Identify posture mistakes like forward lean and improper depth.
          </p>
        </div>

        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-lg">
          <h3 className="text-xl font-semibold text-cyan-400">
            Form Decay Tracking
          </h3>
          <p className="mt-2 text-slate-400">
            Track how your form deteriorates across repetitions.
          </p>
        </div>

        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-lg">
          <h3 className="text-xl font-semibold text-cyan-400">
            Explainable Risk Insights
          </h3>
          <p className="mt-2 text-slate-400">
            Understand why your movement patterns increase injury risk.
          </p>
        </div>

      </div>

    </div>
  );
}

export default Home;