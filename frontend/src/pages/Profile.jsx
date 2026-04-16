import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

function Profile() {
  const navigate = useNavigate();

  const [profile, setProfile] = useState({
    age: '',
    weight: '',
    height: '',
    maxPR: '',
    experience: 'Intermediate',
    injuryHistory: '',
    primaryFocus: ''
  });

  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const savedProfile = localStorage.getItem('athlix_profile');
    if (savedProfile) {
      setProfile(JSON.parse(savedProfile));
    }
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setProfile(prev => ({ ...prev, [name]: value }));
    setSaved(false);
  };

  const handleSave = () => {
    localStorage.setItem('athlix_profile', JSON.stringify(profile));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="min-h-screen bg-black text-white font-sans p-6 md:p-16 selection:bg-zinc-700">
      <div className="max-w-3xl mx-auto">
        <header className="border-b border-zinc-900 pb-10 mb-12 flex justify-between items-end">
          <div>
            <h1 className="text-5xl font-black tracking-tighter uppercase mb-2">Athlete Profile</h1>
            <p className="text-zinc-500 font-light tracking-wide text-lg">Stable baseline biometrics for precision analysis.</p>
          </div>
          <button
            onClick={() => navigate('/upload')}
            className="text-[10px] uppercase font-bold tracking-widest text-zinc-500 hover:text-white transition"
          >
            Skip to Upload →
          </button>
        </header>

        <section className="space-y-12">
          {/* Biometrics Group */}
          <div>
            <h2 className="text-[10px] uppercase tracking-[0.2em] font-bold text-zinc-600 mb-6 flex items-center">
              <span className="text-white mr-4">01 //</span> Core Biometrics
            </h2>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <label className="block text-zinc-500 text-[10px] tracking-widest uppercase mb-4 font-bold">Age (Yrs)</label>
                <input
                  type="number" name="age" value={profile.age} onChange={handleChange} placeholder="e.g. 24"
                  className="w-full bg-[#050505] border border-zinc-900 focus:border-white transition-colors p-4 text-white font-mono text-sm outline-none"
                />
              </div>
              <div>
                <label className="block text-zinc-500 text-[10px] tracking-widest uppercase mb-4 font-bold">Height (cm)</label>
                <input
                  type="number" name="height" value={profile.height} onChange={handleChange} placeholder="e.g. 180"
                  className="w-full bg-[#050505] border border-zinc-900 focus:border-white transition-colors p-4 text-white font-mono text-sm outline-none"
                />
              </div>
              <div>
                <label className="block text-zinc-500 text-[10px] tracking-widest uppercase mb-4 font-bold">Weight (kg)</label>
                <input
                  type="number" name="weight" value={profile.weight} onChange={handleChange} placeholder="e.g. 78"
                  className="w-full bg-[#050505] border border-zinc-900 focus:border-white transition-colors p-4 text-white font-mono text-sm outline-none"
                />
              </div>
              <div>
                <label className="block text-zinc-500 text-[10px] tracking-widest uppercase mb-4 font-bold">Squat Max PR (kg)</label>
                <input
                  type="number" name="maxPR" value={profile.maxPR} onChange={handleChange} placeholder="e.g. 140"
                  className="w-full bg-[#050505] border border-zinc-900 focus:border-white transition-colors p-4 text-white font-mono text-sm outline-none"
                />
              </div>
            </div>
          </div>

          {/* Training Background */}
          <div>
            <h2 className="text-[10px] uppercase tracking-[0.2em] font-bold text-zinc-600 mb-6 flex items-center">
              <span className="text-white mr-4">02 //</span> Training Context
            </h2>
            <div className="grid sm:grid-cols-2 gap-6">
              <div>
                <label className="block text-zinc-500 text-[10px] tracking-widest uppercase mb-4 font-bold">Training Experience</label>
                <select
                  name="experience" value={profile.experience} onChange={handleChange}
                  className="w-full bg-[#050505] border border-zinc-900 focus:border-white transition-colors p-4 text-white font-mono text-sm outline-none appearance-none"
                >
                  <option value="Novice">Novice (&lt; 1 yr)</option>
                  <option value="Intermediate">Intermediate (1-3 yrs)</option>
                  <option value="Advanced">Advanced (3-5 yrs)</option>
                  <option value="Elite">Elite (5+ yrs)</option>
                </select>
              </div>
              <div>
                <label className="block text-zinc-500 text-[10px] tracking-widest uppercase mb-4 font-bold">Primary Sport/Focus</label>
                <input
                  type="text" name="primaryFocus" value={profile.primaryFocus} onChange={handleChange} placeholder="e.g. Powerlifting, Cricket"
                  className="w-full bg-[#050505] border border-zinc-900 focus:border-white transition-colors p-4 text-white font-mono text-sm outline-none"
                />
              </div>
              <div className="sm:col-span-2">
                <label className="block text-zinc-500 text-[10px] tracking-widest uppercase mb-4 font-bold">Previous Injury History</label>
                <textarea
                  name="injuryHistory" value={profile.injuryHistory} onChange={handleChange} placeholder="Detail any joint, muscle, or tendon issues (e.g. Right ACL reconstruction 2021)..." rows="3"
                  className="w-full bg-[#050505] border border-zinc-900 focus:border-white transition-colors p-4 text-white font-mono text-sm outline-none resize-none"
                />
              </div>
            </div>
          </div>

          {/* Action Row */}
          <div className="pt-8 border-t border-zinc-900 flex justify-between items-center">
            {saved ? (
              <span className="text-green-500 font-bold uppercase text-[10px] tracking-widest flex items-center">
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                Profile Synced
              </span>
            ) : (
              <span className="text-zinc-600 font-mono text-xs">Unsaved changes will be discarded</span>
            )}

            <button
              onClick={handleSave}
              className="px-10 py-5 bg-white text-black font-bold uppercase text-[10px] tracking-widest hover:bg-zinc-200 transition shadow-[0_0_30px_rgba(255,255,255,0.1)]"
            >
              Save Profile Context
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Profile;
