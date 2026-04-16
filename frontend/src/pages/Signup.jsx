import { useState } from "react";
import { useNavigate } from "react-router-dom";

function Signup() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSignup = (e) => {
    e.preventDefault();
    
    // Store user id (email) and password in localStorage
    localStorage.setItem('athlix_user_email', email);
    localStorage.setItem('athlix_user_password', password);
    
    // Simulate signup success and navigate to login or dashboard
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col justify-center items-center font-sans selection:bg-zinc-700">
      
      {/* Top Left Logo */}
      <div 
        className="absolute top-8 left-8 md:top-12 md:left-12 flex items-center gap-3 cursor-pointer"
        onClick={() => navigate('/')}
      >
        <div className="w-6 h-6 bg-white text-black flex items-center justify-center rounded-sm">
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
        </div>
        <span className="font-bold text-xl tracking-tight uppercase">Athlix</span>
      </div>

      {/* Main Signup Module */}
      <div className="w-full max-w-sm px-6">
        <h1 className="text-4xl md:text-5xl font-black uppercase tracking-tighter mb-4 text-center">
          Register
        </h1>
        <p className="text-zinc-500 font-light text-center mb-16 tracking-wide text-sm">
          Initialize your biomechanical profile and movement tracking workspace.
        </p>

        <form onSubmit={handleSignup} className="space-y-10">
          <div>
            <label className="block text-[10px] font-bold tracking-[0.2em] uppercase text-zinc-500 mb-2">
              Email Address
            </label>
            <input 
              type="email" 
              required
              placeholder="athlete@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-transparent text-white border-b border-zinc-700 py-3 outline-none focus:border-white transition-colors tracking-wide font-light placeholder:text-zinc-800"
            />
          </div>

          <div>
            <label className="block text-[10px] font-bold tracking-[0.2em] uppercase text-zinc-500 mb-2">
              Password
            </label>
            <input 
              type="password" 
              required
              placeholder="••••••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-transparent text-white border-b border-zinc-700 py-3 outline-none focus:border-white transition-colors tracking-widest font-light placeholder:text-zinc-800"
            />
          </div>

          <button 
            type="submit"
            className="w-full pt-6 pb-6 bg-white text-black text-[10px] font-bold tracking-[0.2em] uppercase hover:bg-zinc-200 transition shadow-2xl mt-4"
          >
            Create Profile
          </button>
        </form>

        <div className="mt-8 text-center">
            <button 
              onClick={() => navigate('/login')}
              className="text-[10px] text-zinc-500 font-bold uppercase tracking-[0.1em] hover:text-white transition-colors"
            >
                Already have an account? Login
            </button>
        </div>

        <div className="mt-16 flex justify-center text-center">
             <p className="text-[10px] text-zinc-600 font-bold uppercase tracking-[0.2em] border-b border-zinc-800 pb-1 inline-block">
               Secure Environment
             </p>
        </div>
      </div>

    </div>
  );
}

export default Signup;
