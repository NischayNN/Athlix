import { Link, useLocation } from "react-router-dom";

function Navbar() {
  const location = useLocation();
  const path = location.pathname;

  // The Analysis page is a cinematic screen, hiding the nav builds immersion
  if (path === "/analysis") return null;

  const isHome = path === "/";
  // Public routes: landing page and login
  const isPublicRoute = path === "/" || path === "/login";

  return (
    <nav className={`w-full z-50 ${isHome ? 'absolute top-0 bg-transparent text-white' : 'sticky top-0 bg-black border-b border-zinc-900 text-white'}`}>
      <div className="max-w-7xl mx-auto px-6 py-6 md:px-12 flex justify-between items-center">
        
        {/* Logo and Brand */}
        <Link to="/" className="text-xl font-bold tracking-tight flex items-center gap-3 group">
          <div className="w-6 h-6 bg-white text-black flex items-center justify-center rounded-sm transition-transform group-hover:scale-105">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
          </div>
          <span className="uppercase tracking-widest text-sm">Athlix</span>
        </Link>

        {/* Navigation Links — conditional on route context */}
        <div className="flex items-center gap-8 text-[10px] uppercase tracking-widest font-bold">
          {isPublicRoute ? (
            <>
              {/* Public navigation */}
              <Link to="/" className="hover:text-zinc-400 transition-colors hidden sm:block">Home</Link>
              <Link to="/#about" className="hover:text-zinc-400 transition-colors hidden sm:block">About</Link>
              <Link to="/login" className="border border-zinc-800 hover:border-white px-5 py-2 hover:bg-white hover:text-black transition-all">
                Login
              </Link>
            </>
          ) : (
            <>
              {/* Product navigation — only visible after login flow */}
              <Link to="/dashboard" className="hover:text-zinc-400 transition-colors hidden sm:block">Dashboard</Link>
              <Link to="/upload" className="border border-zinc-800 hover:border-white px-5 py-2 hover:bg-white hover:text-black transition-all">
                New Analysis
              </Link>
            </>
          )}
        </div>

      </div>
    </nav>
  );
}

export default Navbar;

