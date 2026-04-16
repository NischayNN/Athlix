import { useState } from "react";

// ─── Focused joint data: only squat-relevant lower-body chain ──────
// Coordinates normalized within an 80×100 SVG viewBox.
// Structured for easy backend replacement.
const DEFAULT_ORIGINAL_POSE = {
  shoulder: { x: 38, y: 22 },
  hip:      { x: 36, y: 44 },
  knee:     { x: 30, y: 65 },
  ankle:    { x: 28, y: 88 },
};

const DEFAULT_CORRECTED_POSE = {
  shoulder: { x: 40, y: 20 },
  hip:      { x: 38, y: 48 },
  knee:     { x: 30, y: 68 },
  ankle:    { x: 28, y: 88 },
};

// Focused limb chain: shoulder → hip → knee → ankle
const LIMB_CHAIN = [
  ["shoulder", "hip"],
  ["hip", "knee"],
  ["knee", "ankle"],
];

// Labels keyed to form_flags, anchored to corrected joints
const CORRECTION_LABELS = {
  knee_valgus: {
    label: "Knee tracking corrected",
    joint: "knee",
  },
  incomplete_depth: {
    label: "Depth improved",
    joint: "hip",
  },
  excessive_forward_lean: {
    label: "Torso uprighted",
    joint: "shoulder",
  },
};

// ─── Single-chain skeleton renderer ────────────────────────────────
function ChainSkeleton({ pose, color, opacity, strokeWidth, jointRadius }) {
  return (
    <g opacity={opacity}>
      {LIMB_CHAIN.map(([a, b], i) => {
        const from = pose[a];
        const to = pose[b];
        if (!from || !to) return null;
        return (
          <line
            key={i}
            x1={from.x} y1={from.y}
            x2={to.x} y2={to.y}
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        );
      })}
      {Object.entries(pose).map(([key, pos]) => (
        <circle
          key={key}
          cx={pos.x} cy={pos.y}
          r={jointRadius}
          fill={color}
        />
      ))}
    </g>
  );
}

// ═══════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════
export default function FormCorrectionPreview({
  originalPose = DEFAULT_ORIGINAL_POSE,
  correctedPose = DEFAULT_CORRECTED_POSE,
  formFlags = { knee_valgus: true, incomplete_depth: true, excessive_forward_lean: false },
}) {
  const [view, setView] = useState("both"); // "original" | "corrected" | "both"

  const showOriginal = view === "original" || view === "both";
  const showCorrected = view === "corrected" || view === "both";

  // Active correction labels based on flags
  const activeLabels = Object.entries(CORRECTION_LABELS)
    .filter(([flag]) => formFlags?.[flag])
    .map(([, cfg]) => cfg);

  return (
    <div className="bg-[#1c1c1e] rounded-[32px] p-8 relative overflow-hidden">
      {/* Header row */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold tracking-tight">Form Correction Preview</h3>
          <p className="text-zinc-500 text-sm font-medium mt-1">Lower-body kinematic chain</p>
        </div>

        {/* Segmented toggle */}
        <div className="flex bg-black/50 rounded-full p-0.5 border border-white/5">
          {[
            { key: "original", label: "Current" },
            { key: "both", label: "Compare" },
            { key: "corrected", label: "Corrected" },
          ].map((opt) => (
            <button
              key={opt.key}
              onClick={() => setView(opt.key)}
              className={`px-3 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-widest transition-colors ${
                view === opt.key
                  ? "bg-white/10 text-white"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Canvas */}
      <div className="w-full aspect-[3/4] max-w-sm mx-auto rounded-2xl relative overflow-hidden border border-white/5 bg-gradient-to-b from-[#111113] to-[#0a0a0c]">
        <svg
          viewBox="0 0 80 100"
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Original — faded ghost */}
          {showOriginal && (
            <ChainSkeleton
              pose={originalPose}
              color="#ff453a"
              opacity={0.25}
              strokeWidth={1.5}
              jointRadius={2.5}
            />
          )}

          {/* Corrected — primary */}
          {showCorrected && (
            <>
              <ChainSkeleton
                pose={correctedPose}
                color="#30d158"
                opacity={0.85}
                strokeWidth={2.5}
                jointRadius={3.5}
              />

              {/* Joint labels — minimal inline text */}
              {activeLabels.map((cfg, i) => {
                const pos = correctedPose[cfg.joint];
                if (!pos) return null;
                return (
                  <text
                    key={i}
                    x={pos.x + 5}
                    y={pos.y + 1}
                    fill="#a1a1aa"
                    fontSize="3.2"
                    fontFamily="system-ui, -apple-system, sans-serif"
                    fontWeight="600"
                  >
                    {cfg.label}
                  </text>
                );
              })}
            </>
          )}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-5">
        {showOriginal && (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#ff453a] opacity-40" />
            <span className="text-[11px] text-zinc-500 font-medium">Current</span>
          </div>
        )}
        {showCorrected && (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#30d158]" />
            <span className="text-[11px] text-zinc-500 font-medium">Corrected</span>
          </div>
        )}
      </div>
    </div>
  );
}
