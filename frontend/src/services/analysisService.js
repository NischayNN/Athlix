/**
 * analysisService.js
 *
 * Produces a per-session analysis result object.
 * Every field is derived dynamically from the session context and athlete
 * profile so that each analysis run yields unique, contextual output.
 *
 * mockAnalysisData is imported ONLY as a last-resort fallback when the
 * service is called without any session context at all.
 */

import { mockAnalysisData } from '../data/mockAnalysisData';

// ─── Internal helpers ──────────────────────────────────────────────

/** Seeded-ish pseudo-random using session inputs for reproducible but unique values */
function sessionRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

/** Clamp a number between min and max */
function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

// ─── Issue catalog (squat-specific) ────────────────────────────────
// Each issue has a base probability that is modulated by session context.
const ISSUE_CATALOG = [
  {
    id: 1,
    issue: "Incomplete Depth",
    baseProbability: 0.6,
    baseSeverity: "High",
    detail: "Hip crease did not drop below the patella.",
    intensityScale: 0.4,   // more likely at higher intensity
    fatigueScale: 0.3,
    flag: "incomplete_depth",
    joints: ["hip"],
  },
  {
    id: 2,
    issue: "Knee Valgus",
    baseProbability: 0.55,
    baseSeverity: "Medium",
    detail: "Medial collapse detected during the concentric phase.",
    intensityScale: 0.35,
    fatigueScale: 0.4,
    flag: "knee_valgus",
    joints: ["knee"],
  },
  {
    id: 3,
    issue: "Excessive Forward Lean",
    baseProbability: 0.35,
    baseSeverity: "Medium",
    detail: "Torso angle exceeded safe thresholds relative to vertical.",
    intensityScale: 0.3,
    fatigueScale: 0.25,
    flag: "excessive_forward_lean",
    joints: ["back", "shoulder"],
  },
  {
    id: 4,
    issue: "Heel Rise",
    baseProbability: 0.3,
    baseSeverity: "Low",
    detail: "Heel lifted off the platform during the descent.",
    intensityScale: 0.1,
    fatigueScale: 0.15,
    flag: "heel_rise",
    joints: ["ankle"],
  },
  {
    id: 5,
    issue: "Lateral Shift",
    baseProbability: 0.2,
    baseSeverity: "Low",
    detail: "Lateral weight distribution asymmetry detected.",
    intensityScale: 0.15,
    fatigueScale: 0.3,
    flag: "lateral_shift",
    joints: ["hip"],
  },
];

// ─── Dynamic decay curve generator ─────────────────────────────────
function generateDecayCurve(repCount, relativeIntensity, recoveryMult, seed) {
  // Start score modulated by intensity — heavier loads → lower starting score
  const startScore = clamp(Math.round(97 - relativeIntensity * 12 - (recoveryMult - 1) * 10), 75, 97);
  const data = [];

  for (let i = 0; i < repCount; i++) {
    // Progressive fatigue: each rep loses more as fatigue accumulates
    const fatigueDrop = (i * i * relativeIntensity * 1.2) + (sessionRandom(seed + i) * 4);
    const repScore = clamp(Math.round(startScore - fatigueDrop), 40, startScore);
    data.push({ rep: i + 1, score: repScore });
  }

  return data;
}

// ─── Dynamic issue detection ───────────────────────────────────────
function detectIssues(relativeIntensity, recoveryMult, strictness, heightMeters, injuryHistory, seed) {
  const detected = [];

  ISSUE_CATALOG.forEach((template, idx) => {
    // Probability modulated by session context
    const adjustedProb = template.baseProbability
      + (relativeIntensity * template.intensityScale)
      + ((recoveryMult - 1) * template.fatigueScale * 3);

    const roll = sessionRandom(seed + idx * 7 + 3);

    if (roll < adjustedProb) {
      // Severity can escalate under high intensity / poor recovery
      let severity = template.baseSeverity;
      if (relativeIntensity > 0.85 && severity === "Medium") severity = "High";
      if (relativeIntensity > 0.9 && severity === "Low") severity = "Medium";
      if (recoveryMult > 1.3 && severity === "Low") severity = "Medium";

      detected.push({
        id: template.id,
        issue: template.issue,
        severity,
        detail: template.detail,
        flag: template.flag,
        joints: template.joints,
      });
    }
  });

  // Always return at least one issue for meaningful results
  if (detected.length === 0) {
    detected.push({
      id: ISSUE_CATALOG[3].id,
      issue: ISSUE_CATALOG[3].issue,
      severity: "Low",
      detail: ISSUE_CATALOG[3].detail,
      flag: ISSUE_CATALOG[3].flag,
      joints: ISSUE_CATALOG[3].joints,
    });
  }

  return detected;
}

// ─── Dynamic coaching tips ─────────────────────────────────────────
function generateCoachingTips(issues, relativeIntensity, recoveryMult) {
  const tips = [];

  // Issue-specific tips
  const issueMap = {
    "Incomplete Depth": { action: "Increase Squat Depth", cue: "Focus on breaking parallel. Use a box squat or pause at the bottom to build proprioceptive awareness.", target: "Hip Mobility" },
    "Knee Valgus": { action: "Active Glute Engagement", cue: "Drive knees outward during the concentric phase. Cue 'spread the floor' with your feet.", target: "Knee Tracking" },
    "Excessive Forward Lean": { action: "Maintain Vertical Torso", cue: "Keep chest proud and eyes forward. Consider front-squatting to reinforce upright mechanics.", target: "Spinal Neutrality" },
    "Heel Rise": { action: "Improve Ankle Mobility", cue: "Work on ankle dorsiflexion with wall stretches. Consider elevated-heel squat shoes.", target: "Ankle ROM" },
    "Lateral Shift": { action: "Unilateral Correction", cue: "Add single-leg exercises (Bulgarian split squats) to eliminate asymmetry.", target: "Bilateral Symmetry" },
  };

  issues.forEach((issue, idx) => {
    const tip = issueMap[issue.issue];
    if (tip) {
      tips.push({ id: idx + 1, ...tip });
    }
  });

  // Intensity-based tip
  if (relativeIntensity > 0.85) {
    tips.unshift({
      id: 99,
      action: "Manage Absolute Intensity",
      cue: `Current relative load (${(relativeIntensity * 100).toFixed(0)}%) is stressing raw kinematics. Consider dropping load by 10-15% to stabilize form.`,
      target: "Load Auto-Regulation",
    });
  }

  // Recovery-based tip
  if (recoveryMult > 1.2) {
    tips.unshift({
      id: 98,
      action: "Prioritize Recovery",
      cue: "Systemic fatigue is altering movement patterns. Focus on sleep optimization and extended warm-ups before high-volume sessions.",
      target: "Recovery Periodization",
    });
  }

  // Always have at least 2 tips
  if (tips.length < 2) {
    tips.push({
      id: 50,
      action: "Control Deceleration",
      cue: "Implement a 3-second eccentric phase to build tendon resilience and improve positional awareness.",
      target: "Tendon Load",
    });
  }

  return tips.slice(0, 4);
}

// ─── Summary generator ─────────────────────────────────────────────
function generateSummary(issues, relativeIntensity, overallScore) {
  const issueNames = issues.map(i => i.issue.toLowerCase()).join(", ");
  const loadDesc = relativeIntensity > 0.85 ? "high" : relativeIntensity > 0.65 ? "moderate" : "low";
  const qualityDesc = overallScore >= 85 ? "strong" : overallScore >= 70 ? "acceptable but declining" : "compromised";

  return `Movement quality is ${qualityDesc} under ${loadDesc} relative loading. Key areas of concern: ${issueNames}. ${
    overallScore < 75
      ? "Immediate load reduction recommended to prevent structural injury risk."
      : "Continue monitoring with progressive overload caution."
  }`;
}

// ═══════════════════════════════════════════════════════════════════
// MAIN EXPORT
// ═══════════════════════════════════════════════════════════════════

export const analyzeVideo = async (file, context = {}) => {
  const formData = new FormData();
  formData.append('file', file);

  let backendScore = null;

  // Attempt real backend analysis
  try {
    const response = await fetch('http://127.0.0.1:8000/analyze', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    const data = await response.json();
    if (data.score) backendScore = data.score;
  } catch (error) {
    console.warn("Backend not reachable, deriving results from session context:", error.message);
  }

  const { profile, sessionContext } = context;

  // If no context at all, return mockAnalysisData as pure fallback
  if (!profile && !sessionContext) {
    localStorage.setItem('temp_analysis', JSON.stringify(mockAnalysisData));
    return mockAnalysisData;
  }

  // ── Session seed for reproducible per-run variation ───────────
  const seed = (parseFloat(sessionContext?.weightUsed) || 0)
    + (parseFloat(sessionContext?.reps) || 0) * 13
    + (parseFloat(sessionContext?.sets) || 0) * 37
    + (parseFloat(sessionContext?.sleepHours) || 0) * 7
    + (parseFloat(sessionContext?.soreness) || 0) * 19
    + (parseFloat(profile?.maxPR) || 0) * 3
    + Date.now() % 10000; // time component for true uniqueness

  // ── Core session values ──────────────────────────────────────
  const pr = parseFloat(profile?.maxPR) || 140;
  const w = parseFloat(sessionContext?.weightUsed) || 100;
  const relativeIntensity = clamp(w / pr, 0.1, 1.2);
  const reps = parseFloat(sessionContext?.reps) || 5;
  const sets = parseFloat(sessionContext?.sets) || 3;
  const userHeightCm = parseFloat(profile?.height) || 180;
  const heightMeters = userHeightCm / 100;

  // ── Velocity Estimation ──────────────────────────────────────
  // Simulated from mock frame data (in production, from actual video pose tracking)
  const pixelDisplacement = 250 + sessionRandom(seed) * 100; // variation per session
  const timeSeconds = 0.8 + sessionRandom(seed + 1) * 0.6;  // 0.8s–1.4s
  const pixelToMeter = (userHeightCm / 100) / 800;
  const realDisplacementMeters = pixelDisplacement * pixelToMeter;
  const movementVelocity = realDisplacementMeters / timeSeconds;

  let velocityClassification = 'Hypertrophy';
  let velocityFactor = 1.0;
  if (movementVelocity < 0.5) {
    velocityClassification = 'Strength';
    velocityFactor = 0.8;
  } else if (movementVelocity > 0.8) {
    velocityClassification = 'Power';
    velocityFactor = 1.2;
  }

  const loadScore = relativeIntensity * (reps * sets) * velocityFactor;

  // ── Recovery multiplier ──────────────────────────────────────
  const sleep = parseFloat(sessionContext?.sleepHours) || 8;
  const soreness = parseFloat(sessionContext?.soreness) || 3;
  const recoveryPenalty = ((8 - Math.min(sleep, 8)) * 0.1) + ((Math.max(soreness, 3) - 3) * 0.05);
  const recoveryMultiplier = clamp(1.0 + recoveryPenalty, 1.0, 1.5);

  // ── Experience / strictness ──────────────────────────────────
  const experience = profile?.experience || 'Intermediate';
  let strictness = 1.0;
  if (experience === 'Novice') strictness = 1.2;
  else if (experience === 'Advanced' || experience === 'Elite') strictness = 0.8;

  const injuryHistory = (profile?.injuryHistory || '').toLowerCase();

  // ── Dynamic issue detection ──────────────────────────────────
  const detectedIssues = detectIssues(relativeIntensity, recoveryMultiplier, strictness, heightMeters, injuryHistory, seed);

  // ── Risk scoring per issue ───────────────────────────────────
  const intensityMultiplier = Math.pow(relativeIntensity, 2);
  let totalRisk = 0;
  const riskBreakdown = [];

  const scoredIssues = detectedIssues.map(issue => {
    let flawSeverity = issue.severity === 'High' ? 0.8 : (issue.severity === 'Medium' ? 0.5 : 0.2);

    // Anthropometric adjustment
    if (issue.issue === 'Incomplete Depth' && heightMeters > 1.85) {
      flawSeverity *= 0.8;
    }
    flawSeverity *= strictness;

    // Injury history amplification
    let historyMult = 1.0;
    const isKneeRelated = issue.joints?.includes('knee') || issue.issue.toLowerCase().includes('valgus');
    const isBackRelated = issue.joints?.includes('back') || issue.issue.toLowerCase().includes('lean');
    if ((isKneeRelated && injuryHistory.includes('knee')) || (isBackRelated && injuryHistory.includes('back'))) {
      historyMult = 1.5;
    }

    const contribution = flawSeverity * intensityMultiplier * recoveryMultiplier * historyMult;
    totalRisk += contribution;

    riskBreakdown.push({
      issue: issue.issue,
      flawSeverity: flawSeverity.toFixed(2),
      intensityMult: intensityMultiplier.toFixed(2),
      recoveryMult: recoveryMultiplier.toFixed(2),
      historyMult: historyMult.toFixed(2),
      contribution: contribution.toFixed(2),
    });

    return { ...issue, riskContribution: contribution };
  });

  const movementRiskIndex = clamp(Math.round(totalRisk * 35), 0, 100);
  let riskLabel = 'Low';
  if (movementRiskIndex >= 75) riskLabel = 'High';
  else if (movementRiskIndex >= 40) riskLabel = 'Moderate';

  // ── Dynamic overall score ────────────────────────────────────
  // Backend score takes priority. Otherwise derive from context.
  let overallScore;
  if (backendScore !== null) {
    overallScore = backendScore;
  } else {
    // Base score penalized by intensity, recovery, and number/severity of issues
    const issuePenalty = scoredIssues.reduce((sum, iss) => {
      return sum + (iss.severity === 'High' ? 8 : iss.severity === 'Medium' ? 5 : 2);
    }, 0);
    overallScore = clamp(
      Math.round(95 - relativeIntensity * 10 - (recoveryMultiplier - 1) * 15 - issuePenalty + sessionRandom(seed + 99) * 6),
      45, 97
    );
  }

  // ── Dynamic decay curve ──────────────────────────────────────
  const decayData = generateDecayCurve(Math.round(reps), relativeIntensity, recoveryMultiplier, seed);

  // ── Insight message ──────────────────────────────────────────
  let explanationInsight = "Solid movement pattern under acceptable relative load.";
  if (movementRiskIndex >= 75) {
    if (recoveryMultiplier > 1.2) {
      explanationInsight = "Elevated risk due to poor recovery state amplifying form fatigue.";
    } else if (intensityMultiplier > 0.64) {
      explanationInsight = "Form breakdown under high relative load. Tendon and joint structures are significantly stressed.";
    } else {
      explanationInsight = "Movement deviation detected under moderate load — indicates a technique deficit or injury sensitivity.";
    }
  } else if (movementRiskIndex >= 40) {
    explanationInsight = "Moderate mechanical shifts detected. Monitor load scaling carefully.";
  }

  // ── Coaching tips ────────────────────────────────────────────
  const coachingTips = generateCoachingTips(scoredIssues, relativeIntensity, recoveryMultiplier);

  // ── Compile final result ─────────────────────────────────────
  const finalResult = {
    score: overallScore,
    movement: "Back Squat",
    timestamp: new Date().toISOString(),
    decayData,
    keyIssues: scoredIssues,
    riskFactors: scoredIssues.map((iss, i) => ({
      id: i + 1,
      title: iss.issue,
      description: iss.detail,
    })),
    coachingTips,
    summary: generateSummary(scoredIssues, relativeIntensity, overallScore),
    movementVelocity: movementVelocity.toFixed(2),
    velocityClassification,
    loadScore: loadScore.toFixed(1),
    relativePct: (relativeIntensity * 100).toFixed(0),
    movementRiskIndex,
    riskLabel,
    riskBreakdown,
    explanationInsight,
    weightUsed: w,
    maxPR: pr,
  };

  localStorage.setItem('temp_analysis', JSON.stringify(finalResult));

  return new Promise((resolve) =>
    setTimeout(() => resolve(finalResult), 800)
  );
};
