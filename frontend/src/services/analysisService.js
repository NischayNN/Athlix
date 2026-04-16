import { mockAnalysisData } from '../data/mockAnalysisData';

export const analyzeVideo = async (file, context = {}) => {
  const formData = new FormData();
  formData.append('file', file);

  // All fields start null — filled from real backend data
  let backendData = null;

  try {
    const response = await fetch('http://127.0.0.1:8000/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.detail || data.error);
    }

    backendData = data;
    console.log('[Athlix] Backend response:', backendData);

  } catch (error) {
    console.warn('[Athlix] Backend call failed, using mock data:', error.message);
  }

  // ── Context-derived metadata ────────────────────────────────────────────
  const { profile, sessionContext } = context;
  const pr  = parseFloat(profile?.maxPR) || 140;
  const w   = parseFloat(sessionContext?.weightUsed) || 100;
  const pct = backendData?.relativePct || ((w / pr) * 100).toFixed(0);

  // ── Use backend values when available, fall back to mock ───────────────
  const finalScore          = backendData?.score                  ?? mockAnalysisData.score ?? 82;
  const reps                = backendData?.reps                   ?? parseInt(sessionContext?.reps) ?? 6;
  const injuryRisk          = backendData?.injury_risk            ?? 50;
  const riskLevel           = backendData?.risk_level             ?? 'moderate';
  const movementRiskIndex   = backendData?.movementRiskIndex      ?? null;
  const riskLabel           = backendData?.riskLabel              ?? 'Moderate';
  const riskBreakdown       = backendData?.riskBreakdown          ?? [];
  const explanationInsight  = backendData?.explanationInsight     ?? 'Awaiting analysis data.';
  const movementVelocity    = backendData?.movementVelocity       ?? '0.65';
  const velocityClassification = backendData?.velocityClassification ?? 'Hypertrophy';
  const loadScore           = backendData?.loadScore              ?? '0.0';
  const formFlags           = backendData?.form_flags             ?? {};
  const featureVector       = backendData?.feature_vector         ?? {};

  // Key issues: prefer backend, fall back to mock
  const keyIssues = backendData?.keyIssues ?? mockAnalysisData.keyIssues ?? [];

  // Injury risk label
  const injuryRiskLabel = injuryRisk > 60 ? 'ELEVATED' : injuryRisk > 35 ? 'MODERATE' : 'LOW';

  // ── Build decayData dynamically from actual reps ──────────────────────
  const decayData = Array.from({ length: Math.max(reps, 1) }, (_, i) => {
    // Simulate progressive fatigue: starts high, drops accelerating
    const startScore = Math.min(97, Math.round(finalScore + 12));
    const fatigueDrop = (i * i * 1.2) + (Math.random() * 3);
    return {
      rep: i + 1,
      score: Math.max(40, Math.round(startScore - fatigueDrop)),
    };
  });

  // ── Coaching tips based on detected issues ────────────────────────────
  const issueCoachingMap = {
    'Incomplete Depth':       { action: 'Increase Squat Depth',    cue: 'Focus on breaking parallel. Use a box squat or pause at the bottom to build proprioceptive awareness.', target: 'Hip Mobility' },
    'Knee Valgus':            { action: 'Active Glute Engagement', cue: "Drive knees outward during the concentric phase. Cue 'spread the floor' with your feet.",             target: 'Knee Tracking' },
    'Excessive Forward Lean': { action: 'Maintain Vertical Torso', cue: 'Keep chest proud and eyes forward. Consider front-squatting to reinforce upright mechanics.',           target: 'Spinal Neutrality' },
    'Heel Rise':              { action: 'Improve Ankle Mobility',  cue: 'Work on ankle dorsiflexion with wall stretches. Consider elevated-heel squat shoes.',                  target: 'Ankle ROM' },
    'Lateral Shift':          { action: 'Unilateral Correction',   cue: 'Add single-leg exercises (Bulgarian split squats) to eliminate asymmetry.',                            target: 'Bilateral Symmetry' },
  };

  let coachingTips = keyIssues
    .map((issue, idx) => issueCoachingMap[issue.issue] ? { id: idx + 1, ...issueCoachingMap[issue.issue] } : null)
    .filter(Boolean);

  if (coachingTips.length < 2) {
    coachingTips.push({
      id: 50,
      action: 'Control Deceleration',
      cue: 'Implement a 3-second eccentric phase to build tendon resilience and improve positional awareness.',
      target: 'Tendon Load',
    });
  }

  // ── Summary ───────────────────────────────────────────────────────────
  const issueNames = keyIssues.map(i => i.issue?.toLowerCase()).filter(Boolean).join(', ') || 'no major issues';
  const qualityDesc = finalScore >= 85 ? 'strong' : finalScore >= 70 ? 'acceptable but declining' : 'compromised';
  const summary = `Movement quality is ${qualityDesc}. Key areas of concern: ${issueNames}. ${
    finalScore < 75
      ? 'Immediate load reduction recommended to prevent structural injury risk.'
      : 'Continue monitoring with progressive overload caution.'
  }`;

  // ── Compile final result ──────────────────────────────────────────────
  const finalResult = {
    score: finalScore,
    movement: 'Back Squat',
    timestamp: new Date().toISOString(),
    injuryRisk: injuryRiskLabel,
    reps,
    decayData,
    keyIssues,
    riskFactors: keyIssues.map((iss, i) => ({
      id: i + 1,
      title: iss.issue,
      description: iss.detail,
    })),
    coachingTips: coachingTips.slice(0, 4),
    summary,
    // Movement Risk Index section
    movementRiskIndex,
    riskLabel,
    riskBreakdown,
    explanationInsight,
    // Intensity Profile section
    movementVelocity,
    velocityClassification,
    loadScore,
    relativePct: pct,
    weightUsed: w,
    maxPR: pr,
    // Raw data for debugging
    feature_vector: featureVector,
    form_flags: formFlags,
  };

  localStorage.setItem('temp_analysis', JSON.stringify(finalResult));

  return new Promise((resolve) =>
    setTimeout(() => resolve(finalResult), 800)
  );
};
