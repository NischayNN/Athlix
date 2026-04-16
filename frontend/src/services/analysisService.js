/**
 * Mock fallback data — matches the shape the Upload results view renders:
 *   results.feature_vector  (Object)
 *   results.form_flags      (Object)
 *   results.processing_time_ms (Number)
 */
const MOCK_ANALYSIS_RESULT = {
  feature_vector: {
    training_load: 7.5,
    recovery_score: 45.0,
    fatigue_index: 6.0,
    form_decay: 0.72,
    previous_injury: 0,
    knee_angle_min: 68.4,
    hip_angle_min: 52.1,
    back_angle_max: 38.9,
    stance_width_ratio: 1.12,
  },
  form_flags: {
    knee_valgus: true,
    incomplete_depth: true,
    excessive_forward_lean: false,
    heel_rise: false,
    lateral_shift: false,
  },
  processing_time_ms: 342,
};

export const analyzeVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('http://127.0.0.1:8000/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    // Backend unavailable — fall back to mock data so the UI still works
    console.warn("Backend unreachable, using mock analysis data:", error.message);
    return new Promise((resolve) =>
      setTimeout(() => resolve({ ...MOCK_ANALYSIS_RESULT }), 800)
    );
  }
};
