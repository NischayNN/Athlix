<<<<<<< HEAD
/**
 * Service to handle video upload and analysis API calls.
 */

export const analyzeVideo = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
=======
export const processFrameAnalysis = async (imageFile, trainingLoad, sleepHours) => {
  const formData = new FormData();
  formData.append("file", imageFile);
  formData.append("training_load", trainingLoad);
  formData.append("sleep_hours", sleepHours);

  try {
    const response = await fetch("http://localhost:8000/process-frame", {
      method: "POST",
>>>>>>> a5eab63228f7298a9d13b7950589b3dcce337927
      body: formData,
    });

    if (!response.ok) {
<<<<<<< HEAD
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error analyzing video:", error);
=======
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Error connecting to backend");
    }

    return await response.json();
  } catch (error) {
    console.error("Analysis Service Error:", error);
>>>>>>> a5eab63228f7298a9d13b7950589b3dcce337927
    throw error;
  }
};
