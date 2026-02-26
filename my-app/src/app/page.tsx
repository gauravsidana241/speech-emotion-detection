"use client";

import { useState, useRef, useEffect } from "react";
import "./main.scss";

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [timeLeft, setTimeLeft] = useState(3.3);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);
  const animFrameRef = useRef<number | null>(null);

  const RECORDING_LIMIT = 3.3; // seconds

  const emotions: Record<string, string> = {
    happy: "😊",
    sad: "😢",
    angry: "😡",
    fear: "😨",
    disgust: "🤢",
    neutral: "😐",
    surprise: "😲",
    calm: "😌",
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        setAudioURL(url);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setAudioURL(null);
      setPrediction(null);
      setTimeLeft(RECORDING_LIMIT);
      startTimeRef.current = Date.now();

      // Countdown with requestAnimationFrame for smooth updates
      const updateTimer = () => {
        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        const remaining = Math.max(0, RECORDING_LIMIT - elapsed);
        setTimeLeft(remaining);

        if (remaining > 0) {
          animFrameRef.current = requestAnimationFrame(updateTimer);
        }
      };
      animFrameRef.current = requestAnimationFrame(updateTimer);

      // Auto-stop after limit
      timerRef.current = setTimeout(() => {
        stopRecording();
      }, RECORDING_LIMIT * 1000);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    setTimeLeft(0);

    if (timerRef.current) clearTimeout(timerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
  };

  const analyzeAudio = async () => {
    if (!audioURL) return;
    setIsAnalyzing(true);

    try {
      const blob = await fetch(audioURL).then((r) => r.blob());
      const formData = new FormData();
      formData.append("audio", blob, "recording.wav");

      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/predict`,
        { method: "POST", body: formData }
      );

      const data = await res.json();
      console.log("Backend response:", data);

      if (data.error) {
        console.error("Backend error:", data.error);
      } else {
        setPrediction(data.emotion);
      }
    } catch (err) {
      console.error("Request failed:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const reset = () => {
    setAudioURL(null);
    setPrediction(null);
    setTimeLeft(RECORDING_LIMIT);
    setIsRecording(false);
    setIsAnalyzing(false);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  const progress = ((RECORDING_LIMIT - timeLeft) / RECORDING_LIMIT) * 100;

  return (
    <div className="main-wrapper">
      <div className="dialog-box">
        {/* Header */}
        <div className="header">
          <div className="title-group">
            <h1>Speech Emotion Recognition</h1>
            <p className="subtitle">Record a short audio clip and let the AI detect the emotion</p>
          </div>
          <div className="badge">
            <span className="dot" />
            AI Powered
          </div>
        </div>

        {/* Recorder Area */}
        <div className="recorder-area">
          {/* Waveform / Visual */}
          <div className={`visualizer ${isRecording ? "active" : ""}`}>
            <div className="bars">
              {Array.from({ length: 24 }).map((_, i) => (
                <div key={i} className="bar" style={{ animationDelay: `${i * 0.05}s` }} />
              ))}
            </div>
          </div>
          {!isRecording && !audioURL && (
              <p className="hint">Tap the mic to start recording</p>
            )}

          {/* Timer */}
          {isRecording && (
            <div className="timer-section">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <span className="time-display">{timeLeft.toFixed(1)}s remaining</span>
            </div>
          )}

          {/* Controls */}
          <div className="controls">
            {!audioURL ? (
              <button
                className={`mic-btn ${isRecording ? "recording" : ""}`}
                onClick={isRecording ? stopRecording : startRecording}
              >
                <div className="mic-icon">
                  {isRecording ? (
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  ) : (
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="9" y="2" width="6" height="11" rx="3" />
                      <path d="M5 10a7 7 0 0 0 14 0" />
                      <line x1="12" y1="19" x2="12" y2="22" />
                    </svg>
                  )}
                </div>
                <span>{isRecording ? "Stop" : "Record"}</span>
              </button>
            ) : (
              <div className="post-recording">
                <audio controls src={audioURL} className="audio-player" />
                <div className="action-buttons">
                  <button className="btn-analyze" onClick={analyzeAudio} disabled={isAnalyzing}>
                    {isAnalyzing ? (
                      <>
                        <span className="spinner" /> Analyzing...
                      </>
                    ) : (
                      "Detect Emotion"
                    )}
                  </button>
                  <button className="btn-reset" onClick={reset}>
                    Re-record
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Result */}
        {prediction && (
          <div className="result-card">
            <span className="result-emoji">{emotions[prediction] || "🎭"}</span>
            <div className="result-text">
              <span className="result-label">Detected Emotion</span>
              <span className="result-value">{prediction}</span>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="footer">
          <p>Max recording: {RECORDING_LIMIT}</p>
        </div>
      </div>
    </div>
  );
}