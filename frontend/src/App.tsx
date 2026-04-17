import { useState, useCallback } from 'react';
import Background from './components/Background';
import UploadZone from './components/UploadZone';
import ControlsPanel from './components/ControlsPanel';
import ProcessingView from './components/ProcessingView';
import ResultView from './components/ResultView';
import { useJobPolling } from './hooks/useJobPolling';
import { submitJob } from './api/client';
import type { UploadConfig, Instrument } from './types';

const DEFAULT_CONFIG: UploadConfig = {
  instrument: 'Piano',
  output_mode: 'midi',
  skip_demucs: false,
  threshold: 0.3,
};

type View = 'upload' | 'processing' | 'result';

export default function App() {
  const [view, setView] = useState<View>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [config, setConfig] = useState<UploadConfig>(DEFAULT_CONFIG);
  const [jobId, setJobId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const { status, stop } = useJobPolling(view === 'processing' ? jobId : null);

  // Transition to result when done
  const prevPhase = status?.phase;
  if (prevPhase === 'DONE' && view === 'processing') {
    setView('result');
  }

  const handleSubmit = useCallback(async () => {
    if (!file) return;
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const id = await submitJob(file, config);
      setJobId(id);
      setView('processing');
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setIsSubmitting(false);
    }
  }, [file, config]);

  const handleAbort = useCallback(() => {
    stop();
    setJobId(null);
    setView('upload');
  }, [stop]);

  const handleReset = useCallback(() => {
    stop();
    setJobId(null);
    setFile(null);
    setConfig(DEFAULT_CONFIG);
    setView('upload');
  }, [stop]);

  return (
    <div className="app">
      <Background />

      <div className="page-content">
        {/* ─── HERO ─────────────────────────────────────────────────── */}
        <header className="hero">
          <div className="hero-stamp">
            <span>[ML-POWERED]</span>
          </div>
          <h1 className="hero-title">MelodAI</h1>
          <div className="hero-sub">
            AUDIO INTELLIGENCE. NO CLOUD. NO COMPROMISE.
          </div>
          <div className="hero-tags">
            <span className="stamp-badge">[CNN]</span>
            <span className="stamp-badge">[MIDI]</span>
            <span className="stamp-badge">[v3.0]</span>
            <span className="stamp-badge">[OFFLINE]</span>
          </div>
        </header>

        {/* ─── UPLOAD STATE ─────────────────────────────────────────── */}
        <div className={`view-section upload-section${view !== 'upload' ? ' view-hidden' : ''}`}>
          <div className="ghost-label" aria-hidden="true">UPLOAD</div>

          <div className="upload-container">
            <UploadZone file={file} onChange={setFile} />

            {submitError && (
              <div className="submit-error">⚠ {submitError}</div>
            )}

            <ControlsPanel
              config={config}
              onChange={setConfig}
              hasFile={!!file}
              onSubmit={handleSubmit}
              isSubmitting={isSubmitting}
            />
          </div>
        </div>

        {/* ─── PROCESSING STATE ─────────────────────────────────────── */}
        <div className={`view-section${view !== 'processing' ? ' view-hidden' : ''}`}>
          <ProcessingView status={status} onAbort={handleAbort} />
        </div>

        {/* ─── RESULT STATE ─────────────────────────────────────────── */}
        <div className={`view-section${view !== 'result' ? ' view-hidden' : ''}`}>
          {jobId && view === 'result' && (
            <ResultView
              jobId={jobId}
              status={status}
              instrument={config.instrument}
              outputMode={config.output_mode}
              onReset={handleReset}
            />
          )}
        </div>

        {/* ─── FOOTER ───────────────────────────────────────────────── */}
        <footer className="site-footer">
          <span>MELODAI</span>
          <span className="stamp-badge">[NO RULES. NOISE.]</span>
          <span>CPU-ONLY · OFFLINE · OPEN SOURCE</span>
        </footer>
      </div>
    </div>
  );
}
