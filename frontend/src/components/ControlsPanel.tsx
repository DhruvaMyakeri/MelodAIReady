import type { UploadConfig, Instrument, OutputMode } from '../types';
import { INSTRUMENT_COLORS } from '../types';

interface Props {
  config: UploadConfig;
  onChange: (c: UploadConfig) => void;
  hasFile: boolean;
  onSubmit: () => void;
  isSubmitting: boolean;
}

const INSTRUMENTS: Instrument[] = ['Piano', 'Guitar', 'Violin', 'Sitar', 'Flute', 'Trumpet'];

export default function ControlsPanel({ config, onChange, hasFile, onSubmit, isSubmitting }: Props) {
  const set = (partial: Partial<UploadConfig>) => onChange({ ...config, ...partial });

  return (
    <div className="controls-panel">
      {/* INSTRUMENT SELECTOR */}
      <div className="control-group">
        <div className="control-label">INSTRUMENT</div>
        <div className="instrument-grid">
          {INSTRUMENTS.map(inst => {
            const color = INSTRUMENT_COLORS[inst];
            const selected = config.instrument === inst;
            return (
              <button
                key={inst}
                className={`instrument-btn${selected ? ' selected' : ''}`}
                style={{
                  '--accent': color,
                } as React.CSSProperties}
                onClick={() => set({ instrument: inst })}
                aria-pressed={selected}
              >
                {inst.toUpperCase()}
              </button>
            );
          })}
        </div>
      </div>

      {/* OUTPUT MODE */}
      <div className="control-group">
        <div className="control-label">OUTPUT FORMAT</div>
        <div className="toggle-row">
          {(['midi', 'sheet'] as OutputMode[]).map(mode => (
            <button
              key={mode}
              className={`mode-btn${config.output_mode === mode ? ' selected' : ''}`}
              onClick={() => set({ output_mode: mode })}
              aria-pressed={config.output_mode === mode}
            >
              {mode === 'midi' ? '[ MIDI ]' : '[ SHEET MUSIC ]'}
            </button>
          ))}
        </div>
      </div>

      {/* THRESHOLD SLIDER */}
      <div className="control-group">
        <div className="control-label">
          THRESHOLD: <span className="threshold-value">{config.threshold.toFixed(2)}</span>
        </div>
        <div className="slider-wrapper">
          <input
            type="range"
            className="threshold-slider"
            min={0.1}
            max={0.9}
            step={0.05}
            value={config.threshold}
            onChange={e => set({ threshold: parseFloat(e.target.value) })}
            style={{ '--fill-pct': `${((config.threshold - 0.1) / 0.8) * 100}%` } as React.CSSProperties}
          />
          <div className="slider-labels">
            <span>0.10</span>
            <span>0.50</span>
            <span>0.90</span>
          </div>
        </div>
      </div>

      {/* SKIP DEMUCS */}
      <div className="control-group">
        <label className="checkbox-row" htmlFor="skip-demucs">
          <div
            id="skip-demucs"
            className={`brutal-checkbox${config.skip_demucs ? ' checked' : ''}`}
            role="checkbox"
            aria-checked={config.skip_demucs}
            tabIndex={0}
            onClick={() => set({ skip_demucs: !config.skip_demucs })}
            onKeyDown={e => e.key === ' ' && set({ skip_demucs: !config.skip_demucs })}
          >
            {config.skip_demucs && <span>✕</span>}
          </div>
          <span className="checkbox-label">SKIP VOCAL REMOVAL (3× FASTER)</span>
        </label>
        {!config.skip_demucs && (
          <div className="demucs-warning">⚠ ADDS 4–6 MIN ON CPU</div>
        )}
      </div>

      {/* SUBMIT */}
      <button
        className={`submit-btn${!hasFile || isSubmitting ? ' disabled' : ''}`}
        onClick={onSubmit}
        disabled={!hasFile || isSubmitting}
        aria-label="Run MelodAI pipeline"
      >
        {isSubmitting ? 'LAUNCHING...' : 'RUN MELODAI →'}
      </button>
    </div>
  );
}
