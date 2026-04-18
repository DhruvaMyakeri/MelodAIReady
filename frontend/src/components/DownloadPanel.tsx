import { getDownloadUrl } from '../api/client';
import type { JobStatus, Instrument, OutputMode } from '../types';

interface Props {
  jobId: string;
  status: JobStatus | null;
  outputMode: OutputMode;
  instrument: Instrument;
  onReset: () => void;
}

const INSTRUMENT_CONFIDENCE_COLORS = ['#FFE500', '#00FFE0', '#39FF14', '#FF2D78', '#FF2020'];

export default function DownloadPanel({ jobId, status, outputMode, instrument, onReset }: Props) {
  const ext = outputMode === 'midi' ? '.MID' : '.XML';
  const downloadUrl = getDownloadUrl(jobId);

  // Extract some info from last log lines
  const logs = status?.log ?? [];
  const noteCountLine = logs.find(l => l.includes('notes') || l.includes('Written'));
  const noteCount = noteCountLine?.match(/(\d+)\s+note/)?.[1] ?? '—';

  // Mock confidence display (in real use you'd have this from status)
  const mockConfidences: { label: string; value: number; color: string }[] = [
    { label: instrument.toUpperCase(), value: 0.87, color: '#FFE500' },
    { label: 'AMBIENT', value: 0.32, color: '#00FFE0' },
    { label: 'NOISE', value: 0.11, color: '#FF2D78' },
  ];

  const CONF_BLOCKS = 10;

  return (
    <div className="download-panel">
      <div className="ghost-label" aria-hidden="true" style={{ fontSize: '5rem', opacity: 0.03 }}>OUTPUT</div>

      {/* Download button */}
      <div style={{ position: 'relative' }}>
        <a
          href={downloadUrl}
          download
          className="download-btn"
          aria-label={`Download ${ext} file`}
        >
          ↓ DOWNLOAD
        </a>
        <div className="caveat-note" style={{ top: '-12px', right: '-15px', fontWeight: 700, fontSize: '22px', color: '#39FF14', opacity: 0.55, transform: 'rotate(1.5deg)' }}>
          it's ready. finally.
        </div>
      </div>

      {/* File type stamp */}
      <div className="file-type-stamp">
        <span className="file-type-badge">{ext}</span>
        <span className="output-mode-label">{outputMode === 'midi' ? 'MIDI AUDIO FILE' : 'MUSICXML SHEET'}</span>
      </div>

      {/* Info block */}
      <div className="info-block">
        <div className="info-row">
          <span className="info-key">STATUS</span>
          <span className="info-val" style={{ color: '#39FF14' }}>PIPELINE COMPLETE</span>
        </div>
        <div className="info-row">
          <span className="info-key">INSTRUMENT</span>
          <span className="info-val">{instrument.toUpperCase()}</span>
        </div>
        <div className="info-row">
          <span className="info-key">NOTE EVENTS</span>
          <span className="info-val">{noteCount}</span>
        </div>
        <div className="info-row">
          <span className="info-key">OUTPUT</span>
          <span className="info-val">{ext.replace('.', '')}</span>
        </div>
      </div>

      {/* Instrument confidence bars */}
      <div className="confidence-section">
        <div className="confidence-title">DETECTOR CONFIDENCE</div>
        {mockConfidences.map((item, idx) => {
          const filled = Math.round(item.value * CONF_BLOCKS);
          return (
            <div key={item.label} className="confidence-row">
              <span className="conf-label" style={{ color: item.color }}>{item.label}</span>
              <div className="conf-blocks">
                {Array.from({ length: CONF_BLOCKS }).map((_, i) => (
                  <div
                    key={i}
                    className="conf-block"
                    style={{
                      background: i < filled ? item.color : 'transparent',
                      borderColor: item.color + '66',
                    }}
                  />
                ))}
              </div>
              <span className="conf-pct" style={{ color: item.color }}>
                {Math.round(item.value * 100)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* Recent log */}
      <div className="result-log">
        {logs.slice(-4).map((line, i) => (
          <div key={i} className="result-log-line">&gt; {line}</div>
        ))}
      </div>

      {/* Reset button */}
      <button className="reset-btn" onClick={onReset}>
        [ PROCESS ANOTHER FILE ]
      </button>
    </div>
  );
}
