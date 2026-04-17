import { useEffect, useRef, useState } from 'react';
import type { Phase, JobStatus } from '../types';
import { PHASE_COLORS } from '../types';
import LogTerminal from './LogTerminal';

interface Props {
  status: JobStatus | null;
  onAbort: () => void;
}

const PHASE_ORDER: Phase[] = [
  'DEMUCS', 'BASIC_PITCH', 'LIBROSA', 'CNN_CLASSIFIER',
  'KEY_DETECT', 'ARRANGEMENT', 'MIDI_GEN', 'OUTPUT',
];

const TOTAL_BLOCKS = 20;

export default function ProcessingView({ status, onAbort }: Props) {
  const [displayPhase, setDisplayPhase] = useState<Phase | null>(null);
  const [phaseAnim, setPhaseAnim] = useState<'enter' | 'exit'>('enter');
  const prevPhase = useRef<Phase | null>(null);

  useEffect(() => {
    const current = status?.phase ?? null;
    if (current && current !== prevPhase.current) {
      setPhaseAnim('exit');
      const t = setTimeout(() => {
        setDisplayPhase(current);
        setPhaseAnim('enter');
        prevPhase.current = current;
      }, 250);
      return () => clearTimeout(t);
    }
  }, [status?.phase]);

  const progress = status?.progress ?? 0;
  const phase = status?.phase ?? 'DEMUCS';
  const phaseColor = PHASE_COLORS[displayPhase ?? phase] ?? '#FFE500';
  const filledBlocks = Math.round((progress / 100) * TOTAL_BLOCKS);

  const isError = phase === 'ERROR';

  return (
    <div className="processing-view">
      {/* Ghost label */}
      <div className="ghost-label" aria-hidden="true">PROCESS</div>

      {/* Phase name — concert poster style */}
      <div className={`phase-display phase-anim-${phaseAnim}`} style={{ color: phaseColor }}>
        {displayPhase ?? phase}
      </div>
      <div className="phase-sub" style={{ color: phaseColor }}>
        {isError ? 'PIPELINE FAILED' : 'PIPELINE ACTIVE'}
      </div>

      {/* Progress blocks */}
      <div className="progress-area">
        <div className="progress-blocks">
          {Array.from({ length: TOTAL_BLOCKS }).map((_, i) => (
            <div
              key={i}
              className={`progress-block${i < filledBlocks ? ' filled' : ''}`}
              style={i < filledBlocks ? {
                background: phaseColor,
                borderColor: phaseColor,
              } : { borderColor: phaseColor + '44' }}
            />
          ))}
        </div>
        <div className="progress-pct" style={{ color: phaseColor }}>
          {progress}%
        </div>
      </div>

      {/* Pipeline phase map */}
      <div className="phase-map">
        {PHASE_ORDER.map(p => {
          const phaseIdx = PHASE_ORDER.indexOf(phase as Phase);
          const thisIdx = PHASE_ORDER.indexOf(p);
          const isDone = thisIdx < phaseIdx || phase === 'DONE';
          const isActive = p === phase;
          const c = PHASE_COLORS[p];
          return (
            <div
              key={p}
              className={`phase-chip${isActive ? ' active' : ''}${isDone ? ' done' : ''}`}
              style={{
                borderColor: isActive || isDone ? c : '#333',
                color: isActive ? c : isDone ? c + 'aa' : '#444',
              }}
            >
              {isDone && !isActive && '✓ '}{p.replace('_', ' ')}
            </div>
          );
        })}
      </div>

      {/* Error state */}
      {isError && (
        <div className="error-block">
          <div className="error-headline">PIPELINE FAILED</div>
          <div className="error-detail">{status?.error}</div>
          <button className="try-again-btn" onClick={onAbort}>
            [ TRY AGAIN ]
          </button>
        </div>
      )}

      {/* Log terminal */}
      <LogTerminal status={status} />

      {/* Abort */}
      <div className="processing-footer">
        <button className="abort-btn" onClick={onAbort}>
          [ ABORT ]
        </button>
      </div>
    </div>
  );
}
