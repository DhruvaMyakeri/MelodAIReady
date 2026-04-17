import { useEffect, useRef } from 'react';
import type { JobStatus } from '../types';

interface Props {
  status: JobStatus | null;
}

const LOG_TAG_COLORS: Record<string, string> = {
  ERROR: '#FF2020',
  WARN: '#FFE500',
  DONE: '#39FF14',
};

function colorLine(line: string): string {
  if (line.includes('[ERROR]') || line.includes('Error') || line.includes('failed')) return '#FF2020';
  if (line.includes('[WARN]') || line.includes('WARNING')) return '#FFE500';
  if (line.includes('[DONE]') || line.includes('complete') || line.includes('Done')) return '#39FF14';
  return '#39FF14';
}

export default function LogTerminal({ status }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [status?.log]);

  const lines = status?.log ?? [];

  return (
    <div className="log-terminal">
      <div className="log-terminal-header">
        <span className="log-blink">█</span>
        <span>[ SYSTEM LOG ]</span>
        <span className="log-pid">PID: {status?.job_id?.slice(0, 8) ?? '--------'}</span>
      </div>
      <div className="log-terminal-body">
        {lines.length === 0 && (
          <div className="log-line" style={{ color: '#666' }}>
            {'> '} waiting for pipeline output...
          </div>
        )}
        {lines.map((line, i) => (
          <div
            key={i}
            className="log-line"
            style={{ color: colorLine(line) }}
          >
            {'> '}{line}
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div className="log-scanlines" aria-hidden="true" />
    </div>
  );
}
