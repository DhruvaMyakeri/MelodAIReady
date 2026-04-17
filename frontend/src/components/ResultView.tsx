import AudioPlayer from './AudioPlayer';
import DownloadPanel from './DownloadPanel';
import type { JobStatus, Instrument, OutputMode } from '../types';

interface Props {
  jobId: string;
  status: JobStatus | null;
  instrument: Instrument;
  outputMode: OutputMode;
  onReset: () => void;
}

export default function ResultView({ jobId, status, instrument, outputMode, onReset }: Props) {
  return (
    <div className="result-view">
      <div className="result-header">
        <div className="result-title">OUTPUT READY</div>
        <div className="stamp-badge" style={{ borderColor: '#39FF14', color: '#39FF14' }}>
          ✓ DONE
        </div>
      </div>

      <div className="result-columns">
        {/* Left — Audio player */}
        <div className="result-col result-col-left">
          <AudioPlayer
            jobId={jobId}
            instrument={instrument}
            outputMode={outputMode}
          />
        </div>

        {/* Right — Download + info */}
        <div className="result-col result-col-right">
          <DownloadPanel
            jobId={jobId}
            status={status}
            outputMode={outputMode}
            instrument={instrument}
            onReset={onReset}
          />
        </div>
      </div>
    </div>
  );
}
