import { useRef, useState, useCallback } from 'react';

interface Props {
  file: File | null;
  onChange: (f: File | null) => void;
}

const ACCEPTED = ['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/x-flac', 'audio/mp3'];

export default function UploadZone({ file, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback((f: File) => {
    if (ACCEPTED.includes(f.type) || /\.(mp3|wav|flac)$/i.test(f.name)) {
      onChange(f);
    }
  }, [onChange]);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };
  const onClick = () => inputRef.current?.click();
  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="upload-zone-wrapper">
      {!file ? (
        <div
          className={`upload-zone${isDragging ? ' is-dragging' : ''}`}
          onClick={onClick}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          role="button"
          tabIndex={0}
          onKeyDown={e => e.key === 'Enter' && onClick()}
          aria-label="Upload audio file"
        >
          <input
            ref={inputRef}
            type="file"
            accept=".mp3,.wav,.flac,audio/*"
            style={{ display: 'none' }}
            onChange={onInputChange}
          />
          <div className="upload-zone-icon">▼</div>
          <div className="upload-zone-main">DROP AUDIO HERE</div>
          <div className="upload-zone-sub">MP3 / WAV / FLAC</div>
          <div className="upload-zone-hint">or click to browse</div>
        </div>
      ) : (
        <div className="file-ticket">
          <div className="file-ticket-inner">
            <span className="file-ticket-icon">◉</span>
            <div className="file-ticket-info">
              <div className="file-ticket-name">{file.name}</div>
              <div className="file-ticket-size">{formatSize(file.size)}</div>
            </div>
            <button
              className="file-ticket-clear"
              onClick={() => onChange(null)}
              aria-label="Remove file"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
