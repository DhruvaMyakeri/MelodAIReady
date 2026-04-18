import { useEffect, useRef, useState, useCallback } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';
import { getAudioUrl, getDownloadUrl } from '../api/client';
import type { Instrument } from '../types';
import { useAudioSync } from '../hooks/useAudioSync';

interface Props {
  jobId: string;
  instrument: Instrument;
  outputMode: 'midi' | 'sheet';
}

function formatTime(s: number): string {
  if (!isFinite(s) || isNaN(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

export default function AudioPlayer({ jobId, instrument, outputMode }: Props) {
  const sync = useAudioSync();
  const audioRef = useRef<HTMLAudioElement>(null);
  const samplerRef = useRef<Tone.Sampler | null>(null);
  const [midiLoaded, setMidiLoaded] = useState(false);
  const [midiError, setMidiError] = useState<string | null>(null);

  // Connect audio element to sync hook
  useEffect(() => {
    sync.setOriginalRef(audioRef.current);
  }, [sync]);

  // Cleanup on unmount AND load
  useEffect(() => {
    return () => {
      Tone.Transport.stop();
      Tone.Transport.cancel(0);
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
      samplerRef.current?.dispose();
    };
  }, []);

  // Load and schedule MIDI via Tone.js
  useEffect(() => {
    if (outputMode !== 'midi') return;

    let cancelled = false;

    const loadMidi = async () => {
      try {
        const url = getDownloadUrl(jobId);
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to load MIDI');
        const buf = await res.arrayBuffer();
        const midi = new Midi(buf);

        if (cancelled) return;

        // Clean up previous transport events
        Tone.Transport.cancel(0);
        if (samplerRef.current) {
          samplerRef.current.dispose();
          samplerRef.current = null;
        }

        const SF_NAMES: Record<Instrument, string> = {
          Piano: 'acoustic_grand_piano',
          Guitar: 'acoustic_guitar_nylon',
          'Electric Guitar': 'electric_guitar_clean',
          'Acoustic Electric Guitar': 'acoustic_guitar_steel',
          Violin: 'violin',
          Sitar: 'sitar',
          Flute: 'flute',
          Trumpet: 'trumpet'
        };

        const sf = SF_NAMES[instrument] || 'acoustic_grand_piano';

        const sampler = new Tone.Sampler({
          urls: {
            A1: "A1.mp3",
            C2: "C2.mp3",
            A2: "A2.mp3",
            C3: "C3.mp3",
            A3: "A3.mp3",
            C4: "C4.mp3",
            A4: "A4.mp3",
            C5: "C5.mp3",
            A5: "A5.mp3",
            C6: "C6.mp3",
            A6: "A6.mp3",
            C7: "C7.mp3"
          },
          release: 1,
          baseUrl: `https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/${sf}-mp3/`
        }).toDestination();
        
        // Apply current sync levels
        const toneVolume = sync.midiVolume ? (sync.midiLevel * 20 - 10) : -Infinity;
        sampler.volume.value = toneVolume;
        samplerRef.current = sampler;

        // Make sure Tone loads all the MP3 samples before proceeding
        await Tone.loaded();

        midi.tracks.forEach(track => {
          track.notes.forEach(note => {
            Tone.Transport.schedule(time => {
              sampler.triggerAttackRelease(
                note.name,
                note.duration,
                time,
                note.velocity
              );
            }, note.time);
          });
        });

        setMidiLoaded(true);
      } catch (e) {
        if (!cancelled) setMidiError(String(e));
      }
    };

    loadMidi();
    return () => { cancelled = true; };
  }, [jobId, outputMode]); // Intentionally omitting sync levels so we don't reload midi on volume shift

  // Sync Tone.js volume manually when state updates
  useEffect(() => {
    if (samplerRef.current) {
      if (!sync.midiVolume) samplerRef.current.volume.value = -Infinity;
      else samplerRef.current.volume.value = sync.midiLevel * 20 - 10;
    }
  }, [sync.midiLevel, sync.midiVolume]);

  const handleTogglePlay = useCallback(async () => {
    await Tone.start();

    if (sync.isPlaying) {
      Tone.Transport.pause();
      audioRef.current?.pause();
      sync.togglePlay();
    } else {
      if (Tone.Transport.state === 'stopped' || Tone.Transport.state === 'paused') {
        Tone.Transport.start();
      }
      audioRef.current?.play().catch(console.error);
      sync.togglePlay();
    }
  }, [sync]);

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value);
    sync.seek(t);
    if (Tone.Transport.state !== 'stopped') {
      Tone.Transport.seconds = t;
    }
  }, [sync]);

  const progress = sync.duration > 0 ? (sync.currentTime / sync.duration) * 100 : 0;

  return (
    <div className="audio-player">
      <div className="ghost-label" aria-hidden="true" style={{ fontSize: '6rem', opacity: 0.03 }}>PLAY</div>

      <div className="player-title">AUDIO PLAYBACK</div>

      {/* Play / Pause */}
      <div style={{ position: 'relative' }}>
        <div className="caveat-note" style={{ top: '-15px', right: '10px', fontWeight: 400, fontSize: '18px', color: '#FF2D78', opacity: 0.5, transform: 'rotate(-2deg)' }}>
          play it loud.
        </div>
        <button className="play-btn" onClick={handleTogglePlay} aria-label="Toggle playback">
          {sync.isPlaying ? 'PAUSE' : 'PLAY'}
        </button>
      </div>

      {/* Track rows */}
      <div className="track-rows">
        <TrackRow
          label="ORIGINAL"
          color="#FFE500"
          progress={progress}
          currentTime={sync.currentTime}
          duration={sync.duration}
          volOn={sync.originalVolume}
          onToggleVol={sync.toggleOriginalVolume}
          volumeLevel={sync.originalLevel}
          onChangeLevel={sync.setOriginalVolumeLevel}
        />
        <TrackRow
          label="MELODAI"
          color="#00FFE0"
          progress={progress}
          currentTime={sync.currentTime}
          duration={sync.duration}
          volOn={sync.midiVolume}
          onToggleVol={sync.toggleMidiVolume}
          volumeLevel={sync.midiLevel}
          onChangeLevel={sync.setMidiVolumeLevel}
          disabled={outputMode !== 'midi' || !midiLoaded}
        />
      </div>

      {midiError && (
        <div className="player-error">⚠ MIDI PREVIEW UNAVAILABLE: {midiError}</div>
      )}

      {/* Scrubber */}
      <div className="scrubber-row">
        <span className="time-display">{formatTime(sync.currentTime)}</span>
        <input
          type="range"
          className="scrubber"
          min={0}
          max={sync.duration || 100}
          step={0.1}
          value={sync.currentTime}
          onChange={handleSeek}
          style={{ '--fill-pct': `${progress}%` } as React.CSSProperties}
        />
        <span className="time-display">{formatTime(sync.duration)}</span>
      </div>

      {/* Hidden audio element */}
      <audio
        ref={r => {
          (audioRef as React.MutableRefObject<HTMLAudioElement | null>).current = r;
          sync.setOriginalRef(r);
        }}
        src={getAudioUrl(jobId)}
        preload="metadata"
      />
    </div>
  );
}

interface TrackRowProps {
  label: string;
  color: string;
  progress: number;
  currentTime: number;
  duration: number;
  volOn: boolean;
  onToggleVol: () => void;
  volumeLevel: number;
  onChangeLevel: (val: number) => void;
  disabled?: boolean;
}

function TrackRow({ label, color, progress, currentTime, duration, volOn, onToggleVol, volumeLevel, onChangeLevel, disabled }: TrackRowProps) {
  const BLOCKS = 16;
  const filled = Math.round((progress / 100) * BLOCKS);

  return (
    <div className={`track-row${disabled ? ' disabled' : ''}`} style={{ '--track-color': color } as React.CSSProperties}>
      <div className="track-label" style={{ color }}>[{label}]</div>
      <div className="track-blocks">
        {Array.from({ length: BLOCKS }).map((_, i) => (
          <div
            key={i}
            className={`track-block${i < filled ? ' filled' : ''}`}
            style={{ background: i < filled ? color : 'transparent', borderColor: color + '55' }}
          />
        ))}
      </div>
      <span className="track-time">{formatTime(currentTime)} / {formatTime(duration)}</span>
      <div className="vol-controls" style={{ position: 'relative' }}>
        {label === 'MELODAI' && (
          <div className="caveat-note" style={{ top: '-15px', right: '0', fontWeight: 400, fontSize: '14px', color: '#FFE500', opacity: 0.45, transform: 'rotate(1deg)' }}>
            or don't. your call.
          </div>
        )}
        <button
          className={`vol-btn${volOn ? '' : ' muted'}`}
          onClick={onToggleVol}
          disabled={disabled}
          style={{ borderColor: color, color }}
        >
          {volOn ? '[VOL: ON]' : '[VOL: OFF]'}
        </button>
        <input
          type="range"
          className="vol-slider"
          min="0"
          max="1"
          step="0.01"
          value={volumeLevel}
          onChange={(e) => onChangeLevel(parseFloat(e.target.value))}
          disabled={disabled || !volOn}
          style={{ '--fill-pct': `${volumeLevel * 100}%`, borderColor: color } as React.CSSProperties}
        />
      </div>
    </div>
  );
}
