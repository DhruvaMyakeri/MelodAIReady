/**
 * AudioPlayer.tsx  —  DAW-grade MIDI playback engine
 *
 * Architecture
 * ────────────
 * • HTML <audio> element  → ONLY master clock
 * • Tone.Sampler           → sound generation (no Transport)
 * • Lookahead scheduler    → runs every SCHEDULER_INTERVAL_MS,
 *                            looks LOOKAHEAD_S ahead, schedules notes
 *                            into the Web Audio context using
 *                            Tone.now() + (noteTime - audio.currentTime)
 * • Drift correction       → every tick, compare scheduler position
 *                            with audio.currentTime; if deviation
 *                            exceeds DRIFT_THRESHOLD_S, hard-reset
 *
 * Seeking
 * ───────
 * 1. useAudioSync.seek() moves audio.currentTime
 * 2. A registered callback fires → resetScheduler(newTime)
 * 3. resetScheduler cancels pending Tone events, recomputes
 *    the next-note index, clears the scheduled-set
 *
 * Volume
 * ──────
 * Tone.Sampler.volume follows sync.midiLevel / sync.midiVolume
 * Original audio volume is handled by useAudioSync (on the <audio> element)
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';
import { getAudioUrl, getDownloadUrl } from '../api/client';
import type { Instrument } from '../types';
import { useAudioSync } from '../hooks/useAudioSync';

// ─── Scheduler constants ──────────────────────────────────────────────────────
const SCHEDULER_INTERVAL_MS = 25;   // how often the scheduler fires
const LOOKAHEAD_S            = 0.12; // seconds ahead to schedule
const DRIFT_THRESHOLD_S      = 0.04; // reset if drift > 40 ms

// ─── Soundfont map ────────────────────────────────────────────────────────────
const SF_NAMES: Record<Instrument, string> = {
  Piano:                    'acoustic_grand_piano',
  Guitar:                   'acoustic_guitar_nylon',
  'Electric Guitar':        'electric_guitar_clean',
  'Acoustic Electric Guitar':'acoustic_guitar_steel',
  Violin:                   'violin',
  Sitar:                    'sitar',
  Flute:                    'flute',
  Trumpet:                  'trumpet',
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatTime(s: number): string {
  if (!isFinite(s) || isNaN(s)) return '0:00';
  const m   = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

/** Binary-search for the first note index at or after `time`. */
function findFirstNoteIndex(
  notes: Array<{ time: number }>,
  time: number,
): number {
  let lo = 0, hi = notes.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (notes[mid].time < time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ─── Component ────────────────────────────────────────────────────────────────

interface Props {
  jobId:      string;
  instrument: Instrument;
  outputMode: 'midi' | 'sheet';
}

interface ScheduledNote {
  time:     number;   // seconds from song start
  name:     string;   // e.g. "C4"
  duration: number;   // seconds
  velocity: number;   // 0-1
}

export default function AudioPlayer({ jobId, instrument, outputMode }: Props) {
  const sync = useAudioSync();

  const audioRef   = useRef<HTMLAudioElement | null>(null);
  const samplerRef = useRef<Tone.Sampler | null>(null);

  // ── MIDI data ─────────────────────────────────────────────────────────────
  const allNotesRef       = useRef<ScheduledNote[]>([]);  // flat, time-sorted
  const [midiLoaded,  setMidiLoaded]  = useState(false);
  const [midiError,   setMidiError]   = useState<string | null>(null);

  // ── Scheduler state ───────────────────────────────────────────────────────
  const nextNoteIdxRef     = useRef(0);
  const scheduledUntilRef  = useRef(0);   // wall-song-time up to which we've scheduled
  const scheduledSet       = useRef<Set<number>>(new Set()); // note indices already triggered
  const schedulerTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const isPlayingRef       = useRef(false); // mirror of sync.isPlaying without stale closure

  // ─── Wire audio element to sync hook ──────────────────────────────────────
  useEffect(() => {
    sync.setOriginalRef(audioRef.current);
  }, [sync]);

  // ─── Cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopScheduler();
      samplerRef.current?.dispose();
      samplerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─── Scheduler core ───────────────────────────────────────────────────────

  /** Cancel all pending Tone events and reset scheduling cursors. */
  const resetScheduler = useCallback((fromTime: number) => {
    // Disconnect any previously scheduled Tone callbacks that haven't fired
    // Tone.js doesn't expose "cancel specific event" for schedule() well,
    // so we cancel ALL and set a guard via scheduledSet generation.
    Tone.Transport.cancel(0);           // wipe transport queue
    Tone.Transport.stop();
    Tone.Transport.seconds = 0;

    scheduledSet.current.clear();
    scheduledUntilRef.current = fromTime;
    nextNoteIdxRef.current    = findFirstNoteIndex(allNotesRef.current, fromTime);
  }, []);

  /** One tick of the lookahead scheduler — driven by audio.currentTime. */
  const schedulerTick = useCallback(() => {
    const audio    = audioRef.current;
    const sampler  = samplerRef.current;
    if (!audio || !sampler || !isPlayingRef.current) return;

    const now       = audio.currentTime;           // master clock
    const windowEnd = now + LOOKAHEAD_S;

    // ── Drift correction ──
    if (Math.abs(now - scheduledUntilRef.current) > DRIFT_THRESHOLD_S) {
      // scheduler fell behind or was jumped — realign silently
      nextNoteIdxRef.current   = findFirstNoteIndex(allNotesRef.current, now);
      scheduledUntilRef.current = now;
      scheduledSet.current.clear();
    }

    const notes = allNotesRef.current;
    let   idx   = nextNoteIdxRef.current;

    while (idx < notes.length && notes[idx].time < windowEnd) {
      const n = notes[idx];

      if (!scheduledSet.current.has(idx) && n.time >= now - 0.005) {
        scheduledSet.current.add(idx);

        // Convert song-position to Web Audio absolute time
        const audioCtxOffset = n.time - now;          // seconds from NOW
        const toneAbsTime    = Tone.now() + Math.max(0, audioCtxOffset);

        sampler.triggerAttackRelease(
          n.name,
          n.duration,
          toneAbsTime,
          n.velocity,
        );
      }
      idx++;
    }

    nextNoteIdxRef.current    = idx;
    scheduledUntilRef.current = windowEnd;
  }, []);

  const startScheduler = useCallback(() => {
    if (schedulerTimerRef.current) return;
    schedulerTimerRef.current = setInterval(schedulerTick, SCHEDULER_INTERVAL_MS);
  }, [schedulerTick]);

  const stopScheduler = useCallback(() => {
    if (schedulerTimerRef.current) {
      clearInterval(schedulerTimerRef.current);
      schedulerTimerRef.current = null;
    }
  }, []);

  // ─── Register seek callback with the sync hook ────────────────────────────
  useEffect(() => {
    sync.registerSeekCallback((time: number) => {
      resetScheduler(time);
      // If currently playing, scheduler loop will continue from new position.
      // If paused, it will pick up correctly on next play.
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);   // intentionally once — resetScheduler is stable

  // ─── Load MIDI on mount / jobId change ───────────────────────────────────
  useEffect(() => {
    if (outputMode !== 'midi') return;

    let cancelled = false;

    const loadMidi = async () => {
      try {
        const url = getDownloadUrl(jobId);
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf  = await res.arrayBuffer();
        const midi = new Midi(buf);

        if (cancelled) return;

        // ── Flatten all tracks into a single time-sorted note array ──
        const flat: ScheduledNote[] = [];
        midi.tracks.forEach(track => {
          track.notes.forEach(n => {
            flat.push({
              time:     n.time,
              name:     n.name,
              duration: Math.max(0.02, n.duration),
              velocity: n.velocity,
            });
          });
        });
        flat.sort((a, b) => a.time - b.time);
        allNotesRef.current = flat;

        // ── Build sampler ──
        const sf = SF_NAMES[instrument] ?? 'acoustic_grand_piano';

        if (samplerRef.current) {
          samplerRef.current.dispose();
          samplerRef.current = null;
        }

        const sampler = new Tone.Sampler({
          urls: {
            A1: 'A1.mp3', C2: 'C2.mp3',
            A2: 'A2.mp3', C3: 'C3.mp3',
            A3: 'A3.mp3', C4: 'C4.mp3',
            A4: 'A4.mp3', C5: 'C5.mp3',
            A5: 'A5.mp3', C6: 'C6.mp3',
            A6: 'A6.mp3', C7: 'C7.mp3',
          },
          release: 1,
          baseUrl: `https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/${sf}-mp3/`,
        }).toDestination();

        // Apply current volume
        sampler.volume.value = sync.midiVolume
          ? (sync.midiLevel * 20 - 10)
          : -Infinity;

        samplerRef.current = sampler;
        await Tone.loaded();

        if (cancelled) return;

        // Reset scheduler to beginning
        resetScheduler(audioRef.current?.currentTime ?? 0);

        setMidiLoaded(true);
        setMidiError(null);
      } catch (e) {
        if (!cancelled) setMidiError(String(e));
      }
    };

    loadMidi();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, outputMode]);

  // ─── Volume sync ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!samplerRef.current) return;
    samplerRef.current.volume.value = sync.midiVolume
      ? (sync.midiLevel * 20 - 10)
      : -Infinity;
  }, [sync.midiLevel, sync.midiVolume]);

  // ─── Play / Pause ─────────────────────────────────────────────────────────
  const handleTogglePlay = useCallback(async () => {
    await Tone.start(); // unlock AudioContext on first interaction

    if (sync.isPlaying) {
      // ── PAUSE ──
      audioRef.current?.pause();
      isPlayingRef.current = false;
      stopScheduler();
      sync.togglePlay();
    } else {
      // ── PLAY ──
      const audio = audioRef.current;
      if (!audio) return;

      // Ensure scheduler starts from current audio position
      resetScheduler(audio.currentTime);
      isPlayingRef.current = true;
      startScheduler();

      audio.play().catch(console.error);
      sync.togglePlay();
    }
  }, [sync, stopScheduler, startScheduler, resetScheduler]);

  // ─── Scrubber seek ────────────────────────────────────────────────────────
  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value);
    // useAudioSync.seek() moves audio.currentTime AND fires registerSeekCallback
    // which calls resetScheduler(t) — no extra work needed here.
    sync.seek(t);
  }, [sync]);

  // ─── Keep isPlayingRef in sync with state ────────────────────────────────
  useEffect(() => {
    isPlayingRef.current = sync.isPlaying;
    if (!sync.isPlaying) stopScheduler();
    // Scheduler is started explicitly inside handleTogglePlay, not here,
    // to avoid double-start.
  }, [sync.isPlaying, stopScheduler]);

  const progress = sync.duration > 0
    ? (sync.currentTime / sync.duration) * 100
    : 0;

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="audio-player">
      <div className="ghost-label" aria-hidden="true" style={{ fontSize: '6rem', opacity: 0.03 }}>PLAY</div>

      <div className="player-title">AUDIO PLAYBACK</div>

      {/* Play / Pause */}
      <div style={{ position: 'relative' }}>
        <div 
          className="caveat-note" 
          style={{ 
            top: '-15px', 
            right: '10px', 
            fontWeight: 400, 
            fontSize: '18px', 
            color: '#FF2D78', 
            opacity: 0.5, 
            transform: 'rotate(-2deg)',
            pointerEvents: 'none' 
          }}
        >
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

      {/* Hidden master audio element */}
      <audio
        ref={sync.setOriginalRef}
        src={getAudioUrl(jobId)}
        preload="metadata"
      />
    </div>
  );
}

// ─── TrackRow ─────────────────────────────────────────────────────────────────

interface TrackRowProps {
  label:         string;
  color:         string;
  progress:      number;
  currentTime:   number;
  duration:      number;
  volOn:         boolean;
  onToggleVol:   () => void;
  volumeLevel:   number;
  onChangeLevel: (val: number) => void;
  disabled?:     boolean;
}

function TrackRow({
  label, color, progress, currentTime, duration,
  volOn, onToggleVol, volumeLevel, onChangeLevel, disabled,
}: TrackRowProps) {
  const BLOCKS = 16;
  const filled = Math.round((progress / 100) * BLOCKS);

  return (
    <div
      className={`track-row${disabled ? ' disabled' : ''}`}
      style={{ '--track-color': color } as React.CSSProperties}
    >
      <div className="track-label" style={{ color }}>[{label}]</div>
      <div className="track-blocks">
        {Array.from({ length: BLOCKS }).map((_, i) => (
          <div
            key={i}
            className={`track-block${i < filled ? ' filled' : ''}`}
            style={{
              background:  i < filled ? color : 'transparent',
              borderColor: color + '55',
            }}
          />
        ))}
      </div>
      <span className="track-time">{formatTime(currentTime)} / {formatTime(duration)}</span>
      <div className="vol-controls" style={{ position: 'relative' }}>
        {label === 'MELODAI' && (
          <div
            className="caveat-note"
            style={{ 
              top: '-15px', 
              right: '0', 
              fontWeight: 400, 
              fontSize: '14px', 
              color: '#FFE500', 
              opacity: 0.45, 
              transform: 'rotate(1deg)',
              pointerEvents: 'none'
            }}
          >
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
          onChange={e => onChangeLevel(parseFloat(e.target.value))}
          disabled={disabled || !volOn}
          style={{ '--fill-pct': `${volumeLevel * 100}%`, borderColor: color } as React.CSSProperties}
        />
      </div>
    </div>
  );
}
