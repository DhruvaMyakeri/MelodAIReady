/**
 * useAudioSync.ts
 *
 * Manages the HTML <audio> element as the SINGLE master clock.
 * Exposes an `onSeek` callback so AudioPlayer can reset its
 * lookahead scheduler whenever the user scrubs the timeline.
 */

import { useRef, useCallback, useState, useEffect, useMemo } from 'react';

export interface AudioSyncControls {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  originalVolume: boolean;
  midiVolume: boolean;
  togglePlay: () => void;
  seek: (time: number) => void;
  originalLevel: number;
  midiLevel: number;
  toggleOriginalVolume: () => void;
  toggleMidiVolume: () => void;
  setOriginalVolumeLevel: (val: number) => void;
  setMidiVolumeLevel: (val: number) => void;
  setOriginalRef: (el: HTMLAudioElement | null) => void;
  registerSeekCallback: (cb: (time: number) => void) => void;
}

export function useAudioSync(): AudioSyncControls {
  const originalRef    = useRef<HTMLAudioElement | null>(null);
  const seekCallbacks  = useRef<Array<(t: number) => void>>([]);

  const [isPlaying,      setIsPlaying]      = useState(false);
  const [currentTime,    setCurrentTime]    = useState(0);
  const [duration,       setDuration]       = useState(0);
  const [originalVolume, setOriginalVolume] = useState(true);
  const [midiVolume,     setMidiVolume]     = useState(true);
  const [originalLevel,  setOriginalLevel]  = useState(0.8);
  const [midiLevel,      setMidiLevel]      = useState(0.8);

  // ── MASTER VOLUME SYNC (Declarative) ─────────────────────────────────────
  // This ensures the <audio> element volume ALWAYS matches state,
  // preventing stale closure issues in callbacks.
  useEffect(() => {
    if (originalRef.current) {
      originalRef.current.volume = originalVolume ? originalLevel : 0;
    }
  }, [originalVolume, originalLevel]);

  // ── MASTER REF SETUP ──────────────────────────────────────────────────────
  const setOriginalRef = useCallback((el: HTMLAudioElement | null) => {
    if (originalRef.current === el) return;
    
    // Cleanup old ref if it's changing
    if (originalRef.current) {
      originalRef.current.removeEventListener('timeupdate', () => {});
      originalRef.current.removeEventListener('loadedmetadata', () => {});
    }

    originalRef.current = el;
    if (!el) return;

    // Use event listeners instead of properties for multi-ref stability
    const onTime = () => setCurrentTime(el.currentTime);
    const onMeta = () => setDuration(el.duration);
    
    el.addEventListener('timeupdate', onTime);
    el.addEventListener('loadedmetadata', onMeta);
    
    // Initial sync
    el.volume = originalVolume ? originalLevel : 0;
    setCurrentTime(el.currentTime);
    if (el.duration) setDuration(el.duration);
    
    return () => {
      el.removeEventListener('timeupdate', onTime);
      el.removeEventListener('loadedmetadata', onMeta);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Stable ref setter

  const togglePlay = useCallback(() => {
    const audio = originalRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().catch(console.error);
      setIsPlaying(true);
    }
  }, [isPlaying]);

  const seek = useCallback((time: number) => {
    const audio = originalRef.current;
    if (audio) {
      audio.currentTime = time;
      setCurrentTime(time);
    }
    for (const cb of seekCallbacks.current) {
      cb(time);
    }
  }, []);

  const registerSeekCallback = useCallback((cb: (t: number) => void) => {
    seekCallbacks.current.push(cb);
  }, []);

  const toggleOriginalVolume = useCallback(() => {
    setOriginalVolume(v => !v);
  }, []);

  const toggleMidiVolume = useCallback(() => {
    setMidiVolume(v => !v);
  }, []);

  const setOriginalVolumeLevel = useCallback((val: number) => {
    setOriginalLevel(val);
  }, []);

  const setMidiVolumeLevel = useCallback((val: number) => {
    setMidiLevel(val);
  }, []);

  // ── STABLE RETURN OBJECT ──────────────────────────────────────────────────
  // useMemo prevents AudioPlayer from re-running effects on every render
  // unless a primitive state property actually changed.
  return useMemo(() => ({
    isPlaying,
    currentTime,
    duration,
    originalVolume,
    midiVolume,
    originalLevel,
    midiLevel,
    togglePlay,
    seek,
    toggleOriginalVolume,
    toggleMidiVolume,
    setOriginalVolumeLevel,
    setMidiVolumeLevel,
    setOriginalRef,
    registerSeekCallback,
  }), [
    isPlaying, currentTime, duration, originalVolume, midiVolume, 
    originalLevel, midiLevel, togglePlay, seek, toggleOriginalVolume, 
    toggleMidiVolume, setOriginalVolumeLevel, setMidiVolumeLevel, 
    setOriginalRef, registerSeekCallback
  ]);
}
