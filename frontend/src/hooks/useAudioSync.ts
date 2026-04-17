import { useRef, useCallback, useState } from 'react';

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
}

export function useAudioSync(): AudioSyncControls {
  const originalRef = useRef<HTMLAudioElement | null>(null);
  const midiGainRef = useRef<GainNode | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [originalVolume, setOriginalVolume] = useState(true); // mute toggle
  const [midiVolume, setMidiVolume] = useState(true);         // mute toggle
  const [originalLevel, setOriginalLevel] = useState(0.8);
  const [midiLevel, setMidiLevel] = useState(0.8);
  const driftCheckRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const toneStartTimeRef = useRef(0);
  const toneOffsetRef = useRef(0);

  const setOriginalRef = useCallback((el: HTMLAudioElement | null) => {
    originalRef.current = el;
    if (el) {
      el.ontimeupdate = () => setCurrentTime(el.currentTime);
      el.onloadedmetadata = () => setDuration(el.duration);
    }
  }, []);



  const togglePlay = useCallback(() => {
    const audio = originalRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      if (driftCheckRef.current) clearInterval(driftCheckRef.current);
      setIsPlaying(false);
    } else {
      audio.play().catch(console.error);
      toneStartTimeRef.current = Date.now() / 1000;
      toneOffsetRef.current = audio.currentTime;
      setIsPlaying(true);
    }
  }, [isPlaying]);

  const seek = useCallback((time: number) => {
    const audio = originalRef.current;
    if (audio) {
      audio.currentTime = time;
      setCurrentTime(time);
      toneOffsetRef.current = time;
      toneStartTimeRef.current = Date.now() / 1000;
    }
  }, []);

  const toggleOriginalVolume = useCallback(() => {
    setOriginalVolume(v => {
      const next = !v;
      if (originalRef.current) originalRef.current.volume = next ? originalLevel : 0;
      return next;
    });
  }, [originalLevel]);

  const toggleMidiVolume = useCallback(() => {
    setMidiVolume(v => !v);
  }, []);

  const setOriginalVolumeLevel = useCallback((val: number) => {
    setOriginalLevel(val);
    if (originalRef.current && originalVolume) {
      originalRef.current.volume = val;
    }
  }, [originalVolume]);

  const setMidiVolumeLevel = useCallback((val: number) => {
    setMidiLevel(val);
  }, []);

  return {
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
  };
}
