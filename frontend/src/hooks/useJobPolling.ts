import { useEffect, useRef, useState, useCallback } from 'react';
import { getStatus } from '../api/client';
import type { JobStatus } from '../types';

const POLL_INTERVAL_MS = 1500;

export function useJobPolling(jobId: string | null) {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeRef = useRef(false);

  const stop = useCallback(() => {
    activeRef.current = false;
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const poll = useCallback(async (id: string) => {
    if (!activeRef.current) return;
    try {
      const s = await getStatus(id);
      setStatus(s);
      if (s.phase === 'DONE' || s.phase === 'ERROR') {
        stop();
        return;
      }
    } catch (e) {
      setError(String(e));
    }
    if (activeRef.current) {
      timerRef.current = setTimeout(() => poll(id), POLL_INTERVAL_MS);
    }
  }, [stop]);

  useEffect(() => {
    if (!jobId) return;
    activeRef.current = true;
    setStatus(null);
    setError(null);
    poll(jobId);
    return stop;
  }, [jobId, poll, stop]);

  return { status, error, stop };
}
