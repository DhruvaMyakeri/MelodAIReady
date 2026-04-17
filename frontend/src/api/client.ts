import type { JobStatus, UploadConfig } from '../types';

const BASE = '';

export async function submitJob(file: File, config: UploadConfig): Promise<string> {
  const form = new FormData();
  form.append('file', file);
  form.append('instrument', config.instrument);
  form.append('output_mode', config.output_mode);
  form.append('skip_demucs', String(config.skip_demucs));
  form.append('threshold', String(config.threshold));

  const res = await fetch(`${BASE}/run`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Submit failed: ${err}`);
  }
  const data = await res.json();
  return data.job_id as string;
}

export async function getStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/status/${jobId}`);
  if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`);
  return res.json();
}

export function getDownloadUrl(jobId: string): string {
  return `${BASE}/download/${jobId}`;
}

export function getAudioUrl(jobId: string): string {
  return `${BASE}/audio/${jobId}`;
}
