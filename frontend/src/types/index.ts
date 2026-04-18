export type Phase =
  | 'DEMUCS'
  | 'BASIC_PITCH'
  | 'LIBROSA'
  | 'CNN_CLASSIFIER'
  | 'KEY_DETECT'
  | 'ARRANGEMENT'
  | 'MIDI_GEN'
  | 'OUTPUT'
  | 'DONE'
  | 'ERROR';

export type Instrument = 'Piano' | 'Guitar' | 'Violin' | 'Sitar' | 'Flute' | 'Trumpet' | 'Electric Guitar' | 'Acoustic Electric Guitar';

export type OutputMode = 'midi' | 'sheet';

export interface JobStatus {
  job_id: string;
  phase: Phase;
  progress: number;
  log: string[];
  error: string | null;
}

export interface UploadConfig {
  instrument: Instrument;
  output_mode: OutputMode;
  skip_demucs: boolean;
  threshold: number;
}

export interface AppState {
  view: 'upload' | 'processing' | 'result';
  jobId: string | null;
  jobStatus: JobStatus | null;
  file: File | null;
  config: UploadConfig;
}

export const PHASE_COLORS: Record<Phase, string> = {
  DEMUCS: '#FF2D78',
  BASIC_PITCH: '#00FFE0',
  LIBROSA: '#FFE500',
  CNN_CLASSIFIER: '#39FF14',
  KEY_DETECT: '#FF2020',
  ARRANGEMENT: '#FF2D78',
  MIDI_GEN: '#00FFE0',
  OUTPUT: '#FFE500',
  DONE: '#39FF14',
  ERROR: '#FF2020',
};

export const INSTRUMENT_COLORS: Record<Instrument, string> = {
  Piano: '#FFE500',
  Guitar: '#39FF14',
  Violin: '#FF2D78',
  Sitar: '#00FFE0',
  Flute: '#FF2020',
  Trumpet: '#FF6B00',
  'Electric Guitar': '#AA00FF',
  'Acoustic Electric Guitar': '#FFA000',
};

export const INSTRUMENT_SHADOWS: Record<Instrument, string> = {
  Piano: '#FF2020',
  Guitar: '#FF2D78',
  Violin: '#FFE500',
  Sitar: '#00FFE0',
  Flute: '#39FF14',
  Trumpet: '#FFE500',
  'Electric Guitar': '#FFE500',
  'Acoustic Electric Guitar': '#00FFE0',
};
