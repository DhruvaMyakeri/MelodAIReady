import json
import argparse
import numpy as np

def transcribe(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    all_notes = []
    
    # Collect all notes
    for seg in segments:
        for note_event in seg.get('notes', []):
            all_notes.append(note_event)
            
    if not all_notes:
        print("[WARN] No notes found in the JSON.")
        return
        
    duration = max(n['offset'] for n in all_notes)
    
    # ─── Frame-based Melody Extraction (Skyline Algorithm) ───
    # We slice time into 20ms buckets. 
    # For each bucket, the active note with the HIGHEST VELOCITY wins.
    # This completely eliminates the glitchy overlapping noises.
    
    FRAME_SIZE = 0.02 # 20 milliseconds
    num_frames = int(duration / FRAME_SIZE) + 1
    
    # Array storing (note_name, velocity) for each frame
    frame_winners = [(None, 0)] * num_frames
    
    for note in all_notes:
        start_idx = max(0, int(note['onset'] / FRAME_SIZE))
        end_idx = min(num_frames, int(note['offset'] / FRAME_SIZE))
        
        for i in range(start_idx, end_idx):
            current_winner_vel = frame_winners[i][1]
            # Choose the loudest note at this exact moment
            if note['velocity'] > current_winner_vel:
                frame_winners[i] = (note['note'], note['velocity'])
                
    # ─── Reconstruct Note Sequences from Frames ───
    transcription = []
    current_note = None
    on_time = 0.0
    
    for i in range(num_frames):
        note_name = frame_winners[i][0]
        time_sec = i * FRAME_SIZE
        
        # Note changed or stopped
        if note_name != current_note:
            # End the previous note
            if current_note is not None:
                dur = time_sec - on_time
                if dur > 0.05: # Minimum note length to avoid micro-glitches
                    transcription.append({
                        "time": round(on_time, 3),
                        "note": current_note,
                        "duration": round(dur, 3)
                    })
            # Start new note
            current_note = note_name
            on_time = time_sec

    # Save last note
    if current_note is not None:
        dur = (num_frames * FRAME_SIZE) - on_time
        if dur > 0.05:
            transcription.append({
                "time": round(on_time, 3),
                "note": current_note,
                "duration": round(dur, 3)
            })

    with open(output_file, 'w') as f:
        json.dump(transcription, f, indent=2)
        
    print(f"[DONE] Algorithmic transcription saved to {output_file}")
    print(f"       Extracted {len(transcription)} clean melodic notes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="llm_input.json")
    parser.add_argument("--output", default="transcription.json")
    args = parser.parse_args()
    
    transcribe(args.input, args.output)
