import pygame
import time
import argparse

def play_audio(midi_file, backing_file=None, vol=0.2):
    # Initialize Pygame mixer for audio playback
    pygame.mixer.init()
    
    try:
        print(f"[INFO] Loading MIDI: {midi_file}...")
        pygame.mixer.music.load(midi_file)
        
        backing_sound = None
        if backing_file:
            print(f"[INFO] Loading original audio overlay: {backing_file} (Volume: {vol})")
            backing_sound = pygame.mixer.Sound(backing_file)
            backing_sound.set_volume(vol)
        
        print("[INFO] Playing Audio! (Press Ctrl+C to stop)")
        if backing_sound:
            backing_sound.play()
            
        pygame.mixer.music.play()
        
        # Keep the script running while the audio is playing
        while pygame.mixer.music.get_busy() or (backing_sound and backing_sound.get_num_channels() > 0):
            time.sleep(0.5)
            
    except Exception as e:
        print(f"[ERROR] Could not play audio: {e}")
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_piano.mid", help="MIDI file to play")
    parser.add_argument("--backing", default=None, help="Original Audio MP3/WAV to play underneath")
    parser.add_argument("--vol", type=float, default=0.2, help="Volume of the backing track (0.0 - 1.0)")
    args = parser.parse_args()
    
    play_audio(args.input, args.backing, args.vol)
