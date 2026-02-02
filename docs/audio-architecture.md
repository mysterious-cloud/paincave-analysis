# Pain Cave: Audio Architecture

> **Update (2026-01-31):** The architecture is evolving toward **pre-mixed coached variants** instead of separate two-track sync. See [Studio Design](../plans/2026-01-31-studio-design.md) for the full vision. The pre-mixed approach bakes coaching + ducking into single audio files at build time, giving better audio quality and a simpler player. The two-track approach documented below remains as reference but is being superseded.

## Overview

Pain Cave uses a **two-track audio architecture** for workout playback:

1. **Music track** — The full workout music (45 minutes, ~40MB)
2. **Coaching track** — Pre-baked audio with cues placed at correct timestamps (45 minutes of mostly silence + voice cues, ~2-5MB)

Both tracks start simultaneously and stay in sync. Users can toggle coaching on/off by muting/unmitting the coaching track.

---

## Why Two Parallel Tracks (Not Runtime Cue Triggering)

| Approach | Complexity | Sync | Fault Tolerance |
|----------|------------|------|-----------------|
| **Runtime cue triggering** | Timer + fetch + trigger logic | You manage it | Any cue fetch can fail mid-workout |
| **Pre-baked coaching track** | Play two streams | Automatic | Standard streaming, no mid-workout fetches |

**Pre-baked wins because:**
- Simpler player logic
- Automatic sync (same start time = same playhead)
- Seek/scrub just works (both tracks seek together)
- Toggle coaching = mute/unmute (no state management)
- No network requests during workout
- Standard audio streaming, battle-tested

---

## File Structure

```
Per workout in R2:
├── workout_123_music.mp3      (~40MB for 45 min)
├── workout_123_coaching.mp3   (~2-5MB, mostly silence)
└── workout_123.json           (metadata)
```

### Workout JSON

```json
{
  "id": "workout_123",
  "title": "Thursday Thunder",
  "duration_seconds": 2700,
  "music_url": "https://r2.paincave.app/workouts/workout_123_music.mp3",
  "coaching_url": "https://r2.paincave.app/workouts/workout_123_coaching.mp3",
  "segments": [
    {
      "title": "Warm Up",
      "start_time": 0,
      "duration_seconds": 300,
      "intensity": "low",
      "bpm": 120
    },
    {
      "title": "Build",
      "start_time": 300,
      "duration_seconds": 240,
      "intensity": "medium",
      "bpm": 128
    }
  ]
}
```

Note: Individual cue timestamps are baked into the coaching track. The `segments` array is for UI display (progress bar, "what's coming next"), not for audio triggering.

---

## Generating the Coaching Track

### Input: Cue Definitions

```json
{
  "cues": [
    { "time_ms": 30000, "text": "Light resistance. Find your rhythm." },
    { "time_ms": 120000, "text": "Add two turns. Start to build." },
    { "time_ms": 300000, "text": "Heavy resistance. Out of the saddle." },
    { "time_ms": 450000, "text": "Hold this effort. You've got this." }
  ]
}
```

### Step 1: Generate Voice Clips (TTS)

**Option A: Qwen3-TTS (Local, Free)**

```python
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="mps",  # Apple Silicon
    dtype=torch.float16,
)

# Reference audio for consistent voice
ref_audio = "coach_voice_sample.wav"  # 3-5 second sample
ref_text = "The text spoken in the sample"

def generate_cue(text: str, output_path: str):
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    sf.write(output_path, wavs[0], sr)

# Generate all cues
cues = [
    { "time_ms": 30000, "text": "Light resistance. Find your rhythm." },
    { "time_ms": 120000, "text": "Add two turns. Start to build." },
    # ...
]

for i, cue in enumerate(cues):
    generate_cue(cue["text"], f"cue_{i:03d}.wav")
```

**Option B: Fish Audio (Cloud API)**

```python
import requests

def generate_cue_fish(text: str, output_path: str):
    response = requests.post(
        "https://api.fish.audio/v1/tts",
        headers={"Authorization": f"Bearer {FISH_API_KEY}"},
        json={
            "text": text,
            "voice_id": "your_coach_voice_id",
            # Fish Audio emotional tags if needed
        }
    )
    with open(output_path, "wb") as f:
        f.write(response.content)
```

### Step 2: Assemble Coaching Track

```python
from pydub import AudioSegment
import json

def build_coaching_track(
    cue_definitions: list,
    cue_files_dir: str,
    total_duration_ms: int,
    output_path: str
):
    """
    Build a single coaching track with cues placed at correct timestamps.
    
    Args:
        cue_definitions: List of {"time_ms": int, "file": str}
        cue_files_dir: Directory containing generated cue wav files
        total_duration_ms: Total workout duration in milliseconds
        output_path: Where to save the final coaching track
    """
    # Create silent track of full workout duration
    coaching_track = AudioSegment.silent(duration=total_duration_ms)
    
    # Overlay each cue at its timestamp
    for cue in cue_definitions:
        cue_audio = AudioSegment.from_wav(f"{cue_files_dir}/{cue['file']}")
        coaching_track = coaching_track.overlay(cue_audio, position=cue["time_ms"])
    
    # Export as MP3 (silence compresses well)
    coaching_track.export(output_path, format="mp3", bitrate="128k")
    
    return output_path

# Example usage
cues = [
    {"time_ms": 30000, "file": "cue_000.wav"},
    {"time_ms": 120000, "file": "cue_001.wav"},
    {"time_ms": 300000, "file": "cue_002.wav"},
]

build_coaching_track(
    cue_definitions=cues,
    cue_files_dir="./generated_cues",
    total_duration_ms=45 * 60 * 1000,  # 45 minutes
    output_path="workout_123_coaching.mp3"
)
```

### Complete Pipeline Script

```python
#!/usr/bin/env python3
"""
Generate coaching track for a Pain Cave workout.

Usage:
    python generate_coaching.py workout_definition.json
"""

import json
import sys
from pathlib import Path
from pydub import AudioSegment

# Choose your TTS provider
USE_QWEN = True  # Set False to use Fish Audio

if USE_QWEN:
    from qwen_tts import Qwen3TTSModel
    import torch
    import soundfile as sf
    
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map="mps",
        dtype=torch.float16,
    )
    REF_AUDIO = "coach_voice_sample.wav"
    REF_TEXT = "Light resistance. Find your rhythm."
    
    def generate_cue(text: str, output_path: str):
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
        )
        sf.write(output_path, wavs[0], sr)
else:
    import requests
    FISH_API_KEY = "your_api_key"
    FISH_VOICE_ID = "your_voice_id"
    
    def generate_cue(text: str, output_path: str):
        response = requests.post(
            "https://api.fish.audio/v1/tts",
            headers={"Authorization": f"Bearer {FISH_API_KEY}"},
            json={"text": text, "voice_id": FISH_VOICE_ID}
        )
        with open(output_path, "wb") as f:
            f.write(response.content)


def main(workout_json_path: str):
    # Load workout definition
    with open(workout_json_path) as f:
        workout = json.load(f)
    
    workout_id = workout["id"]
    duration_ms = workout["duration_seconds"] * 1000
    cues = workout["cues"]
    
    # Create temp directory for cue files
    temp_dir = Path(f"./temp_{workout_id}")
    temp_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate all voice cues
    print(f"Generating {len(cues)} voice cues...")
    cue_files = []
    for i, cue in enumerate(cues):
        cue_path = temp_dir / f"cue_{i:03d}.wav"
        print(f"  [{i+1}/{len(cues)}] {cue['text'][:40]}...")
        generate_cue(cue["text"], str(cue_path))
        cue_files.append({
            "time_ms": cue["time_ms"],
            "file": str(cue_path)
        })
    
    # Step 2: Assemble coaching track
    print("Assembling coaching track...")
    coaching_track = AudioSegment.silent(duration=duration_ms)
    
    for cue in cue_files:
        cue_audio = AudioSegment.from_wav(cue["file"])
        coaching_track = coaching_track.overlay(cue_audio, position=cue["time_ms"])
    
    # Step 3: Export
    output_path = f"{workout_id}_coaching.mp3"
    coaching_track.export(output_path, format="mp3", bitrate="128k")
    print(f"Done! Saved to {output_path}")
    
    # Cleanup temp files
    for f in temp_dir.glob("*.wav"):
        f.unlink()
    temp_dir.rmdir()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_coaching.py workout_definition.json")
        sys.exit(1)
    main(sys.argv[1])
```

### Example Workout Definition Input

```json
{
  "id": "workout_123",
  "duration_seconds": 2700,
  "cues": [
    { "time_ms": 5000, "text": "Welcome to Pain Cave. Let's warm up." },
    { "time_ms": 30000, "text": "Light resistance. Find your rhythm." },
    { "time_ms": 120000, "text": "Add two turns. Start to build." },
    { "time_ms": 300000, "text": "Heavy resistance. Out of the saddle. Let's climb." },
    { "time_ms": 450000, "text": "Hold this effort. Breathe." },
    { "time_ms": 600000, "text": "Back in the saddle. Recovery spin." }
  ]
}
```

---

## Player Implementation

### Web (React)

```typescript
// hooks/useWorkoutPlayer.ts
import { useRef, useState, useEffect } from 'react'

interface WorkoutPlayerOptions {
  musicUrl: string
  coachingUrl: string
  onTimeUpdate?: (currentTime: number) => void
}

export function useWorkoutPlayer({ musicUrl, coachingUrl, onTimeUpdate }: WorkoutPlayerOptions) {
  const musicRef = useRef<HTMLAudioElement | null>(null)
  const coachingRef = useRef<HTMLAudioElement | null>(null)
  
  const [isPlaying, setIsPlaying] = useState(false)
  const [coachingEnabled, setCoachingEnabled] = useState(true)
  const [currentTime, setCurrentTime] = useState(0)
  
  useEffect(() => {
    musicRef.current = new Audio(musicUrl)
    coachingRef.current = new Audio(coachingUrl)
    
    // Sync time updates
    musicRef.current.ontimeupdate = () => {
      const time = musicRef.current?.currentTime ?? 0
      setCurrentTime(time)
      onTimeUpdate?.(time)
    }
    
    return () => {
      musicRef.current?.pause()
      coachingRef.current?.pause()
    }
  }, [musicUrl, coachingUrl])
  
  const play = async () => {
    if (!musicRef.current || !coachingRef.current) return
    
    // Start both at same time
    await Promise.all([
      musicRef.current.play(),
      coachingRef.current.play()
    ])
    setIsPlaying(true)
  }
  
  const pause = () => {
    musicRef.current?.pause()
    coachingRef.current?.pause()
    setIsPlaying(false)
  }
  
  const seek = (time: number) => {
    if (musicRef.current) musicRef.current.currentTime = time
    if (coachingRef.current) coachingRef.current.currentTime = time
  }
  
  const toggleCoaching = () => {
    if (coachingRef.current) {
      coachingRef.current.muted = coachingEnabled
      setCoachingEnabled(!coachingEnabled)
    }
  }
  
  return {
    isPlaying,
    currentTime,
    coachingEnabled,
    play,
    pause,
    seek,
    toggleCoaching,
  }
}
```

### iOS (Swift)

```swift
import AVFoundation

class WorkoutAudioManager: ObservableObject {
    private var musicPlayer: AVAudioPlayer?
    private var coachingPlayer: AVAudioPlayer?
    
    @Published var isPlaying = false
    @Published var coachingEnabled = true
    @Published var currentTime: TimeInterval = 0
    
    private var timer: Timer?
    
    func load(musicURL: URL, coachingURL: URL) async throws {
        let musicData = try await URLSession.shared.data(from: musicURL).0
        let coachingData = try await URLSession.shared.data(from: coachingURL).0
        
        musicPlayer = try AVAudioPlayer(data: musicData)
        coachingPlayer = try AVAudioPlayer(data: coachingData)
        
        musicPlayer?.prepareToPlay()
        coachingPlayer?.prepareToPlay()
    }
    
    func play() {
        // Start both simultaneously
        let startTime = musicPlayer!.deviceCurrentTime + 0.1
        musicPlayer?.play(atTime: startTime)
        coachingPlayer?.play(atTime: startTime)
        
        isPlaying = true
        startTimer()
    }
    
    func pause() {
        musicPlayer?.pause()
        coachingPlayer?.pause()
        isPlaying = false
        stopTimer()
    }
    
    func seek(to time: TimeInterval) {
        musicPlayer?.currentTime = time
        coachingPlayer?.currentTime = time
    }
    
    func toggleCoaching() {
        coachingEnabled.toggle()
        coachingPlayer?.volume = coachingEnabled ? 1.0 : 0.0
    }
    
    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            self.currentTime = self.musicPlayer?.currentTime ?? 0
        }
    }
    
    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}
```

---

## Ducking (Optional Enhancement)

If you want music to duck (lower volume) when coaching is playing, you have two options:

### Option A: Bake Ducking into Music Track

When assembling the workout, analyze the coaching track for non-silent sections and reduce music volume at those timestamps. More complex, but zero runtime logic.

### Option B: Runtime Ducking via Audio Analysis

Detect when coaching audio is non-silent and duck music in real-time. Works but adds complexity.

### Option C: Skip Ducking

Coaching cues are typically short and punchy. They cut through music fine without ducking. Many spin classes work this way. **Start here.**

---

## Storage Estimates

| Component | Size (45 min workout) |
|-----------|----------------------|
| Music track | ~40MB (320kbps) |
| Coaching track | ~2-5MB (mostly silence, compresses well) |
| **Total per workout** | ~45MB |

For 100 workouts at launch: ~4.5GB in R2. Negligible cost.

---

## TTS Options Comparison

| Feature | Qwen3-TTS (Local) | Fish Audio (Cloud) |
|---------|-------------------|-------------------|
| Cost | Free | ~$15-20/month subscription |
| Quality | Very good (new, improving) | Good, proven |
| Setup | Requires GPU, some tinkering | API call |
| Emotional control | Natural language prompts | Emotional tags |
| Voice cloning | 3 seconds of audio | Yes |
| Hardware needed | M1 Max works, 0.6B model fits easily | None |
| Commercial use | Apache 2.0 (yes) | Yes with subscription |

**Recommendation:** Try Qwen3-TTS locally on your M1 Max. If quality is good enough, use it. Keep Fish Audio as fallback.

---

## Summary

1. **Generate voice cues** via TTS (Qwen3-TTS or Fish Audio)
2. **Assemble coaching track** — silence + cues at correct timestamps
3. **Store both tracks** in R2 per workout
4. **Player starts both simultaneously** — sync is automatic
5. **Toggle coaching** = mute/unmute coaching track
6. **Seek** = seek both tracks to same position

Simple, fault-tolerant, no runtime cue triggering logic needed.