#!/usr/bin/env python3
"""
Extract song boundaries from Ableton Live Set (.als) files.

ALS files are gzipped XML. This extracts locators and converts
beat positions to seconds using the project tempo.
"""

import argparse
import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_locators(als_path: Path) -> dict:
    """Extract locators and tempo from ALS file."""
    # Decompress and parse XML
    with gzip.open(als_path, 'rt', encoding='utf-8') as f:
        tree = ET.parse(f)

    root = tree.getroot()

    # Find tempo (in LiveSet/MainTrack/DeviceChain/Mixer/Tempo/Manual)
    tempo_elem = root.find('.//Tempo/Manual')
    if tempo_elem is None:
        raise ValueError("Could not find tempo in ALS file")

    tempo = float(tempo_elem.get('Value', 120))
    print(f"Project tempo: {tempo:.2f} BPM")

    # Find locators
    locators_elem = root.find('.//Locators/Locators')
    if locators_elem is None:
        print("No locators found in ALS file")
        return {"tempo": tempo, "locators": []}

    locators = []
    for loc in locators_elem.findall('Locator'):
        time_beats = float(loc.find('Time').get('Value', 0))
        name = loc.find('Name').get('Value', '')

        # Convert beats to seconds
        time_seconds = time_beats * 60 / tempo

        locators.append({
            "beats": time_beats,
            "seconds": round(time_seconds, 1),
            "name": name
        })

    # Sort by time
    locators.sort(key=lambda x: x['seconds'])

    print(f"Found {len(locators)} locators")

    return {"tempo": tempo, "locators": locators}


def generate_songs_json(locators_data: dict, duration: float | None = None) -> dict:
    """Generate songs.json from locator data."""
    locators = locators_data["locators"]

    if not locators:
        return {"songs": []}

    songs = []

    # First song starts at 0
    prev_time = 0
    prev_name = "Track 1"

    for i, loc in enumerate(locators):
        songs.append({
            "start": prev_time,
            "end": loc["seconds"],
            "title": prev_name
        })
        prev_time = loc["seconds"]
        prev_name = loc["name"] if loc["name"] else f"Track {i + 2}"

    # Last song ends at duration (if provided) or estimate from last locator
    if duration:
        end_time = duration
    else:
        # Estimate: last song is ~same length as average
        avg_length = prev_time / len(locators) if locators else 200
        end_time = prev_time + avg_length

    songs.append({
        "start": prev_time,
        "end": round(end_time, 1),
        "title": prev_name if prev_name else f"Track {len(locators) + 1}"
    })

    return {"songs": songs}


def main():
    parser = argparse.ArgumentParser(
        description="Extract song boundaries from Ableton Live Set"
    )
    parser.add_argument("input", type=Path, help="Path to .als file")
    parser.add_argument("--output", "-o", type=Path, help="Output songs.json path")
    parser.add_argument("--duration", "-d", type=float,
                        help="Total duration in seconds (for last song end time)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return

    # Extract locators
    locators_data = extract_locators(args.input)

    # Print locators
    print("\nLocators:")
    print(f"{'Name':<10} {'Beats':>10} {'Seconds':>10} {'Time':>10}")
    print("-" * 45)
    for loc in locators_data["locators"]:
        mins = int(loc["seconds"] // 60)
        secs = int(loc["seconds"] % 60)
        print(f"{loc['name']:<10} {loc['beats']:>10.1f} {loc['seconds']:>10.1f} {mins:>5}:{secs:02d}")

    # Generate songs.json
    songs = generate_songs_json(locators_data, args.duration)

    print(f"\nGenerated {len(songs['songs'])} songs:")
    for song in songs["songs"]:
        start_m, start_s = divmod(int(song["start"]), 60)
        end_m, end_s = divmod(int(song["end"]), 60)
        print(f"  {song['title']}: {start_m}:{start_s:02d} - {end_m}:{end_s:02d}")

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(songs, f, indent=2)
        print(f"\nSaved: {args.output}")
    else:
        print("\n" + json.dumps(songs, indent=2))


if __name__ == "__main__":
    main()
