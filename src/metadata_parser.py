"""
Parse markdown metadata files for song import.

Expected format:
    # Song Title

    ## SUNO
    - Generated: 2024-01-15
    - URL: https://suno.ai/song/abc123
    - Styles: EDM, House, Techno, -Slow, -Acoustic
    - Weirdness: 0.3
    - Influence: 0.5
    - Vocals: Instrumental

    ## Tags
    high-energy, cycling, cardio

    ## Lyrics
    (lyrics content here)

Note: In SUNO, styles prefixed with "-" are anti-styles (e.g., "-Slow" means exclude slow).
These are kept together in the styles field.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class SunoMetadata:
    generated: date | None = None
    url: str | None = None
    styles: list[str] = field(default_factory=list)  # Includes -prefixed anti-styles
    weirdness: float | None = None
    influence: float | None = None
    vocals: str | None = None


@dataclass
class SongMetadata:
    title: str | None = None
    suno: SunoMetadata = field(default_factory=SunoMetadata)
    tags: list[str] = field(default_factory=list)
    lyrics: str | None = None


def parse_date(value: str) -> date | None:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def parse_float(value: str) -> float | None:
    """Parse a float value."""
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_list(value: str) -> list[str]:
    """Parse comma-separated or newline-separated values."""
    # Split by comma or newline, strip whitespace, filter empty
    items = re.split(r'[,\n]', value)
    return [item.strip() for item in items if item.strip()]


def parse_suno_section(lines: list[str]) -> SunoMetadata:
    """Parse the SUNO section into metadata."""
    suno = SunoMetadata()

    for line in lines:
        line = line.strip()
        if not line.startswith('-'):
            continue

        # Remove leading dash and whitespace
        content = line[1:].strip()

        # Split on first colon
        if ':' not in content:
            continue

        key, value = content.split(':', 1)
        key = key.strip().lower()
        value = value.strip()

        if key == 'generated':
            suno.generated = parse_date(value)
        elif key == 'url':
            suno.url = value if value else None
        elif key == 'styles':
            suno.styles = parse_list(value)
        elif key == 'weirdness':
            suno.weirdness = parse_float(value)
        elif key == 'influence':
            suno.influence = parse_float(value)
        elif key == 'vocals':
            suno.vocals = value if value else None

    return suno


def parse_metadata(content: str) -> SongMetadata:
    """Parse markdown content into SongMetadata."""
    metadata = SongMetadata()
    lines = content.split('\n')

    # Find title (first # header)
    for line in lines:
        if line.startswith('# ') and not line.startswith('## '):
            metadata.title = line[2:].strip()
            break

    # Find sections
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith('## '):
            # Save previous section
            if current_section:
                sections[current_section] = current_lines

            current_section = line[3:].strip().lower()
            current_lines = []
        elif current_section:
            current_lines.append(line)

    # Save last section
    if current_section:
        sections[current_section] = current_lines

    # Parse SUNO section
    if 'suno' in sections:
        metadata.suno = parse_suno_section(sections['suno'])

    # Parse tags section
    if 'tags' in sections:
        tags_content = '\n'.join(sections['tags'])
        metadata.tags = parse_list(tags_content)

    # Parse lyrics section (everything after ## Lyrics until next section or EOF)
    if 'lyrics' in sections:
        lyrics_content = '\n'.join(sections['lyrics']).strip()
        metadata.lyrics = lyrics_content if lyrics_content else None

    return metadata


def parse_metadata_file(path: Path) -> SongMetadata:
    """Parse a markdown metadata file."""
    content = path.read_text(encoding='utf-8')
    return parse_metadata(content)
