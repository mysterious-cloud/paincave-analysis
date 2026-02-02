"""
Generate markdown metadata files from song data.

Output format matches metadata_parser.py expectations.
"""

from pathlib import Path

from metadata_parser import SongMetadata, SunoMetadata


def format_metadata(metadata: SongMetadata) -> str:
    """Format SongMetadata as markdown."""
    lines: list[str] = []

    # Title
    if metadata.title:
        lines.append(f"# {metadata.title}")
        lines.append("")

    # SUNO section
    suno = metadata.suno
    if _has_suno_data(suno):
        lines.append("## SUNO")
        if suno.generated:
            lines.append(f"- Generated: {suno.generated.isoformat()}")
        if suno.url:
            lines.append(f"- URL: {suno.url}")
        if suno.styles:
            lines.append(f"- Styles: {', '.join(suno.styles)}")
        if suno.weirdness is not None:
            lines.append(f"- Weirdness: {suno.weirdness}")
        if suno.influence is not None:
            lines.append(f"- Influence: {suno.influence}")
        if suno.vocals:
            lines.append(f"- Vocals: {suno.vocals}")
        lines.append("")

    # Tags section
    if metadata.tags:
        lines.append("## Tags")
        lines.append(", ".join(metadata.tags))
        lines.append("")

    # Lyrics section
    if metadata.lyrics:
        lines.append("## Lyrics")
        lines.append(metadata.lyrics)
        lines.append("")

    return "\n".join(lines)


def _has_suno_data(suno: SunoMetadata) -> bool:
    """Check if SUNO metadata has any data."""
    return any([
        suno.generated,
        suno.url,
        suno.styles,
        suno.weirdness is not None,
        suno.influence is not None,
        suno.vocals,
    ])


def write_metadata_file(metadata: SongMetadata, path: Path) -> None:
    """Write SongMetadata to a markdown file."""
    content = format_metadata(metadata)
    path.write_text(content, encoding='utf-8')
