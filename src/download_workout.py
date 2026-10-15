#!/usr/bin/env python3
"""Download LES MILLS workout audio and captions using a signed-in Chrome profile."""

import argparse
import base64
import concurrent.futures
import datetime
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen


class DownloadError(Exception):
    pass


def attributes(line):
    return {m[0]: m[1] or m[2]
            for m in re.findall(r'([\w-]+)=(?:"([^"]*)"|([^,]*))', line)}


def parse_master(text, base):
    lines = text.splitlines()
    variants, subtitles = [], []
    for i, line in enumerate(lines):
        if line.startswith('#EXT-X-MEDIA:'):
            a = attributes(line)
            if a.get('TYPE') == 'SUBTITLES' and a.get('URI'):
                subtitles.append({**a, 'url': urljoin(base, a['URI'])})
        if line.startswith('#EXT-X-STREAM-INF:'):
            a = attributes(line)
            uri = next((s.strip() for s in lines[i + 1:] if s.strip() and not s.startswith('#')), None)
            if uri:
                variants.append((int(a.get('BANDWIDTH', 0)), urljoin(base, uri), a))
    if not variants:
        raise DownloadError('No HLS media variants found in the captured playlist.')
    # Prefer a separate audio rendition, otherwise use the smallest muxed stream.
    _, video_url, selected = min(variants, key=lambda v: v[0])
    for line in lines:
        if line.startswith('#EXT-X-MEDIA:'):
            a = attributes(line)
            if (a.get('TYPE') == 'AUDIO' and a.get('URI')
                    and a.get('GROUP-ID') == selected.get('AUDIO')):
                return urljoin(base, a['URI']), subtitles
    return video_url, subtitles


def fetch_text(url) -> str:
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=45) as r:
                return r.read().decode('utf-8-sig')
        except Exception:
            if attempt < 2:
                time.sleep(attempt + 1)
    raise DownloadError('A media request failed after three attempts. Run again to get fresh playback URLs.')


def capture_master(url, profile, timeout):
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        raise DownloadError('Playwright is missing. Run: python -m pip install -r requirements.txt') from None
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    captured = []
    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                str(profile), channel='chrome', headless=False,
                args=['--mute-audio'], chromium_sandbox=True)
        except Exception:
            raise DownloadError('Could not open Chrome. Install Google Chrome and close any '
                                'other downloader window using this profile, then retry.') from None
        try:
            page = context.pages[0] if context.pages else context.new_page()

            def inspect_response(response):
                if captured or '.m3u8' not in urlparse(response.url).path or response.status != 200:
                    return
                try:
                    body = response.text()
                    if '#EXT-X-STREAM-INF:' in body:
                        captured.append((response.url, body))
                except Exception:
                    pass  # A cancelled request may be followed by a successful retry.

            context.on('response', inspect_response)
            print('Chrome is opening. Sign in if prompted; click Play if needed.', flush=True)
            deadline = time.monotonic() + timeout
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=min(timeout * 1000, 60000))
            except PlaywrightTimeout:
                pass  # Authentication or slow page resources can still finish in the loop.
            clicked = False
            while not captured and time.monotonic() < deadline:
                if page.is_closed():
                    raise DownloadError('The workout window was closed before media was found.')
                if not clicked:
                    play = page.get_by_role('button', name=re.compile(r'^play$', re.I)).first
                    try:
                        if play.is_visible():
                            play.click(timeout=1000)
                            clicked = True
                    except PlaywrightTimeout:
                        pass
                page.wait_for_timeout(500)
            if not captured:
                raise DownloadError('No workout playlist appeared. Check your login/subscription '
                                    'and play the workout. Increase --timeout if login takes longer.')
            print('Found workout media.', flush=True)
            return captured[0]
        finally:
            context.close()


def master_from_har(path):
    candidates = []
    for entry in json.loads(path.read_text())['log']['entries']:
        url = entry['request']['url']
        if '.m3u8' not in urlparse(url).path:
            continue
        c = entry['response']['content']
        text = c.get('text', '')
        if c.get('encoding') == 'base64':
            text = base64.b64decode(text).decode('utf-8-sig')
        if '#EXT-X-STREAM-INF:' in text:
            candidates.append((url, text))
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise DownloadError('HAR must contain exactly one workout master playlist, with response bodies.')
    return unique[0]


def timestamp_ms(value):
    parts = value.split(':')
    if len(parts) == 2:
        parts.insert(0, '0')
    hours, minutes, seconds = parts
    whole, fraction = seconds.split('.')
    return ((int(hours) * 3600 + int(minutes) * 60 + int(whole)) * 1000
            + int(fraction.ljust(3, '0')))


def format_time(ms, separator='.'):
    seconds, fraction = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}{separator}{fraction:03}'


def merge_captions(parts):
    cues = set()
    clock_offsets = set()
    for text in parts:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Mux's segments carry absolute cue times and a common MPEGTS origin.
        # Refuse other timestamp schemes rather than silently producing shifted captions.
        mapping = re.search(r'X-TIMESTAMP-MAP=LOCAL:([^,\s]+),MPEGTS:(\d+)', text)
        if mapping:
            clock_offsets.add(int(mapping[2]) / 90 - timestamp_ms(mapping[1]))
        for block in re.split(r'\n\s*\n', text.strip()):
            lines = block.splitlines()
            for i, line in enumerate(lines):
                match = re.fullmatch(r'([\d:.]+)\s+-->\s+([\d:.]+)(.*)', line)
                if match:
                    start, end = timestamp_ms(match[1]), timestamp_ms(match[2])
                    if end <= start:
                        raise DownloadError('Found an invalid caption time range.')
                    cues.add((start, end, match[3], '\n'.join(lines[i + 1:])))
                    break
    if len(clock_offsets) > 1:
        raise DownloadError('These captions use changing HLS timestamp offsets; merging is unsupported.')
    if not cues:
        raise DownloadError('The subtitle track contained no caption cues.')
    return sorted(cues)


def save_captions(track, folder, stem):
    playlist = fetch_text(track['url'])
    if not playlist.lstrip().startswith('#EXTM3U'):
        raise DownloadError('Expected an HLS subtitle playlist.')
    urls = [urljoin(track['url'], line.strip()) for line in playlist.splitlines()
            if line.strip() and not line.startswith('#')]
    if not urls:
        raise DownloadError('The subtitle playlist was empty.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(fetch_text, urls))
    cues = merge_captions(parts)
    raw = folder / 'subtitle-segments'
    raw.mkdir()
    for i, text in enumerate(parts):
        (raw / f'{i:03}.vtt').write_text(text, encoding='utf-8')
    vtt, srt, plain = ['WEBVTT\n'], [], []
    for i, (start, end, settings, text) in enumerate(cues, 1):
        vtt.append(f'{format_time(start)} --> {format_time(end)}{settings}\n{text}\n')
        # Remove WebVTT positioning/voice markup for readable SRT and text.
        readable = html.unescape(re.sub(r'<[^>]*>', '', text))
        srt.append(f'{i}\n{format_time(start, ",")} --> {format_time(end, ",")}\n{readable}\n')
        plain.append(readable)
    (folder / f'{stem}.vtt').write_text('\n'.join(vtt), encoding='utf-8')
    (folder / f'{stem}.srt').write_text('\n'.join(srt), encoding='utf-8')
    (folder / f'{stem}.txt').write_text('\n'.join(plain) + '\n', encoding='utf-8')
    print(f'Saved {len(cues)} caption cues from {len(parts)} segments.', flush=True)
    return {'language': track.get('LANGUAGE'), 'name': track.get('NAME'),
            'cues': len(cues), 'segments': len(parts), 'final_cue_ms': cues[-1][1]}


def run_media(command, description):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        # Do not expose signed playback URLs in terminal errors.
        detail = re.sub(r'https?://\S+', '[media URL]', result.stderr[-1500:])
        raise DownloadError(f'{description} failed.\n{detail}')
    return result.stdout


def download(url, master_url, master, output, language):
    audio_url, tracks = parse_master(master, master_url)
    matches = [t for t in tracks if t.get('LANGUAGE', '').lower() == language.lower()]
    if not matches:
        available = ', '.join(sorted({t.get('LANGUAGE', '?') for t in tracks})) or 'none'
        raise DownloadError(f'No {language!r} subtitle track. Available languages: {available}')
    rendition = fetch_text(audio_url)
    if re.search(r'METHOD=(?:SAMPLE-AES|SAMPLE-AES-CTR)', rendition) or '#EXT-X-SESSION-KEY:' in master:
        raise DownloadError('This workout uses protected media that this downloader does not support.')
    if '#EXT-X-ENDLIST' not in rendition:
        raise DownloadError('Expected a complete on-demand playlist, not a live stream.')
    name = re.sub(r'[^A-Za-z0-9]+', '_', unquote(urlparse(url).path.rstrip('/').split('/')[-1])).strip('_')
    name = name or 'workout'
    output.mkdir(parents=True, exist_ok=True)
    destination = output / name
    if destination.exists():
        raise DownloadError(f'Output already exists: {destination}\nChoose another --output folder.')
    # Publish only a complete, verified download. Temporary files are cleaned on failure.
    with tempfile.TemporaryDirectory(prefix=f'.{name}-', dir=output) as tmp:
        folder = Path(tmp)
        captions = save_captions(matches[0], folder, f'{name}.{language}')
        audio = folder / f'{name}.m4a'
        expected = sum(float(t) for t in re.findall(r'#EXTINF:([\d.]+)', rendition))
        print('Downloading original audio (this may take a few minutes)…', flush=True)
        run_media(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-i', audio_url, '-map', '0:a:0', '-vn', '-c:a', 'copy', str(audio)], 'Audio download')
        probe = json.loads(run_media(['ffprobe', '-v', 'error', '-show_entries',
                                    'format=duration,size:stream=codec_name,sample_rate,channels',
                                    '-of', 'json', str(audio)], 'Audio inspection'))
        duration = float(probe['format']['duration'])
        if expected and abs(duration - expected) > 2:
            raise DownloadError(f'Incomplete audio: expected {expected:.2f}s, received {duration:.2f}s.')
        print('Checking the complete audio for decoding errors…', flush=True)
        run_media(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(audio),
                   '-f', 'null', '-'], 'Audio verification')
        metadata = {'source_url': url, 'retrieved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'audio': probe, 'captions': captions, 'audio_decode_verified': True}
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        folder.rename(destination)
    print(f'Done: {destination.resolve()}\nAudio duration: {format_time(round(duration * 1000))}', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', help='LES MILLS workout URL (quote it in your shell)')
    # Defaults stay outside the repo: the profile holds session cookies and the audio is research-only.
    parser.add_argument('-o', '--output', type=Path, default=Path('~/paincave-research/workouts'),
                        help='Parent folder for workout folders (default: ~/paincave-research/workouts)')
    parser.add_argument('--language', default='en', help='Caption language code (default: en)')
    parser.add_argument('--profile', type=Path, default=Path('~/.paincave-analysis/workout-chrome'),
                        help='Dedicated persistent Chrome profile (default: ~/.paincave-analysis/workout-chrome); '
                             'do not use your everyday Chrome profile')
    parser.add_argument('--timeout', type=int, default=300, help='Seconds allowed for login/media discovery (default: 300)')
    parser.add_argument('--har', type=Path, help='Use a Chrome HAR export with response bodies instead of opening Chrome')
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme != 'https' or parsed.hostname != 'my.lesmillsondemand.com' or not parsed.path.startswith('/workout/'):
        parser.error('Provide an https://my.lesmillsondemand.com/workout/... URL.')
    if args.timeout < 1 or not re.fullmatch(r'[A-Za-z0-9-]+', args.language):
        parser.error('--timeout must be positive and --language must be a language code.')
    for tool in ['ffmpeg', 'ffprobe']:
        if not shutil.which(tool):
            parser.error(f'{tool} is missing. On macOS: brew install ffmpeg')
    os.umask(0o077)
    try:
        master_url, master = (master_from_har(args.har) if args.har else
                              capture_master(args.url, args.profile.expanduser().resolve(), args.timeout))
        download(args.url, master_url, master, args.output.expanduser().resolve(), args.language)
    except KeyboardInterrupt:
        print('\nCancelled.', file=sys.stderr)
        return 130
    except (DownloadError, OSError, ValueError, KeyError) as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
