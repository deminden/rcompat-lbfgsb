#!/usr/bin/env python3
"""Relocate copied macOS R-package library paths in an independent reference env."""
import argparse
import os
from pathlib import Path
import re
import subprocess


def repair(old, new):
    changed = 0
    for path in sorted((new / 'lib/R/library').rglob('*')):
        if not path.is_file() or path.suffix not in {'.so', '.dylib'}:
            continue
        metadata = subprocess.run(['otool', '-l', str(path)], check=True, capture_output=True, text=True).stdout
        paths = re.findall(r'cmd LC_RPATH\n\s*cmdsize \d+\n\s*path (.*?) \(offset', metadata)
        edits = [('-rpath', value, str(new)+value[len(str(old)):]) for value in paths if value.startswith(str(old)+'/')]
        dependencies = subprocess.run(['otool', '-L', str(path)], check=True, capture_output=True, text=True).stdout.splitlines()[1:]
        for dependency in dependencies:
            value = dependency.strip().split(' (compatibility')[0]
            if value.startswith(str(old)+'/'):
                edits.append(('-change', value, str(new)+value[len(str(old)):]))
        if not edits:
            continue
        original = old / path.relative_to(new)
        if original.exists() and os.stat(original).st_ino == os.stat(path).st_ino:
            raise RuntimeError(f'{path}: shared inode; refusing to alter the old reference')
        command = ['install_name_tool']
        for edit in edits:
            command.extend(edit)
        subprocess.run([*command, str(path)], check=True)
        subprocess.run(['codesign', '--force', '--sign', '-', str(path)], check=True, capture_output=True)
        changed += 1
    print(f'Relocated {changed} copied R-package binaries; old environment was not modified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_prefix', type=Path)
    parser.add_argument('new_prefix', type=Path)
    args = parser.parse_args()
    if args.old_prefix.resolve() == args.new_prefix.resolve():
        parser.error('source and destination must be different')
    repair(args.old_prefix.resolve(), args.new_prefix.resolve())
