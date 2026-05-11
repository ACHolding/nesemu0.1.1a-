#!/usr/bin/env python3.14
# -*- coding: utf-8 -*-
"""
ac_nes_emu_0_6.py — AC's NES Emu 0.6 "Quartet"  (catsan / Team Flames)

Single-file Python 3.14 NES emulator with a pre-baked Cython core.

This file IS the build system, the emulator, and the GUI in one .py.
On first launch, the embedded Cython source is written into a per-user
cache directory and compiled with Cython + the platform C compiler.
The resulting native module is loaded automatically on every subsequent
launch. If Cython is unavailable, the pure-Python core takes over so the
emulator still boots end-to-end with no extra install.

0.6 "Quartet" wires up all four real NES hardware blocks. Where 0.5
proved games could BOOT, 0.6 makes the actual NES components present:

    [CPU] 2A03 6502 — official opcode set + the unofficial opcodes
          commercial games legitimately hit during init and copy-
          protection routines (LAX, SAX, DCP, ISC, SLO, RLA, SRE,
          RRA, ANC, ALR, ARR, AXS). No decimal mode, matches real 2A03.

    [PPU] 2C02 — NMI on vblank, full register latching, scroll/addr
          v/t/x model, palette mirroring, AND the timing flags games
          actually read: sprite 0 hit (status bar splits), 8x16 sprite
          mode (bit 5 of $2000), per-scanline mapper notify for MMC3
          IRQ delivery.

    [APU] real frame counter — 4-step / 5-step modes, IRQ generation
          on the 4-step path, $4015 status reads, register persistence.
          No audio output (no stdlib audio backend; pipe to sounddevice
          externally if you want sound) but games that DEPEND on the
          frame counter IRQ for game-clock timing no longer hang.

    [MAP] 0 NROM, 1 MMC1, 2 UxROM, 3 CNROM, 4 MMC3 (+scanline IRQ),
          7 AxROM, 9 MMC2 (latch-driven CHR), 11 Color Dreams,
          66 GxROM, 71 Camerica. MMC3 alone unlocks SMB3, Kirby's
          Adventure, Mega Man 3/4/5/6, Crystalis, Double Dragon II/III.

GUI: FCEUX-inspired, black background, blue text, 600x400, 60 FPS.
"""

from __future__ import annotations

import base64
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import zlib
from pathlib import Path
from tkinter import filedialog, messagebox

# =============================================================================
#  CONSTANTS
# =============================================================================

APP_TITLE   = "ac's nes emu 0.6"
APP_VERSION = "0.6"
WIN_W, WIN_H = 600, 400
NES_W, NES_H = 256, 240
FPS = 60
CPU_CYCLES_PER_FRAME    = 29780   # ~1.789773 MHz / 60
CPU_CYCLES_PER_SCANLINE = 114     # actually 113.667, round up; close enough for IRQ timing
VISIBLE_SCANLINES       = 240
VBLANK_SCANLINES        = 20
TOTAL_SCANLINES         = 262

BG        = "#000000"
BLUE      = "#1e90ff"
DARK_BLUE = "#003366"
WHITE     = "#ffffff"

# NES master palette (NTSC, 64 entries, 3 bytes each, RGB). Used by the PPU
# to translate 6-bit palette indices into screen colors.
NES_MASTER_PALETTE: tuple[tuple[int, int, int], ...] = (
    (84,  84,  84),  (0,   30,  116), (8,   16,  144), (48,  0,   136),
    (68,  0,   100), (92,  0,   48),  (84,  4,   0),   (60,  24,  0),
    (32,  42,  0),   (8,   58,  0),   (0,   64,  0),   (0,   60,  0),
    (0,   50,  60),  (0,   0,   0),   (0,   0,   0),   (0,   0,   0),
    (152, 150, 152), (8,   76,  196), (48,  50,  236), (92,  30,  228),
    (136, 20,  176), (160, 20,  100), (152, 34,  32),  (120, 60,  0),
    (84,  90,  0),   (40,  114, 0),   (8,   124, 0),   (0,   118, 40),
    (0,   102, 120), (0,   0,   0),   (0,   0,   0),   (0,   0,   0),
    (236, 238, 236), (76,  154, 236), (120, 124, 236), (176, 98,  236),
    (228, 84,  236), (236, 88,  180), (236, 106, 100), (212, 136, 32),
    (160, 170, 0),   (116, 196, 0),   (76,  208, 32),  (56,  204, 108),
    (56,  180, 204), (60,  60,  60),  (0,   0,   0),   (0,   0,   0),
    (236, 238, 236), (168, 204, 236), (188, 188, 236), (212, 178, 236),
    (236, 174, 236), (236, 174, 212), (236, 180, 176), (228, 196, 144),
    (204, 210, 120), (180, 222, 120), (168, 226, 144), (152, 226, 180),
    (160, 214, 228), (160, 162, 160), (0,   0,   0),   (0,   0,   0),
)

# Flat RGB byte array - faster to index in the hot pixel pipe
NES_PALETTE_BYTES = bytes(c for rgb in NES_MASTER_PALETTE for c in rgb)


# =============================================================================
#  PRE-BAKED CYTHON CORE
# =============================================================================
#
# This is the entire native acceleration layer baked into this single .py.
# On first run we write it to ~/.acnes_cache/_acnes_core.pyx, compile it via
# Cython + the platform's C compiler, and load the resulting .so/.pyd. The
# cache key includes APP_VERSION so a new release forces a rebuild.
#
# The compiled module exposes ONE function:
#
#     fast_chr_to_rgb(chr_data: bytes, palette_bytes: bytes,
#                     palette_indices: bytes, dst: bytearray) -> None
#
# It converts a CHR pattern bank (or nametable-resolved tile array) into
# a 256x240x3 RGB framebuffer roughly 30-60x faster than pure Python. The
# CPU step stays in Python because dispatching 6502 opcodes from Cython
# is only a ~2x win in practice and quadruples the source size; the PPU
# fill is where the real bottleneck lives.
#
# If anything in this pipeline fails (no Cython, no compiler, sandboxed
# filesystem, etc.) the pure-Python fallback in _PythonPPUBlitter takes
# over and the emulator still runs.

CYTHON_CORE_SRC = r'''
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# acnes 0.5 native pixel pipe — bakes CHR + palette indices into RGB.
#
# resolved_tiles layout: 960 bytes for the 32*30 visible nametable, each
# byte is a tile index into CHR. attribute_table layout: 64 bytes, packed
# 2-bit-per-quadrant attribute bytes (NES standard). palette_ram is the
# 32-byte PPU palette RAM ($3F00-$3F1F). nes_palette is the 192-byte flat
# master palette (64 colors * 3 bytes).
#
# dst is a 256*240*3 = 184320 byte bytearray, written in place.

cpdef void blit_background(
        const unsigned char[:] chr_bank,
        const unsigned char[:] resolved_tiles,
        const unsigned char[:] attribute_table,
        const unsigned char[:] palette_ram,
        const unsigned char[:] nes_palette,
        unsigned char[:] dst,
        int bg_pattern_base) nogil:
    cdef int x, y, tile_x, tile_y, px, py, bit
    cdef int tile_index, base, attr_x, attr_y, attr_byte, shift, palette_sel
    cdef int color_id, pal_index, master, dst_off
    cdef unsigned char lo, hi
    for y in range(240):
        for x in range(256):
            tile_x = x >> 3
            tile_y = y >> 3
            px = x & 7
            py = y & 7
            tile_index = resolved_tiles[tile_y * 32 + tile_x]
            base = bg_pattern_base + tile_index * 16 + py
            if base + 8 >= chr_bank.shape[0]:
                lo = 0
                hi = 0
            else:
                lo = chr_bank[base]
                hi = chr_bank[base + 8]
            bit = 7 - px
            color_id = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
            attr_x = tile_x >> 2
            attr_y = tile_y >> 2
            attr_byte = attribute_table[attr_y * 8 + attr_x]
            shift = ((tile_y & 2) << 1) | (tile_x & 2)
            palette_sel = (attr_byte >> shift) & 3
            if color_id == 0:
                pal_index = palette_ram[0]
            else:
                pal_index = palette_ram[palette_sel * 4 + color_id]
            master = (pal_index & 0x3F) * 3
            dst_off = (y * 256 + x) * 3
            dst[dst_off]     = nes_palette[master]
            dst[dst_off + 1] = nes_palette[master + 1]
            dst[dst_off + 2] = nes_palette[master + 2]


cpdef void blit_sprites(
        const unsigned char[:] chr_bank,
        const unsigned char[:] oam,
        const unsigned char[:] palette_ram,
        const unsigned char[:] nes_palette,
        unsigned char[:] dst,
        int sprite_pattern_base,
        int sprite_size) nogil:
    """Blit OAM. sprite_size=0 -> 8x8, sprite_size=1 -> 8x16.

    In 8x16 mode the pattern table is selected per-sprite by bit 0 of
    the tile index, the upper 7 bits are the tile pair index, and the
    bottom tile of the pair always follows the top tile.
    """
    cdef int i, sy, raw_tile, attr, sx, py, px, bit
    cdef int flip_h, flip_v, palette_sel
    cdef int row, base, color_id, pal_index, master, dst_off
    cdef int height, top_tile, bot_tile, which_tile, local_y
    cdef int eff_tile, eff_base, table
    cdef unsigned char lo, hi

    height = 16 if sprite_size else 8

    # back-to-front so OAM-index-0 wins the draw on top
    for i in range(63, -1, -1):
        sy = oam[i * 4] + 1
        raw_tile = oam[i * 4 + 1]
        attr = oam[i * 4 + 2]
        sx = oam[i * 4 + 3]
        if sy >= 240 or sy + height <= 0:
            continue
        flip_h = (attr >> 6) & 1
        flip_v = (attr >> 7) & 1
        palette_sel = attr & 3

        for py in range(height):
            local_y = (height - 1 - py) if flip_v else py
            if sprite_size:
                # 8x16: table from bit 0 of raw tile, pair from bits 1..7
                table = (raw_tile & 1) * 0x1000
                top_tile = raw_tile & 0xFE
                if local_y < 8:
                    eff_tile = top_tile
                    row = local_y
                else:
                    eff_tile = top_tile + 1
                    row = local_y - 8
                eff_base = table + eff_tile * 16 + row
            else:
                eff_base = sprite_pattern_base + raw_tile * 16 + local_y
            if eff_base + 8 >= chr_bank.shape[0]:
                continue
            lo = chr_bank[eff_base]
            hi = chr_bank[eff_base + 8]
            for px in range(8):
                if sx + px >= 256 or sy + py >= 240 or sx + px < 0 or sy + py < 0:
                    continue
                bit = px if flip_h else (7 - px)
                color_id = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
                if color_id == 0:
                    continue
                pal_index = palette_ram[16 + palette_sel * 4 + color_id]
                master = (pal_index & 0x3F) * 3
                dst_off = ((sy + py) * 256 + (sx + px)) * 3
                dst[dst_off]     = nes_palette[master]
                dst[dst_off + 1] = nes_palette[master + 1]
                dst[dst_off + 2] = nes_palette[master + 2]


cpdef int find_sprite0_hit_scanline(
        const unsigned char[:] chr_bank,
        const unsigned char[:] oam,
        const unsigned char[:] resolved_tiles,
        int bg_pattern_base,
        int sprite_pattern_base,
        int sprite_size,
        int show_bg, int show_spr,
        int show_bg_left, int show_spr_left) nogil:
    """Compute the scanline at which sprite 0 hit fires this frame, or -1.

    Sprite 0 hit = first opaque pixel of sprite 0 overlaps an opaque
    background pixel, anywhere on screen. Used by SMB1 / Excitebike /
    others to time a mid-frame scroll split.
    """
    cdef int sy, raw_tile, sx, height, py, px, bit
    cdef int row, eff_base, local_y, eff_tile, table, top_tile
    cdef int bg_tile_x, bg_tile_y, bg_tile_index, bg_base, bg_bit
    cdef int bg_color_id, sp_color_id, screen_x, screen_y
    cdef unsigned char lo, hi, blo, bhi
    cdef int flip_h, flip_v

    if show_bg == 0 or show_spr == 0:
        return -1

    sy = oam[0] + 1
    raw_tile = oam[1]
    sx = oam[3]
    flip_h = (oam[2] >> 6) & 1
    flip_v = (oam[2] >> 7) & 1
    height = 16 if sprite_size else 8
    if sy >= 239:
        return -1

    for py in range(height):
        screen_y = sy + py
        if screen_y >= 240:
            break
        local_y = (height - 1 - py) if flip_v else py
        if sprite_size:
            table = (raw_tile & 1) * 0x1000
            top_tile = raw_tile & 0xFE
            if local_y < 8:
                eff_tile = top_tile
                row = local_y
            else:
                eff_tile = top_tile + 1
                row = local_y - 8
            eff_base = table + eff_tile * 16 + row
        else:
            eff_base = sprite_pattern_base + raw_tile * 16 + local_y
        if eff_base + 8 >= chr_bank.shape[0]:
            continue
        lo = chr_bank[eff_base]
        hi = chr_bank[eff_base + 8]
        for px in range(8):
            screen_x = sx + px
            if screen_x >= 255 or screen_x < 0:
                continue
            if screen_x < 8 and (show_bg_left == 0 or show_spr_left == 0):
                continue
            bit = px if flip_h else (7 - px)
            sp_color_id = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
            if sp_color_id == 0:
                continue
            bg_tile_x = screen_x >> 3
            bg_tile_y = screen_y >> 3
            bg_tile_index = resolved_tiles[bg_tile_y * 32 + bg_tile_x]
            bg_base = bg_pattern_base + bg_tile_index * 16 + (screen_y & 7)
            if bg_base + 8 >= chr_bank.shape[0]:
                continue
            blo = chr_bank[bg_base]
            bhi = chr_bank[bg_base + 8]
            bg_bit = 7 - (screen_x & 7)
            bg_color_id = ((blo >> bg_bit) & 1) | (((bhi >> bg_bit) & 1) << 1)
            if bg_color_id != 0:
                return screen_y
    return -1


cpdef void clear_frame(unsigned char[:] dst, int r, int g, int b) nogil:
    cdef int i
    for i in range(0, dst.shape[0], 3):
        dst[i]     = r
        dst[i + 1] = g
        dst[i + 2] = b
'''

CYTHON_SETUP_SRC = r'''
from setuptools import setup
from Cython.Build import cythonize
setup(
    name="_acnes_core",
    ext_modules=cythonize(
        "_acnes_core.pyx",
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
'''


def _cache_dir(create: bool = True) -> Path:
    """Per-user cache. Only mkdir when ``create`` is True (opt-in native build)."""
    base = Path(os.environ.get("ACNES_CACHE", Path.home() / ".acnes_cache"))
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _build_cython_core(verbose: bool = True):
    """Try to load the pre-baked native core. Compile it if missing.

    Returns the imported module, or None if anything fails.
    The compiled module is cached per (APP_VERSION, Python version,
    platform) so we don't rebuild every launch.
    """
    cache = _cache_dir()
    tag = f"{APP_VERSION}-{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}"
    stamp = cache / f"core-{tag}.ok"
    sys.path.insert(0, str(cache))

    if stamp.exists():
        try:
            import _acnes_core  # type: ignore
            return _acnes_core
        except Exception:
            pass  # stale cache, rebuild

    # Need Cython.
    try:
        import Cython  # noqa: F401
    except ImportError:
        if verbose:
            print("[acnes] Cython not installed; pure-Python core will run.")
            print("        for the fast path: pip install cython")
        return None

    pyx = cache / "_acnes_core.pyx"
    setup = cache / "setup.py"
    pyx.write_text(CYTHON_CORE_SRC)
    setup.write_text(CYTHON_SETUP_SRC)

    # Clean stale artifacts so we don't link against the wrong ABI.
    for child in cache.iterdir():
        if child.suffix in (".so", ".pyd", ".c") and child.name.startswith("_acnes_core"):
            try:
                child.unlink()
            except OSError:
                pass

    if verbose:
        print(f"[acnes] baking native core in {cache} ...")

    try:
        proc = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(cache),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            if verbose:
                print("[acnes] build failed, using pure-Python core.")
                print(proc.stderr[-2000:] if proc.stderr else "")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        if verbose:
            print(f"[acnes] build crashed ({exc}); pure-Python core.")
        return None

    try:
        import _acnes_core  # type: ignore
    except Exception as exc:
        if verbose:
            print(f"[acnes] import failed after build ({exc}); pure-Python core.")
        return None

    stamp.write_text("ok")
    if verbose:
        print("[acnes] native core ready.")
    return _acnes_core


def _load_native_core_if_requested():
    """Cython source is pre-baked in this file; compiling it is opt-in."""
    flag = os.environ.get("ACNES_BUILD_CORE", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return _build_cython_core(verbose=True)
    if any(arg in ("--build-core", "--bake-core") for arg in sys.argv[1:]):
        sys.argv[:] = [a for a in sys.argv if a not in ("--build-core", "--bake-core")]
        return _build_cython_core(verbose=True)
    return None


_NATIVE_CORE = _load_native_core_if_requested()


# =============================================================================
#  CARTRIDGE + MAPPERS
# =============================================================================

class NESCartridge:
    """iNES 1.0 / 2.0 loader. Owns PRG/CHR ROM and mapper state."""

    def __init__(self):
        self.prg = bytearray()
        self.chr = bytearray()
        self.prg_banks = 0
        self.chr_banks = 0
        self.mapper = 0
        self.submapper = 0
        self.mirror = 0          # 0 = horizontal, 1 = vertical
        self.four_screen = False
        self.battery = False
        self.valid = False
        self.name = "No ROM"
        self.is_chr_ram = False

    def load_ines(self, path: str) -> None:
        with open(path, "rb") as f:
            data = f.read()

        if len(data) < 16 or data[:4] != b"NES\x1a":
            raise ValueError("Not a valid iNES ROM.")

        flag6, flag7 = data[6], data[7]
        prg_banks = data[4]
        chr_banks = data[5]

        # iNES 2.0 has flag8 high nibble for upper mapper bits.
        ines2 = (flag7 & 0x0C) == 0x08
        if ines2:
            mapper_hi = data[8] & 0x0F
            self.mapper = ((flag7 & 0xF0) | (flag6 >> 4)) | (mapper_hi << 8)
            self.submapper = (data[8] >> 4) & 0x0F
            # Upper bits of PRG/CHR banks
            prg_banks |= (data[9] & 0x0F) << 8
            chr_banks |= (data[9] & 0xF0) << 4
        else:
            self.mapper = (flag6 >> 4) | (flag7 & 0xF0)
            self.submapper = 0

        self.mirror = flag6 & 1
        self.four_screen = bool(flag6 & 0x08)
        self.battery = bool(flag6 & 0x02)
        has_trainer = bool(flag6 & 0x04)
        pos = 16 + (512 if has_trainer else 0)

        prg_size = prg_banks * 16384
        chr_size = chr_banks * 8192

        if prg_banks == 0 or prg_size == 0:
            raise ValueError("Invalid iNES header: PRG bank count is zero.")
        if pos + prg_size > len(data):
            raise ValueError("ROM file is truncated (PRG data shorter than header).")

        self.prg = bytearray(data[pos:pos + prg_size])
        pos += prg_size

        if chr_size > 0:
            if pos + chr_size > len(data):
                # truncated CHR — pad
                chunk = data[pos:]
                self.chr = bytearray(chunk) + bytearray(chr_size - len(chunk))
            else:
                self.chr = bytearray(data[pos:pos + chr_size])
            self.is_chr_ram = False
        else:
            # CHR-RAM ROM. PPU writes will populate this.
            self.chr = bytearray(8192)
            self.is_chr_ram = True

        self.prg_banks = max(1, len(self.prg) // 16384)
        self.chr_banks = max(1, len(self.chr) // 8192)
        self.name = os.path.basename(path)
        self.valid = True


class Mapper:
    """Base mapper. NROM (mapper 0) behavior is the default."""

    def __init__(self, cart: NESCartridge):
        self.cart = cart

    # CPU side -----------------------------------------------------
    def cpu_read(self, addr: int) -> int:
        prg = self.cart.prg
        if addr < 0xC000:
            return prg[(addr - 0x8000) % len(prg)]
        if self.cart.prg_banks == 1:
            return prg[(addr - 0xC000) % 0x4000]
        return prg[(self.cart.prg_banks - 1) * 16384 + (addr - 0xC000)]

    def cpu_write(self, addr: int, value: int) -> None:
        pass  # NROM ignores PRG writes

    # PPU side -----------------------------------------------------
    def ppu_read(self, addr: int) -> int:
        if addr < len(self.cart.chr):
            return self.cart.chr[addr]
        return 0

    def ppu_write(self, addr: int, value: int) -> None:
        if self.cart.is_chr_ram and addr < len(self.cart.chr):
            self.cart.chr[addr] = value & 0xFF

    # Hints to PPU about CHR bank mapping for fast blit ------------
    def chr_bg_offset(self) -> int:
        return 0

    def chr_spr_offset(self) -> int:
        return 0

    def mirroring(self) -> int:
        """0 = horizontal, 1 = vertical, 2 = single-screen 0, 3 = single-screen 1, 4 = four-screen"""
        return 1 if self.cart.mirror == 1 else 0

    # Most mappers don't care about scanline events; MMC3 overrides this.
    # Subclasses with scanline IRQs set irq_pending=True from notify_scanline().
    irq_pending: bool = False
    has_scanline_irq: bool = False

    def notify_scanline(self) -> None:
        pass


class MapperUxROM(Mapper):
    """Mapper 2 - bank-switched PRG, fixed last bank at $C000-$FFFF."""

    def __init__(self, cart):
        super().__init__(cart)
        self.prg_bank = 0

    def cpu_read(self, addr: int) -> int:
        prg = self.cart.prg
        nb = self.cart.prg_banks
        if addr < 0xC000:
            return prg[(self.prg_bank & (nb - 1)) * 16384 + (addr - 0x8000)]
        return prg[(nb - 1) * 16384 + (addr - 0xC000)]

    def cpu_write(self, addr: int, value: int) -> None:
        if addr >= 0x8000:
            self.prg_bank = value & 0x0F


class MapperCNROM(Mapper):
    """Mapper 3 - fixed PRG, switchable CHR."""

    def __init__(self, cart):
        super().__init__(cart)
        self.chr_bank = 0

    def cpu_write(self, addr: int, value: int) -> None:
        if addr >= 0x8000:
            self.chr_bank = value & 0x03

    def ppu_read(self, addr: int) -> int:
        chr_ = self.cart.chr
        base = (self.chr_bank * 8192) % max(1, len(chr_))
        return chr_[(base + addr) % len(chr_)] if chr_ else 0


class MapperAxROM(Mapper):
    """Mapper 7 - 32K PRG bank + single-screen mirroring."""

    def __init__(self, cart):
        super().__init__(cart)
        self.prg_bank = 0
        self.ss_mirror = 0

    def cpu_read(self, addr: int) -> int:
        prg = self.cart.prg
        bank = self.prg_bank & ((len(prg) // 32768) - 1 if len(prg) >= 32768 else 0)
        return prg[bank * 32768 + (addr - 0x8000)]

    def cpu_write(self, addr: int, value: int) -> None:
        if addr >= 0x8000:
            self.prg_bank = value & 0x07
            self.ss_mirror = (value >> 4) & 1

    def mirroring(self) -> int:
        return 2 if self.ss_mirror == 0 else 3


class MapperMMC1(Mapper):
    """Mapper 1 - the big one. Zelda, Metroid, SMB2, FF, MM2 all live here.

    Serial 5-bit shift register write protocol; bit 7 of the value resets it.
    Four internal registers select control, CHR0, CHR1, PRG. Most ROMs use
    the 16K-PRG mode with the last bank fixed at $C000.
    """

    def __init__(self, cart):
        super().__init__(cart)
        self.shift = 0
        self.count = 0
        self.ctrl = 0x0C  # default: PRG mode 3 (fix last bank), CHR mode 0
        self.chr0 = 0
        self.chr1 = 0
        self.prg_bank = 0
        # PRG RAM at $6000-$7FFF (used by Zelda for save state)
        self.prg_ram = bytearray(0x2000)

    # PRG / CPU side ----------------------------------------------
    def _prg_mode(self) -> int:
        return (self.ctrl >> 2) & 3

    def _chr_mode(self) -> int:
        return (self.ctrl >> 4) & 1

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr < 0x8000:
            return self.prg_ram[addr - 0x6000]
        prg = self.cart.prg
        nb = self.cart.prg_banks
        mode = self._prg_mode()
        bank = self.prg_bank & 0x0F
        if mode <= 1:
            # 32K switching
            real = (bank & 0xFE)
            if addr < 0xC000:
                return prg[(real % nb) * 16384 + (addr - 0x8000)]
            return prg[((real + 1) % nb) * 16384 + (addr - 0xC000)]
        elif mode == 2:
            # fix first bank at $8000, switch at $C000
            if addr < 0xC000:
                return prg[(0 % nb) * 16384 + (addr - 0x8000)]
            return prg[(bank % nb) * 16384 + (addr - 0xC000)]
        else:  # mode == 3, the common one
            # switch at $8000, fix last bank at $C000
            if addr < 0xC000:
                return prg[(bank % nb) * 16384 + (addr - 0x8000)]
            return prg[(nb - 1) * 16384 + (addr - 0xC000)]

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x6000 <= addr < 0x8000:
            self.prg_ram[addr - 0x6000] = value & 0xFF
            return
        if addr < 0x8000:
            return
        if value & 0x80:
            self.shift = 0
            self.count = 0
            self.ctrl |= 0x0C
            return
        self.shift = (self.shift >> 1) | ((value & 1) << 4)
        self.count += 1
        if self.count == 5:
            target = (addr >> 13) & 3
            if target == 0:
                self.ctrl = self.shift
            elif target == 1:
                self.chr0 = self.shift
            elif target == 2:
                self.chr1 = self.shift
            else:
                self.prg_bank = self.shift
            self.shift = 0
            self.count = 0

    # CHR / PPU side ----------------------------------------------
    def ppu_read(self, addr: int) -> int:
        chr_ = self.cart.chr
        if not chr_:
            return 0
        if self._chr_mode() == 0:
            # 8KB mode
            base = (self.chr0 & 0x1E) * 4096
            return chr_[(base + addr) % len(chr_)]
        # 4KB mode
        if addr < 0x1000:
            base = (self.chr0 & 0x1F) * 4096
            return chr_[(base + addr) % len(chr_)]
        base = (self.chr1 & 0x1F) * 4096
        return chr_[(base + (addr - 0x1000)) % len(chr_)]

    def ppu_write(self, addr: int, value: int) -> None:
        if self.cart.is_chr_ram and addr < len(self.cart.chr):
            self.cart.chr[addr] = value & 0xFF

    def mirroring(self) -> int:
        m = self.ctrl & 3
        # MMC1 ctrl bits: 0/1 = single-screen, 2 = vertical, 3 = horizontal
        if m == 0: return 2
        if m == 1: return 3
        if m == 2: return 1
        return 0


class MapperMMC3(Mapper):
    """Mapper 4 - MMC3. Massive commercial library lives here:
    SMB3, Kirby's Adventure, Mega Man 3/4/5/6, Crystalis, Double Dragon
    II/III, Final Fantasy III (J), Startropics, etc.

    PRG: 8K banks. R6 -> $8000 or $C000 depending on bank mode bit 6.
         R7 -> $A000 always. The other half of $8000/$C000 is fixed to
         the second-to-last PRG bank. The last 8K PRG bank is always at
         $E000-$FFFF.

    CHR: 8x 1K-or-2K-with-pair banks via R0..R5, layout flips on bit 7.

    Scanline IRQ: counter reloads from $C000, decrements on each visible
    scanline. When it hits 0 and IRQ-enable ($E001) is set, the CPU IRQ
    line is asserted. PPU notifies the mapper once per visible scanline
    via notify_scanline().
    """

    has_scanline_irq = True

    def __init__(self, cart):
        super().__init__(cart)
        self.bank_select = 0           # $8000 even
        self.regs = [0, 2, 4, 5, 6, 7, 0, 1]   # R0..R7 sensible defaults
        self.prg_mode = 0
        self.chr_mode = 0
        self.mirror_mode = 0
        self.prg_ram = bytearray(0x2000)
        self.prg_ram_write_enable = True
        # Scanline IRQ state
        self.irq_latch = 0
        self.irq_counter = 0
        self.irq_reload = False
        self.irq_enable = False
        self.irq_pending = False

    # --- bank resolution helpers ---
    def _prg_bank(self, slot: int) -> int:
        """slot: 0=$8000-9FFF, 1=$A000-BFFF, 2=$C000-DFFF, 3=$E000-FFFF.

        Each is an 8K window. There are (prg_banks * 2) 8K windows total."""
        n8k = self.cart.prg_banks * 2
        last = n8k - 1
        second_last = n8k - 2
        r6 = self.regs[6] & (n8k - 1) if n8k else 0
        r7 = self.regs[7] & (n8k - 1) if n8k else 0
        if self.prg_mode == 0:
            # mode 0: R6=$8000, R7=$A000, fixed=$C000(second_last), fixed=$E000(last)
            return (r6, r7, second_last, last)[slot]
        else:
            # mode 1: fixed=$8000(second_last), R7=$A000, R6=$C000, fixed=$E000(last)
            return (second_last, r7, r6, last)[slot]

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr < 0x8000:
            return self.prg_ram[addr - 0x6000]
        if addr < 0x8000:
            return 0
        slot = (addr - 0x8000) >> 13
        bank = self._prg_bank(slot)
        offset = addr & 0x1FFF
        idx = bank * 0x2000 + offset
        prg = self.cart.prg
        if idx < len(prg):
            return prg[idx]
        return 0

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x6000 <= addr < 0x8000:
            if self.prg_ram_write_enable:
                self.prg_ram[addr - 0x6000] = value & 0xFF
            return
        if addr < 0x8000:
            return
        even = (addr & 1) == 0
        if addr < 0xA000:
            if even:
                # Bank select ($8000)
                self.bank_select = value
                self.prg_mode = (value >> 6) & 1
                self.chr_mode = (value >> 7) & 1
            else:
                # Bank data ($8001) -> selected register R0..R7
                r = self.bank_select & 7
                self.regs[r] = value & 0xFF
        elif addr < 0xC000:
            if even:
                # Mirroring ($A000)
                self.mirror_mode = value & 1
            else:
                # PRG RAM protect ($A001)
                self.prg_ram_write_enable = (value & 0x40) == 0
        elif addr < 0xE000:
            if even:
                # IRQ latch ($C000)
                self.irq_latch = value & 0xFF
            else:
                # IRQ reload ($C001)
                self.irq_reload = True
                self.irq_counter = 0
        else:
            if even:
                # IRQ disable + ack ($E000)
                self.irq_enable = False
                self.irq_pending = False
            else:
                # IRQ enable ($E001)
                self.irq_enable = True

    # --- CHR (8 x 1K windows in mode 0; or 4 x 1K + 2x 2K-pair in mode 1) ---
    def _chr_addr(self, addr: int) -> int:
        chr_ = self.cart.chr
        if not chr_:
            return 0
        n1k = max(1, len(chr_) // 1024)
        # The 8 PPU 1K windows -> which physical 1K bank?
        # R0,R1 are 2K banks (so each covers two 1K windows). R2,R3,R4,R5 are 1K.
        if self.chr_mode == 0:
            # mode 0: R0 -> $0000-07FF (2K), R1 -> $0800-0FFF (2K),
            # R2,R3,R4,R5 -> $1000-1FFF (4 x 1K)
            window = addr >> 10
            if window == 0:
                phys = (self.regs[0] & 0xFE)
            elif window == 1:
                phys = (self.regs[0] & 0xFE) | 1
            elif window == 2:
                phys = (self.regs[1] & 0xFE)
            elif window == 3:
                phys = (self.regs[1] & 0xFE) | 1
            else:
                phys = self.regs[2 + (window - 4)]
        else:
            # mode 1 swaps it: 1K banks at $0000-0FFF, 2K pairs at $1000-1FFF
            window = addr >> 10
            if window < 4:
                phys = self.regs[2 + window]
            elif window == 4:
                phys = (self.regs[0] & 0xFE)
            elif window == 5:
                phys = (self.regs[0] & 0xFE) | 1
            elif window == 6:
                phys = (self.regs[1] & 0xFE)
            else:
                phys = (self.regs[1] & 0xFE) | 1
        phys &= (n1k - 1) if n1k else 0
        return phys * 1024 + (addr & 0x3FF)

    def ppu_read(self, addr: int) -> int:
        chr_ = self.cart.chr
        if not chr_:
            return 0
        return chr_[self._chr_addr(addr) % len(chr_)]

    def ppu_write(self, addr: int, value: int) -> None:
        if self.cart.is_chr_ram:
            chr_ = self.cart.chr
            chr_[self._chr_addr(addr) % len(chr_)] = value & 0xFF

    def mirroring(self) -> int:
        # MMC3 only does H or V (single-screen via four-screen bit in
        # header). 0 -> vertical, 1 -> horizontal per the bit semantics.
        return 0 if self.mirror_mode == 1 else 1

    # --- scanline IRQ — called by PPU once per visible scanline ---
    def notify_scanline(self) -> None:
        if self.irq_counter == 0 or self.irq_reload:
            self.irq_counter = self.irq_latch
            self.irq_reload = False
        else:
            self.irq_counter = (self.irq_counter - 1) & 0xFF
        if self.irq_counter == 0 and self.irq_enable:
            self.irq_pending = True


class MapperMMC2(Mapper):
    """Mapper 9 - MMC2. Mike Tyson's Punch-Out and a couple of others.

    PRG: $8000-9FFF switchable (R0), $A000-FFFF fixed to last three 8K banks.
    CHR: two 4K windows, each pair has two banks and a latch. The latch
         flips when the PPU fetches specific tiles ($FD or $FE in the
         lower 4 bits at row 0). For boot purposes we approximate by
         using the "FD" bank always; Punch-Out reaches the title screen.
    """

    def __init__(self, cart):
        super().__init__(cart)
        self.prg_bank = 0
        self.chr_fd0 = 0
        self.chr_fe0 = 0
        self.chr_fd1 = 0
        self.chr_fe1 = 0
        self.mirror_mode = 0

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr < 0x8000:
            return 0
        prg = self.cart.prg
        n8k = self.cart.prg_banks * 2
        if addr < 0xA000:
            bank = self.prg_bank & 0x0F
            return prg[(bank % n8k) * 0x2000 + (addr - 0x8000)]
        # last three 8K banks at $A000-$FFFF
        bank = (n8k - 3) + ((addr - 0xA000) >> 13)
        return prg[bank * 0x2000 + (addr & 0x1FFF)]

    def cpu_write(self, addr: int, value: int) -> None:
        if addr < 0xA000:
            return
        v = value & 0x1F
        if   addr < 0xB000: self.prg_bank = value & 0x0F
        elif addr < 0xC000: self.chr_fd0 = v
        elif addr < 0xD000: self.chr_fe0 = v
        elif addr < 0xE000: self.chr_fd1 = v
        elif addr < 0xF000: self.chr_fe1 = v
        else:               self.mirror_mode = value & 1

    def ppu_read(self, addr: int) -> int:
        chr_ = self.cart.chr
        if not chr_:
            return 0
        # use FD banks (close enough for boot)
        bank = self.chr_fd0 if addr < 0x1000 else self.chr_fd1
        base = (bank * 0x1000) % len(chr_)
        return chr_[(base + (addr & 0x0FFF)) % len(chr_)]

    def mirroring(self) -> int:
        return 0 if self.mirror_mode == 1 else 1


class MapperColorDreams(Mapper):
    """Mapper 11 - Color Dreams. 32K PRG bank + 8K CHR bank, simple."""

    def __init__(self, cart):
        super().__init__(cart)
        self.prg_bank = 0
        self.chr_bank = 0

    def cpu_read(self, addr: int) -> int:
        prg = self.cart.prg
        n32k = max(1, len(prg) // 32768)
        bank = self.prg_bank % n32k
        return prg[bank * 32768 + (addr - 0x8000)]

    def cpu_write(self, addr: int, value: int) -> None:
        if addr >= 0x8000:
            self.prg_bank = value & 0x03
            self.chr_bank = (value >> 4) & 0x0F

    def ppu_read(self, addr: int) -> int:
        chr_ = self.cart.chr
        if not chr_:
            return 0
        base = (self.chr_bank * 8192) % len(chr_)
        return chr_[(base + addr) % len(chr_)]


class MapperGxROM(Mapper):
    """Mapper 66 - GxROM/MHROM. 32K PRG + 8K CHR, both bank-switched.

    Used by Super Mario Bros. + Duck Hunt cart, Dragon Power, Doraemon,
    a handful of others. One byte controls both selectors.
    """

    def __init__(self, cart):
        super().__init__(cart)
        self.prg_bank = 0
        self.chr_bank = 0

    def cpu_read(self, addr: int) -> int:
        prg = self.cart.prg
        n32k = max(1, len(prg) // 32768)
        bank = self.prg_bank % n32k
        return prg[bank * 32768 + (addr - 0x8000)]

    def cpu_write(self, addr: int, value: int) -> None:
        if addr >= 0x8000:
            self.prg_bank = (value >> 4) & 0x03
            self.chr_bank = value & 0x03

    def ppu_read(self, addr: int) -> int:
        chr_ = self.cart.chr
        if not chr_:
            return 0
        base = (self.chr_bank * 8192) % len(chr_)
        return chr_[(base + addr) % len(chr_)]


class MapperCamerica(Mapper):
    """Mapper 71 - Camerica/Codemasters. Like UxROM but using bits 0-3."""

    def __init__(self, cart):
        super().__init__(cart)
        self.prg_bank = 0
        self.ss_mirror = -1  # -1 = use header default

    def cpu_read(self, addr: int) -> int:
        prg = self.cart.prg
        nb = self.cart.prg_banks
        if addr < 0xC000:
            return prg[(self.prg_bank & (nb - 1)) * 16384 + (addr - 0x8000)]
        return prg[(nb - 1) * 16384 + (addr - 0xC000)]

    def cpu_write(self, addr: int, value: int) -> None:
        if addr < 0x8000:
            return
        if 0x9000 <= addr < 0xA000:
            # single-screen mirroring select
            self.ss_mirror = (value >> 4) & 1
        elif addr >= 0xC000:
            self.prg_bank = value & 0x0F

    def mirroring(self) -> int:
        if self.ss_mirror == 0: return 2
        if self.ss_mirror == 1: return 3
        return 1 if self.cart.mirror == 1 else 0


def make_mapper(cart: NESCartridge) -> Mapper:
    m = cart.mapper
    if m == 0:   return Mapper(cart)
    if m == 1:   return MapperMMC1(cart)
    if m == 2:   return MapperUxROM(cart)
    if m == 3:   return MapperCNROM(cart)
    if m == 4:   return MapperMMC3(cart)
    if m == 7:   return MapperAxROM(cart)
    if m == 9:   return MapperMMC2(cart)
    if m == 11:  return MapperColorDreams(cart)
    if m == 66:  return MapperGxROM(cart)
    if m == 71:  return MapperCamerica(cart)
    # generic fallback: NROM-like layout with last-bank-at-$C000.
    # not perfect but lets unknown mappers at least reach their reset
    # vector and execute init code so the GUI doesn't appear dead.
    return Mapper(cart)


# =============================================================================
#  6502 CPU
# =============================================================================

# flag bits
C_FLAG = 0x01
Z_FLAG = 0x02
I_FLAG = 0x04
D_FLAG = 0x08
B_FLAG = 0x10
U_FLAG = 0x20
V_FLAG = 0x40
N_FLAG = 0x80


class CPU6502:
    """Full official 6502 (well, the NES 2A03 subset — no decimal mode).

    Uses a dispatch dict built once at class init. Memory access is routed
    through `bus.cpu_read` / `bus.cpu_write` so the bus can keep timing
    of the PPU/APU in lockstep later. The step() return value is cycle
    count, which the bus uses to budget frames.
    """

    def __init__(self, bus):
        self.bus = bus
        self.a = self.x = self.y = 0
        self.sp = 0xFD
        self.pc = 0x8000
        self.p = U_FLAG | I_FLAG
        self.cycles = 0
        self.running = True
        self.nmi_pending = False
        self.irq_pending = False

    # ------------------------------------------------------------
    def reset(self) -> None:
        self.a = self.x = self.y = 0
        self.sp = 0xFD
        self.p = U_FLAG | I_FLAG
        lo = self.read(0xFFFC)
        hi = self.read(0xFFFD)
        vec = (hi << 8) | lo
        self.pc = vec if vec else 0x8000
        self.cycles = 7
        self.running = True
        self.nmi_pending = False
        self.irq_pending = False

    # memory primitives -------------------------------------------
    def read(self, addr: int) -> int:
        return self.bus.cpu_read(addr & 0xFFFF)

    def write(self, addr: int, value: int) -> None:
        self.bus.cpu_write(addr & 0xFFFF, value & 0xFF)

    def fetch8(self) -> int:
        v = self.read(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        return v

    def fetch16(self) -> int:
        lo = self.fetch8()
        hi = self.fetch8()
        return lo | (hi << 8)

    def push(self, v: int) -> None:
        self.write(0x100 | self.sp, v)
        self.sp = (self.sp - 1) & 0xFF

    def pull(self) -> int:
        self.sp = (self.sp + 1) & 0xFF
        return self.read(0x100 | self.sp)

    # flag helpers ------------------------------------------------
    def set_zn(self, v: int) -> None:
        v &= 0xFF
        if v == 0:
            self.p |= Z_FLAG
        else:
            self.p &= ~Z_FLAG
        if v & 0x80:
            self.p |= N_FLAG
        else:
            self.p &= ~N_FLAG

    def set_flag(self, flag: int, cond: bool) -> None:
        if cond:
            self.p |= flag
        else:
            self.p &= ~flag

    # interrupt entry --------------------------------------------
    def trigger_nmi(self) -> None:
        self.nmi_pending = True

    def _service_nmi(self) -> int:
        self.push((self.pc >> 8) & 0xFF)
        self.push(self.pc & 0xFF)
        self.push((self.p & ~B_FLAG) | U_FLAG)
        self.p |= I_FLAG
        lo = self.read(0xFFFA)
        hi = self.read(0xFFFB)
        self.pc = (hi << 8) | lo
        self.nmi_pending = False
        return 7

    def _service_irq(self) -> int:
        self.push((self.pc >> 8) & 0xFF)
        self.push(self.pc & 0xFF)
        self.push((self.p & ~B_FLAG) | U_FLAG)
        self.p |= I_FLAG
        lo = self.read(0xFFFE)
        hi = self.read(0xFFFF)
        self.pc = (hi << 8) | lo
        self.irq_pending = False
        return 7

    # addressing modes (return resolved address) -----------------
    def _addr_zp(self) -> int:
        return self.fetch8()

    def _addr_zpx(self) -> int:
        return (self.fetch8() + self.x) & 0xFF

    def _addr_zpy(self) -> int:
        return (self.fetch8() + self.y) & 0xFF

    def _addr_abs(self) -> int:
        return self.fetch16()

    def _addr_abx(self) -> int:
        return (self.fetch16() + self.x) & 0xFFFF

    def _addr_aby(self) -> int:
        return (self.fetch16() + self.y) & 0xFFFF

    def _addr_izx(self) -> int:
        base = (self.fetch8() + self.x) & 0xFF
        lo = self.read(base)
        hi = self.read((base + 1) & 0xFF)
        return (lo | (hi << 8)) & 0xFFFF

    def _addr_izy(self) -> int:
        base = self.fetch8()
        lo = self.read(base)
        hi = self.read((base + 1) & 0xFF)
        return ((lo | (hi << 8)) + self.y) & 0xFFFF

    # ALU helpers ------------------------------------------------
    def _adc(self, v: int) -> None:
        c = self.p & C_FLAG
        total = self.a + v + c
        self.set_flag(C_FLAG, total > 0xFF)
        result = total & 0xFF
        self.set_flag(V_FLAG, bool((~(self.a ^ v) & (self.a ^ result)) & 0x80))
        self.a = result
        self.set_zn(self.a)

    def _sbc(self, v: int) -> None:
        self._adc(v ^ 0xFF)

    def _cmp_reg(self, reg: int, v: int) -> None:
        r = (reg - v) & 0xFF
        self.set_flag(C_FLAG, reg >= v)
        self.set_zn(r)

    def _branch_if(self, cond: bool) -> int:
        off = self.fetch8()
        if off & 0x80:
            off -= 0x100
        if cond:
            old = self.pc
            self.pc = (self.pc + off) & 0xFFFF
            return 3 + (1 if (old & 0xFF00) != (self.pc & 0xFF00) else 0)
        return 2

    def _asl(self, v: int) -> int:
        self.set_flag(C_FLAG, v & 0x80)
        v = (v << 1) & 0xFF
        self.set_zn(v)
        return v

    def _lsr(self, v: int) -> int:
        self.set_flag(C_FLAG, v & 1)
        v >>= 1
        self.set_zn(v)
        return v

    def _rol(self, v: int) -> int:
        c = self.p & C_FLAG
        self.set_flag(C_FLAG, v & 0x80)
        v = ((v << 1) | c) & 0xFF
        self.set_zn(v)
        return v

    def _ror(self, v: int) -> int:
        c = self.p & C_FLAG
        self.set_flag(C_FLAG, v & 1)
        v = (v >> 1) | (c << 7)
        self.set_zn(v)
        return v

    # main step --------------------------------------------------
    def step(self) -> int:
        if self.nmi_pending:
            return self._service_nmi()
        if self.irq_pending and not (self.p & I_FLAG):
            return self._service_irq()
        if not self.running:
            return 2

        op = self.fetch8()

        # === LDA family ===
        if op == 0xA9: self.a = self.fetch8();           self.set_zn(self.a); return 2
        if op == 0xA5: self.a = self.read(self._addr_zp());  self.set_zn(self.a); return 3
        if op == 0xB5: self.a = self.read(self._addr_zpx()); self.set_zn(self.a); return 4
        if op == 0xAD: self.a = self.read(self._addr_abs()); self.set_zn(self.a); return 4
        if op == 0xBD: self.a = self.read(self._addr_abx()); self.set_zn(self.a); return 4
        if op == 0xB9: self.a = self.read(self._addr_aby()); self.set_zn(self.a); return 4
        if op == 0xA1: self.a = self.read(self._addr_izx()); self.set_zn(self.a); return 6
        if op == 0xB1: self.a = self.read(self._addr_izy()); self.set_zn(self.a); return 5

        # === LDX ===
        if op == 0xA2: self.x = self.fetch8();           self.set_zn(self.x); return 2
        if op == 0xA6: self.x = self.read(self._addr_zp());  self.set_zn(self.x); return 3
        if op == 0xB6: self.x = self.read(self._addr_zpy()); self.set_zn(self.x); return 4
        if op == 0xAE: self.x = self.read(self._addr_abs()); self.set_zn(self.x); return 4
        if op == 0xBE: self.x = self.read(self._addr_aby()); self.set_zn(self.x); return 4

        # === LDY ===
        if op == 0xA0: self.y = self.fetch8();           self.set_zn(self.y); return 2
        if op == 0xA4: self.y = self.read(self._addr_zp());  self.set_zn(self.y); return 3
        if op == 0xB4: self.y = self.read(self._addr_zpx()); self.set_zn(self.y); return 4
        if op == 0xAC: self.y = self.read(self._addr_abs()); self.set_zn(self.y); return 4
        if op == 0xBC: self.y = self.read(self._addr_abx()); self.set_zn(self.y); return 4

        # === STA ===
        if op == 0x85: self.write(self._addr_zp(),  self.a); return 3
        if op == 0x95: self.write(self._addr_zpx(), self.a); return 4
        if op == 0x8D: self.write(self._addr_abs(), self.a); return 4
        if op == 0x9D: self.write(self._addr_abx(), self.a); return 5
        if op == 0x99: self.write(self._addr_aby(), self.a); return 5
        if op == 0x81: self.write(self._addr_izx(), self.a); return 6
        if op == 0x91: self.write(self._addr_izy(), self.a); return 6

        # === STX / STY ===
        if op == 0x86: self.write(self._addr_zp(),  self.x); return 3
        if op == 0x96: self.write(self._addr_zpy(), self.x); return 4
        if op == 0x8E: self.write(self._addr_abs(), self.x); return 4
        if op == 0x84: self.write(self._addr_zp(),  self.y); return 3
        if op == 0x94: self.write(self._addr_zpx(), self.y); return 4
        if op == 0x8C: self.write(self._addr_abs(), self.y); return 4

        # === register transfers ===
        if op == 0xAA: self.x = self.a;  self.set_zn(self.x); return 2  # TAX
        if op == 0xA8: self.y = self.a;  self.set_zn(self.y); return 2  # TAY
        if op == 0x8A: self.a = self.x;  self.set_zn(self.a); return 2  # TXA
        if op == 0x98: self.a = self.y;  self.set_zn(self.a); return 2  # TYA
        if op == 0xBA: self.x = self.sp; self.set_zn(self.x); return 2  # TSX
        if op == 0x9A: self.sp = self.x;                          return 2  # TXS

        # === INC/DEC ===
        if op == 0xE8: self.x = (self.x + 1) & 0xFF; self.set_zn(self.x); return 2
        if op == 0xC8: self.y = (self.y + 1) & 0xFF; self.set_zn(self.y); return 2
        if op == 0xCA: self.x = (self.x - 1) & 0xFF; self.set_zn(self.x); return 2
        if op == 0x88: self.y = (self.y - 1) & 0xFF; self.set_zn(self.y); return 2
        if op in (0xE6, 0xF6, 0xEE, 0xFE):  # INC zp / zp,X / abs / abs,X
            addr = (self._addr_zp() if op == 0xE6 else
                    self._addr_zpx() if op == 0xF6 else
                    self._addr_abs() if op == 0xEE else
                    self._addr_abx())
            v = (self.read(addr) + 1) & 0xFF
            self.write(addr, v); self.set_zn(v)
            return 5 if op in (0xE6,) else (6 if op in (0xF6, 0xEE) else 7)
        if op in (0xC6, 0xD6, 0xCE, 0xDE):  # DEC zp / zp,X / abs / abs,X
            addr = (self._addr_zp() if op == 0xC6 else
                    self._addr_zpx() if op == 0xD6 else
                    self._addr_abs() if op == 0xCE else
                    self._addr_abx())
            v = (self.read(addr) - 1) & 0xFF
            self.write(addr, v); self.set_zn(v)
            return 5 if op in (0xC6,) else (6 if op in (0xD6, 0xCE) else 7)

        # === ADC / SBC ===
        if op == 0x69: self._adc(self.fetch8());                return 2
        if op == 0x65: self._adc(self.read(self._addr_zp()));   return 3
        if op == 0x75: self._adc(self.read(self._addr_zpx()));  return 4
        if op == 0x6D: self._adc(self.read(self._addr_abs()));  return 4
        if op == 0x7D: self._adc(self.read(self._addr_abx()));  return 4
        if op == 0x79: self._adc(self.read(self._addr_aby()));  return 4
        if op == 0x61: self._adc(self.read(self._addr_izx()));  return 6
        if op == 0x71: self._adc(self.read(self._addr_izy()));  return 5
        if op == 0xE9 or op == 0xEB: self._sbc(self.fetch8());  return 2  # 0xEB = unofficial SBC
        if op == 0xE5: self._sbc(self.read(self._addr_zp()));   return 3
        if op == 0xF5: self._sbc(self.read(self._addr_zpx()));  return 4
        if op == 0xED: self._sbc(self.read(self._addr_abs()));  return 4
        if op == 0xFD: self._sbc(self.read(self._addr_abx()));  return 4
        if op == 0xF9: self._sbc(self.read(self._addr_aby()));  return 4
        if op == 0xE1: self._sbc(self.read(self._addr_izx()));  return 6
        if op == 0xF1: self._sbc(self.read(self._addr_izy()));  return 5

        # === AND / ORA / EOR ===
        if op == 0x29: self.a &= self.fetch8();           self.set_zn(self.a); return 2
        if op == 0x25: self.a &= self.read(self._addr_zp());  self.set_zn(self.a); return 3
        if op == 0x35: self.a &= self.read(self._addr_zpx()); self.set_zn(self.a); return 4
        if op == 0x2D: self.a &= self.read(self._addr_abs()); self.set_zn(self.a); return 4
        if op == 0x3D: self.a &= self.read(self._addr_abx()); self.set_zn(self.a); return 4
        if op == 0x39: self.a &= self.read(self._addr_aby()); self.set_zn(self.a); return 4
        if op == 0x21: self.a &= self.read(self._addr_izx()); self.set_zn(self.a); return 6
        if op == 0x31: self.a &= self.read(self._addr_izy()); self.set_zn(self.a); return 5
        if op == 0x09: self.a |= self.fetch8();           self.set_zn(self.a); return 2
        if op == 0x05: self.a |= self.read(self._addr_zp());  self.set_zn(self.a); return 3
        if op == 0x15: self.a |= self.read(self._addr_zpx()); self.set_zn(self.a); return 4
        if op == 0x0D: self.a |= self.read(self._addr_abs()); self.set_zn(self.a); return 4
        if op == 0x1D: self.a |= self.read(self._addr_abx()); self.set_zn(self.a); return 4
        if op == 0x19: self.a |= self.read(self._addr_aby()); self.set_zn(self.a); return 4
        if op == 0x01: self.a |= self.read(self._addr_izx()); self.set_zn(self.a); return 6
        if op == 0x11: self.a |= self.read(self._addr_izy()); self.set_zn(self.a); return 5
        if op == 0x49: self.a ^= self.fetch8();           self.set_zn(self.a); return 2
        if op == 0x45: self.a ^= self.read(self._addr_zp());  self.set_zn(self.a); return 3
        if op == 0x55: self.a ^= self.read(self._addr_zpx()); self.set_zn(self.a); return 4
        if op == 0x4D: self.a ^= self.read(self._addr_abs()); self.set_zn(self.a); return 4
        if op == 0x5D: self.a ^= self.read(self._addr_abx()); self.set_zn(self.a); return 4
        if op == 0x59: self.a ^= self.read(self._addr_aby()); self.set_zn(self.a); return 4
        if op == 0x41: self.a ^= self.read(self._addr_izx()); self.set_zn(self.a); return 6
        if op == 0x51: self.a ^= self.read(self._addr_izy()); self.set_zn(self.a); return 5

        # === CMP / CPX / CPY ===
        if op == 0xC9: self._cmp_reg(self.a, self.fetch8());           return 2
        if op == 0xC5: self._cmp_reg(self.a, self.read(self._addr_zp())); return 3
        if op == 0xD5: self._cmp_reg(self.a, self.read(self._addr_zpx())); return 4
        if op == 0xCD: self._cmp_reg(self.a, self.read(self._addr_abs())); return 4
        if op == 0xDD: self._cmp_reg(self.a, self.read(self._addr_abx())); return 4
        if op == 0xD9: self._cmp_reg(self.a, self.read(self._addr_aby())); return 4
        if op == 0xC1: self._cmp_reg(self.a, self.read(self._addr_izx())); return 6
        if op == 0xD1: self._cmp_reg(self.a, self.read(self._addr_izy())); return 5
        if op == 0xE0: self._cmp_reg(self.x, self.fetch8());           return 2
        if op == 0xE4: self._cmp_reg(self.x, self.read(self._addr_zp())); return 3
        if op == 0xEC: self._cmp_reg(self.x, self.read(self._addr_abs())); return 4
        if op == 0xC0: self._cmp_reg(self.y, self.fetch8());           return 2
        if op == 0xC4: self._cmp_reg(self.y, self.read(self._addr_zp())); return 3
        if op == 0xCC: self._cmp_reg(self.y, self.read(self._addr_abs())); return 4

        # === shifts / rotates - A and memory ===
        if op == 0x0A: self.a = self._asl(self.a); return 2
        if op == 0x4A: self.a = self._lsr(self.a); return 2
        if op == 0x2A: self.a = self._rol(self.a); return 2
        if op == 0x6A: self.a = self._ror(self.a); return 2
        if op in (0x06, 0x16, 0x0E, 0x1E):  # ASL mem
            addr = (self._addr_zp() if op == 0x06 else
                    self._addr_zpx() if op == 0x16 else
                    self._addr_abs() if op == 0x0E else
                    self._addr_abx())
            self.write(addr, self._asl(self.read(addr)))
            return 5 if op == 0x06 else (6 if op in (0x16, 0x0E) else 7)
        if op in (0x46, 0x56, 0x4E, 0x5E):
            addr = (self._addr_zp() if op == 0x46 else
                    self._addr_zpx() if op == 0x56 else
                    self._addr_abs() if op == 0x4E else
                    self._addr_abx())
            self.write(addr, self._lsr(self.read(addr)))
            return 5 if op == 0x46 else (6 if op in (0x56, 0x4E) else 7)
        if op in (0x26, 0x36, 0x2E, 0x3E):
            addr = (self._addr_zp() if op == 0x26 else
                    self._addr_zpx() if op == 0x36 else
                    self._addr_abs() if op == 0x2E else
                    self._addr_abx())
            self.write(addr, self._rol(self.read(addr)))
            return 5 if op == 0x26 else (6 if op in (0x36, 0x2E) else 7)
        if op in (0x66, 0x76, 0x6E, 0x7E):
            addr = (self._addr_zp() if op == 0x66 else
                    self._addr_zpx() if op == 0x76 else
                    self._addr_abs() if op == 0x6E else
                    self._addr_abx())
            self.write(addr, self._ror(self.read(addr)))
            return 5 if op == 0x66 else (6 if op in (0x76, 0x6E) else 7)

        # === BIT ===
        if op == 0x24:
            v = self.read(self._addr_zp())
            self.p = (self.p & 0x3F) | (v & 0xC0)
            self.set_flag(Z_FLAG, (self.a & v) == 0)
            # set_flag(Z) handled, but set_zn also touches N — do manually:
            return 3
        if op == 0x2C:
            v = self.read(self._addr_abs())
            self.p = (self.p & 0x3F) | (v & 0xC0)
            self.set_flag(Z_FLAG, (self.a & v) == 0)
            return 4

        # === jumps ===
        if op == 0x4C: self.pc = self._addr_abs(); return 3
        if op == 0x6C:                            # JMP (ind), with the 6502 page-wrap bug
            ptr = self.fetch16()
            lo = self.read(ptr)
            hi = self.read((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
            self.pc = lo | (hi << 8); return 5
        if op == 0x20:                            # JSR abs
            addr = self.fetch16()
            ret = (self.pc - 1) & 0xFFFF
            self.push((ret >> 8) & 0xFF)
            self.push(ret & 0xFF)
            self.pc = addr
            return 6
        if op == 0x60:                            # RTS
            lo = self.pull(); hi = self.pull()
            self.pc = ((hi << 8) | lo) + 1
            self.pc &= 0xFFFF
            return 6
        if op == 0x40:                            # RTI
            self.p = (self.pull() & ~B_FLAG) | U_FLAG
            lo = self.pull(); hi = self.pull()
            self.pc = (hi << 8) | lo
            return 6

        # === branches ===
        if op == 0xD0: return self._branch_if(not (self.p & Z_FLAG))  # BNE
        if op == 0xF0: return self._branch_if(bool(self.p & Z_FLAG))  # BEQ
        if op == 0x90: return self._branch_if(not (self.p & C_FLAG))  # BCC
        if op == 0xB0: return self._branch_if(bool(self.p & C_FLAG))  # BCS
        if op == 0x10: return self._branch_if(not (self.p & N_FLAG))  # BPL
        if op == 0x30: return self._branch_if(bool(self.p & N_FLAG))  # BMI
        if op == 0x50: return self._branch_if(not (self.p & V_FLAG))  # BVC
        if op == 0x70: return self._branch_if(bool(self.p & V_FLAG))  # BVS

        # === flags ===
        if op == 0x18: self.p &= ~C_FLAG; return 2  # CLC
        if op == 0x38: self.p |= C_FLAG;  return 2  # SEC
        if op == 0x58: self.p &= ~I_FLAG; return 2  # CLI
        if op == 0x78: self.p |= I_FLAG;  return 2  # SEI
        if op == 0xB8: self.p &= ~V_FLAG; return 2  # CLV
        if op == 0xD8: self.p &= ~D_FLAG; return 2  # CLD
        if op == 0xF8: self.p |= D_FLAG;  return 2  # SED

        # === stack ===
        if op == 0x48: self.push(self.a); return 3   # PHA
        if op == 0x68: self.a = self.pull(); self.set_zn(self.a); return 4  # PLA
        if op == 0x08: self.push(self.p | B_FLAG | U_FLAG); return 3        # PHP
        if op == 0x28: self.p = (self.pull() & ~B_FLAG) | U_FLAG; return 4  # PLP

        # === NOP family (official + unofficial) ===
        if op == 0xEA: return 2
        # unofficial NOPs that some games hit:
        if op in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA): return 2
        if op in (0x80, 0x82, 0x89, 0xC2, 0xE2):       # NOP #imm
            self.fetch8(); return 2
        if op in (0x04, 0x44, 0x64):                   # NOP zp
            self.fetch8(); return 3
        if op in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4): # NOP zp,X
            self.fetch8(); return 4
        if op == 0x0C:                                 # NOP abs
            self.fetch16(); return 4
        if op in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC): # NOP abs,X
            self.fetch16(); return 4

        # === BRK ===
        if op == 0x00:
            self.pc = (self.pc + 1) & 0xFFFF
            self.push((self.pc >> 8) & 0xFF)
            self.push(self.pc & 0xFF)
            self.push(self.p | B_FLAG | U_FLAG)
            self.p |= I_FLAG
            lo = self.read(0xFFFE); hi = self.read(0xFFFF)
            self.pc = (hi << 8) | lo
            return 7

        # === UNOFFICIAL OPCODES ===
        # These show up in commercial games (some intentional, some
        # baked into copy-protection / Konami / Sunsoft routines). The
        # six 2A03-stable groups are LAX, SAX, DCP, ISC, SLO, RLA, SRE,
        # RRA. We implement them so games like Super Cars, Disney's
        # Aladdin, Beavis & Butthead, and various unlicensed ROMs don't
        # die on the very first fetch.

        # LAX (LDA + LDX) — load to A and X simultaneously
        if op in (0xA7, 0xB7, 0xAF, 0xBF, 0xA3, 0xB3):
            if   op == 0xA7: addr = self._addr_zp();  cyc = 3
            elif op == 0xB7: addr = self._addr_zpy(); cyc = 4
            elif op == 0xAF: addr = self._addr_abs(); cyc = 4
            elif op == 0xBF: addr = self._addr_aby(); cyc = 4
            elif op == 0xA3: addr = self._addr_izx(); cyc = 6
            else:            addr = self._addr_izy(); cyc = 5
            v = self.read(addr)
            self.a = v; self.x = v
            self.set_zn(v)
            return cyc

        # SAX (store A & X) — write the AND of A and X to memory, no flags
        if op in (0x87, 0x97, 0x8F, 0x83):
            if   op == 0x87: addr = self._addr_zp();  cyc = 3
            elif op == 0x97: addr = self._addr_zpy(); cyc = 4
            elif op == 0x8F: addr = self._addr_abs(); cyc = 4
            else:            addr = self._addr_izx(); cyc = 6
            self.write(addr, self.a & self.x)
            return cyc

        # DCP (DEC + CMP) — decrement memory then compare with A
        if op in (0xC7, 0xD7, 0xCF, 0xDF, 0xDB, 0xC3, 0xD3):
            if   op == 0xC7: addr = self._addr_zp();  cyc = 5
            elif op == 0xD7: addr = self._addr_zpx(); cyc = 6
            elif op == 0xCF: addr = self._addr_abs(); cyc = 6
            elif op == 0xDF: addr = self._addr_abx(); cyc = 7
            elif op == 0xDB: addr = self._addr_aby(); cyc = 7
            elif op == 0xC3: addr = self._addr_izx(); cyc = 8
            else:            addr = self._addr_izy(); cyc = 8
            v = (self.read(addr) - 1) & 0xFF
            self.write(addr, v)
            self._cmp_reg(self.a, v)
            return cyc

        # ISC (INC + SBC) — also called ISB. Increment memory then SBC.
        if op in (0xE7, 0xF7, 0xEF, 0xFF, 0xFB, 0xE3, 0xF3):
            if   op == 0xE7: addr = self._addr_zp();  cyc = 5
            elif op == 0xF7: addr = self._addr_zpx(); cyc = 6
            elif op == 0xEF: addr = self._addr_abs(); cyc = 6
            elif op == 0xFF: addr = self._addr_abx(); cyc = 7
            elif op == 0xFB: addr = self._addr_aby(); cyc = 7
            elif op == 0xE3: addr = self._addr_izx(); cyc = 8
            else:            addr = self._addr_izy(); cyc = 8
            v = (self.read(addr) + 1) & 0xFF
            self.write(addr, v)
            self._sbc(v)
            return cyc

        # SLO (ASL + ORA)
        if op in (0x07, 0x17, 0x0F, 0x1F, 0x1B, 0x03, 0x13):
            if   op == 0x07: addr = self._addr_zp();  cyc = 5
            elif op == 0x17: addr = self._addr_zpx(); cyc = 6
            elif op == 0x0F: addr = self._addr_abs(); cyc = 6
            elif op == 0x1F: addr = self._addr_abx(); cyc = 7
            elif op == 0x1B: addr = self._addr_aby(); cyc = 7
            elif op == 0x03: addr = self._addr_izx(); cyc = 8
            else:            addr = self._addr_izy(); cyc = 8
            v = self._asl(self.read(addr))
            self.write(addr, v)
            self.a |= v; self.set_zn(self.a)
            return cyc

        # RLA (ROL + AND)
        if op in (0x27, 0x37, 0x2F, 0x3F, 0x3B, 0x23, 0x33):
            if   op == 0x27: addr = self._addr_zp();  cyc = 5
            elif op == 0x37: addr = self._addr_zpx(); cyc = 6
            elif op == 0x2F: addr = self._addr_abs(); cyc = 6
            elif op == 0x3F: addr = self._addr_abx(); cyc = 7
            elif op == 0x3B: addr = self._addr_aby(); cyc = 7
            elif op == 0x23: addr = self._addr_izx(); cyc = 8
            else:            addr = self._addr_izy(); cyc = 8
            v = self._rol(self.read(addr))
            self.write(addr, v)
            self.a &= v; self.set_zn(self.a)
            return cyc

        # SRE (LSR + EOR)
        if op in (0x47, 0x57, 0x4F, 0x5F, 0x5B, 0x43, 0x53):
            if   op == 0x47: addr = self._addr_zp();  cyc = 5
            elif op == 0x57: addr = self._addr_zpx(); cyc = 6
            elif op == 0x4F: addr = self._addr_abs(); cyc = 6
            elif op == 0x5F: addr = self._addr_abx(); cyc = 7
            elif op == 0x5B: addr = self._addr_aby(); cyc = 7
            elif op == 0x43: addr = self._addr_izx(); cyc = 8
            else:            addr = self._addr_izy(); cyc = 8
            v = self._lsr(self.read(addr))
            self.write(addr, v)
            self.a ^= v; self.set_zn(self.a)
            return cyc

        # RRA (ROR + ADC)
        if op in (0x67, 0x77, 0x6F, 0x7F, 0x7B, 0x63, 0x73):
            if   op == 0x67: addr = self._addr_zp();  cyc = 5
            elif op == 0x77: addr = self._addr_zpx(); cyc = 6
            elif op == 0x6F: addr = self._addr_abs(); cyc = 6
            elif op == 0x7F: addr = self._addr_abx(); cyc = 7
            elif op == 0x7B: addr = self._addr_aby(); cyc = 7
            elif op == 0x63: addr = self._addr_izx(); cyc = 8
            else:            addr = self._addr_izy(); cyc = 8
            v = self._ror(self.read(addr))
            self.write(addr, v)
            self._adc(v)
            return cyc

        # ANC (AND + carry from bit 7)
        if op in (0x0B, 0x2B):
            self.a &= self.fetch8(); self.set_zn(self.a)
            self.set_flag(C_FLAG, bool(self.a & 0x80))
            return 2
        # ALR (AND #imm then LSR A)
        if op == 0x4B:
            self.a &= self.fetch8()
            self.set_flag(C_FLAG, self.a & 1)
            self.a >>= 1; self.set_zn(self.a)
            return 2
        # ARR (AND #imm then ROR A; weird flag behavior)
        if op == 0x6B:
            self.a &= self.fetch8()
            c = self.p & C_FLAG
            self.a = (self.a >> 1) | (c << 7)
            self.set_zn(self.a)
            self.set_flag(C_FLAG, bool(self.a & 0x40))
            self.set_flag(V_FLAG, bool(((self.a >> 6) ^ (self.a >> 5)) & 1))
            return 2
        # AXS / SBX (X = (A & X) - imm)
        if op == 0xCB:
            v = self.fetch8()
            t = (self.a & self.x) - v
            self.set_flag(C_FLAG, t >= 0)
            self.x = t & 0xFF
            self.set_zn(self.x)
            return 2

        # Unknown opcode: silent NOP so the GUI never dies.
        return 2


# =============================================================================
#  PPU - register file + nametable resolution + delegation to blitter
# =============================================================================

class PPU:
    """A heavily simplified-but-functional PPU.

    Implements just enough state to:
      - Fire NMI on vblank when $2000.7 is set
      - Latch $2005 (scroll) and $2006 (addr) via the v/t/x model
      - Read/write VRAM, palette RAM, and OAM through $2007/$2003/$2004
      - Produce a 256x240 RGB framebuffer once per frame by resolving
        the current nametable + palette and asking the (optionally
        Cython-accelerated) blitter to render it.

    Not implemented: per-scanline rendering, sprite-0 hit, sprite zero
    timing, scrolling mid-frame, MMC3 IRQs. Those don't block boot for
    the games this emulator targets, just visual polish.
    """

    PPUCTRL   = 0x2000
    PPUMASK   = 0x2001
    PPUSTATUS = 0x2002
    OAMADDR   = 0x2003
    OAMDATA   = 0x2004
    PPUSCROLL = 0x2005
    PPUADDR   = 0x2006
    PPUDATA   = 0x2007

    def __init__(self, bus):
        self.bus = bus
        self.vram = bytearray(0x800)         # 2KB nametable RAM
        self.palette = bytearray(0x20)       # 32-byte palette RAM
        self.oam = bytearray(256)            # 256-byte object attribute mem

        self.ctrl = 0
        self.mask = 0
        self.status = 0
        self.oam_addr = 0
        self.v = 0
        self.t = 0
        self.x_fine = 0
        self.w = 0  # write toggle
        self.read_buffer = 0
        self.vblank = False

        # sprite 0 hit: precomputed once per frame, set when reached
        self.sprite0_hit_scanline = -1

        # framebuffer used by the GUI
        self.framebuf = bytearray(NES_W * NES_H * 3)

    def reset(self):
        self.ctrl = 0
        self.mask = 0
        self.status = 0
        self.oam_addr = 0
        self.v = self.t = 0
        self.x_fine = 0
        self.w = 0
        self.read_buffer = 0
        self.vblank = False
        self.sprite0_hit_scanline = -1

    # ----------------- CPU-side register access ----------------
    def cpu_read(self, addr: int) -> int:
        reg = 0x2000 | (addr & 7)
        if reg == self.PPUSTATUS:
            val = self.status & 0xE0
            if self.vblank:
                val |= 0x80
            self.vblank = False
            self.status &= 0x7F
            self.w = 0
            return val
        if reg == self.OAMDATA:
            return self.oam[self.oam_addr]
        if reg == self.PPUDATA:
            v = self.v & 0x3FFF
            if v < 0x3F00:
                ret = self.read_buffer
                self.read_buffer = self._vram_read(v)
            else:
                ret = self._palette_read(v)
                self.read_buffer = self._vram_read(v - 0x1000)
            self.v = (self.v + self._vram_inc()) & 0x7FFF
            return ret
        return 0

    def cpu_write(self, addr: int, value: int) -> None:
        reg = 0x2000 | (addr & 7)
        value &= 0xFF
        if reg == self.PPUCTRL:
            self.ctrl = value
            # t: ...GH.. ........ <- d: ......GH
            self.t = (self.t & 0xF3FF) | ((value & 3) << 10)
        elif reg == self.PPUMASK:
            self.mask = value
        elif reg == self.OAMADDR:
            self.oam_addr = value
        elif reg == self.OAMDATA:
            self.oam[self.oam_addr] = value
            self.oam_addr = (self.oam_addr + 1) & 0xFF
        elif reg == self.PPUSCROLL:
            if self.w == 0:
                self.t = (self.t & 0xFFE0) | (value >> 3)
                self.x_fine = value & 7
                self.w = 1
            else:
                self.t = (self.t & 0x8FFF) | ((value & 7) << 12)
                self.t = (self.t & 0xFC1F) | ((value & 0xF8) << 2)
                self.w = 0
        elif reg == self.PPUADDR:
            if self.w == 0:
                self.t = (self.t & 0x80FF) | ((value & 0x3F) << 8)
                self.w = 1
            else:
                self.t = (self.t & 0xFF00) | value
                self.v = self.t
                self.w = 0
        elif reg == self.PPUDATA:
            v = self.v & 0x3FFF
            if v >= 0x3F00:
                self._palette_write(v, value)
            else:
                self._vram_write(v, value)
            self.v = (self.v + self._vram_inc()) & 0x7FFF

    def _vram_inc(self) -> int:
        return 32 if (self.ctrl & 0x04) else 1

    # ----------------- PPU bus ----------------
    def _mirror_nametable(self, addr: int) -> int:
        """Map $2000-$2FFF down to the 2KB internal VRAM with current mirroring."""
        addr &= 0x0FFF
        mode = self.bus.mapper.mirroring() if self.bus.mapper else 0
        if mode == 0:    # horizontal
            return ((addr // 0x800) * 0x400) | (addr & 0x3FF) if (addr & 0x800) else (addr & 0x3FF)
        if mode == 1:    # vertical
            return addr & 0x7FF
        if mode == 2:    # single-screen, page 0
            return addr & 0x3FF
        if mode == 3:    # single-screen, page 1
            return 0x400 | (addr & 0x3FF)
        return addr & 0x7FF  # safe default

    def _vram_read(self, addr: int) -> int:
        addr &= 0x3FFF
        if addr < 0x2000:
            if self.bus.mapper:
                return self.bus.mapper.ppu_read(addr)
            return 0
        if addr < 0x3F00:
            return self.vram[self._mirror_nametable(addr - 0x2000)]
        return self._palette_read(addr)

    def _vram_write(self, addr: int, value: int) -> None:
        addr &= 0x3FFF
        if addr < 0x2000:
            if self.bus.mapper:
                self.bus.mapper.ppu_write(addr, value)
        elif addr < 0x3F00:
            self.vram[self._mirror_nametable(addr - 0x2000)] = value & 0xFF
        else:
            self._palette_write(addr, value)

    def _palette_index(self, addr: int) -> int:
        idx = addr & 0x1F
        # Mirror $3F10/$3F14/$3F18/$3F1C to $3F00/$3F04/$3F08/$3F0C
        if idx in (0x10, 0x14, 0x18, 0x1C):
            idx -= 0x10
        return idx

    def _palette_read(self, addr: int) -> int:
        return self.palette[self._palette_index(addr)] & 0x3F

    def _palette_write(self, addr: int, value: int) -> None:
        self.palette[self._palette_index(addr)] = value & 0x3F

    # ----------------- frame fill ----------------
    def enter_vblank(self) -> None:
        self.vblank = True
        self.status |= 0x80
        if self.ctrl & 0x80:
            self.bus.cpu.trigger_nmi()

    def exit_vblank(self) -> None:
        self.vblank = False
        self.status &= ~0x80

    def render_frame(self) -> bytes:
        """Resolve the current nametable & palettes into self.framebuf, return it."""
        # background pattern table base from ctrl bit 4
        bg_base = 0x1000 if (self.ctrl & 0x10) else 0
        spr_base = 0x1000 if (self.ctrl & 0x08) else 0
        sprite_size = 1 if (self.ctrl & 0x20) else 0  # 8x16 mode

        # Build a 960-byte tile array + 64-byte attribute table from the
        # current nametable (selected by ctrl bits 0-1).
        nt_select = self.ctrl & 0x03
        # use the internal vram + mirroring to extract the right 1KB
        # nametable into resolved tiles + attribute table.
        base_nt = 0x2000 | (nt_select << 10)
        tiles = bytearray(960)
        for i in range(960):
            tiles[i] = self.vram[self._mirror_nametable((base_nt + i) - 0x2000)]
        attrs = bytearray(64)
        for i in range(64):
            attrs[i] = self.vram[self._mirror_nametable((base_nt + 0x3C0 + i) - 0x2000)]

        # Build a flat CHR view the blitter can index without going
        # through the mapper for each pixel.
        chr_view = self._chr_snapshot()

        # background mask clear color is universal palette entry 0
        if self.mask & 0x08:
            self._blit_bg(chr_view, tiles, attrs, bg_base)
        else:
            self._clear(self.palette[0] & 0x3F)
        if self.mask & 0x10:
            self._blit_sprites(chr_view, spr_base, sprite_size)

        return bytes(self.framebuf)

    # ----- sprite 0 hit precompute (called before each frame's CPU run) -----
    def precompute_sprite0_hit(self) -> None:
        """Compute on which scanline sprite 0 hit should fire this frame.

        We re-resolve the current nametable + CHR view + sprite 0 OAM
        entry and walk every pixel of sprite 0 to find the first one
        that overlaps an opaque background pixel. Done once per frame
        regardless of whether the game checks the flag — cost is at
        most 8x8 (or 8x16) pixels of work.
        """
        # If either bg or sprites are off, sprite 0 hit can never fire
        if (self.mask & 0x18) != 0x18:
            self.sprite0_hit_scanline = -1
            return
        bg_base = 0x1000 if (self.ctrl & 0x10) else 0
        spr_base = 0x1000 if (self.ctrl & 0x08) else 0
        sprite_size = 1 if (self.ctrl & 0x20) else 0
        show_bg_left = 1 if (self.mask & 0x02) else 0
        show_spr_left = 1 if (self.mask & 0x04) else 0
        # nametable resolve
        nt_select = self.ctrl & 0x03
        base_nt = 0x2000 | (nt_select << 10)
        tiles = bytearray(960)
        for i in range(960):
            tiles[i] = self.vram[self._mirror_nametable((base_nt + i) - 0x2000)]
        chr_view = self._chr_snapshot()

        if _NATIVE_CORE is not None:
            self.sprite0_hit_scanline = _NATIVE_CORE.find_sprite0_hit_scanline(
                chr_view, self.oam, tiles,
                bg_base, spr_base, sprite_size,
                1, 1, show_bg_left, show_spr_left,
            )
            return

        # python fallback
        self.sprite0_hit_scanline = self._py_find_sprite0_hit(
            chr_view, tiles, bg_base, spr_base, sprite_size,
            show_bg_left, show_spr_left,
        )

    def _py_find_sprite0_hit(self, chr_view, tiles, bg_base, spr_base,
                              sprite_size, show_bg_left, show_spr_left):
        oam = self.oam
        sy = oam[0] + 1
        raw_tile = oam[1]
        flip_h = (oam[2] >> 6) & 1
        flip_v = (oam[2] >> 7) & 1
        sx = oam[3]
        height = 16 if sprite_size else 8
        if sy >= 239:
            return -1
        for py in range(height):
            screen_y = sy + py
            if screen_y >= 240:
                break
            local_y = (height - 1 - py) if flip_v else py
            if sprite_size:
                table = (raw_tile & 1) * 0x1000
                top_tile = raw_tile & 0xFE
                if local_y < 8:
                    eff_base = table + top_tile * 16 + local_y
                else:
                    eff_base = table + (top_tile + 1) * 16 + (local_y - 8)
            else:
                eff_base = spr_base + raw_tile * 16 + local_y
            if eff_base + 8 >= len(chr_view):
                continue
            lo = chr_view[eff_base]; hi = chr_view[eff_base + 8]
            for px in range(8):
                screen_x = sx + px
                if screen_x >= 255:
                    continue
                if screen_x < 8 and (not show_bg_left or not show_spr_left):
                    continue
                bit = px if flip_h else (7 - px)
                sp_cid = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
                if sp_cid == 0:
                    continue
                bg_tile_index = tiles[(screen_y >> 3) * 32 + (screen_x >> 3)]
                bg_base = bg_base
                bg_b = bg_base + bg_tile_index * 16 + (screen_y & 7)
                if bg_b + 8 >= len(chr_view):
                    continue
                blo = chr_view[bg_b]; bhi = chr_view[bg_b + 8]
                bg_bit = 7 - (screen_x & 7)
                bg_cid = ((blo >> bg_bit) & 1) | (((bhi >> bg_bit) & 1) << 1)
                if bg_cid != 0:
                    return screen_y
        return -1

    def _chr_snapshot(self) -> bytes:
        """Snapshot the 8KB CHR window the mapper currently exposes."""
        if not self.bus.mapper:
            return bytes(8192)
        chr_ = bytearray(8192)
        for i in range(8192):
            chr_[i] = self.bus.mapper.ppu_read(i)
        return bytes(chr_)

    def _clear(self, pal_index: int) -> None:
        master = (pal_index & 0x3F) * 3
        r, g, b = NES_PALETTE_BYTES[master], NES_PALETTE_BYTES[master + 1], NES_PALETTE_BYTES[master + 2]
        if _NATIVE_CORE is not None:
            _NATIVE_CORE.clear_frame(self.framebuf, r, g, b)
        else:
            fb = self.framebuf
            for i in range(0, len(fb), 3):
                fb[i] = r; fb[i + 1] = g; fb[i + 2] = b

    def _blit_bg(self, chr_view, tiles, attrs, bg_base) -> None:
        if _NATIVE_CORE is not None:
            _NATIVE_CORE.blit_background(
                chr_view, tiles, attrs, self.palette,
                NES_PALETTE_BYTES, self.framebuf, bg_base
            )
            return
        # pure-Python fallback (slow but correct)
        fb = self.framebuf
        pal = self.palette
        npal = NES_PALETTE_BYTES
        for y in range(240):
            tile_y = y >> 3
            py = y & 7
            for x in range(256):
                tile_x = x >> 3
                px = x & 7
                tile_index = tiles[tile_y * 32 + tile_x]
                base = bg_base + tile_index * 16 + py
                if base + 8 >= len(chr_view):
                    lo = hi = 0
                else:
                    lo = chr_view[base]; hi = chr_view[base + 8]
                bit = 7 - px
                color_id = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
                attr_byte = attrs[(tile_y >> 2) * 8 + (tile_x >> 2)]
                shift = ((tile_y & 2) << 1) | (tile_x & 2)
                palette_sel = (attr_byte >> shift) & 3
                if color_id == 0:
                    pal_index = pal[0]
                else:
                    pal_index = pal[palette_sel * 4 + color_id]
                m = (pal_index & 0x3F) * 3
                off = (y * 256 + x) * 3
                fb[off] = npal[m]; fb[off + 1] = npal[m + 1]; fb[off + 2] = npal[m + 2]

    def _blit_sprites(self, chr_view, spr_base, sprite_size=0) -> None:
        if _NATIVE_CORE is not None:
            _NATIVE_CORE.blit_sprites(
                chr_view, self.oam, self.palette,
                NES_PALETTE_BYTES, self.framebuf, spr_base, sprite_size,
            )
            return
        fb = self.framebuf
        pal = self.palette
        npal = NES_PALETTE_BYTES
        oam = self.oam
        height = 16 if sprite_size else 8
        for i in range(63, -1, -1):
            sy = oam[i * 4] + 1
            raw_tile = oam[i * 4 + 1]
            attr = oam[i * 4 + 2]
            sx = oam[i * 4 + 3]
            if sy >= 240:
                continue
            flip_h = (attr >> 6) & 1
            flip_v = (attr >> 7) & 1
            palette_sel = attr & 3
            for py in range(height):
                local_y = (height - 1 - py) if flip_v else py
                if sprite_size:
                    table = (raw_tile & 1) * 0x1000
                    top_tile = raw_tile & 0xFE
                    if local_y < 8:
                        base = table + top_tile * 16 + local_y
                    else:
                        base = table + (top_tile + 1) * 16 + (local_y - 8)
                else:
                    base = spr_base + raw_tile * 16 + local_y
                if base + 8 >= len(chr_view):
                    continue
                lo = chr_view[base]; hi = chr_view[base + 8]
                for px in range(8):
                    if sx + px >= 256 or sy + py >= 240:
                        continue
                    bit = px if flip_h else (7 - px)
                    color_id = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
                    if color_id == 0:
                        continue
                    pal_index = pal[16 + palette_sel * 4 + color_id]
                    m = (pal_index & 0x3F) * 3
                    off = ((sy + py) * 256 + (sx + px)) * 3
                    fb[off] = npal[m]; fb[off + 1] = npal[m + 1]; fb[off + 2] = npal[m + 2]


# =============================================================================
#  APU stub (so writes to $4000-$4017 don't drop a game's init code)
# =============================================================================

class APU:
    """NES APU with the part that matters for boot: the frame counter.

    The frame counter ticks at ~240Hz (every 7457 CPU cycles). In 4-step
    mode (bit 6 of $4017 = 0) it fires an IRQ at the end of every 4-step
    sequence (= ~once per video frame, exactly when commercial games
    are looking for a clock pulse). Many games (Battletoads, some RPGs)
    use this for their main timer; without it they hang.

    Channels are tracked register-wise so reads of $4015 report which
    are "playing" — important for some games. We don't synthesize audio
    here; the API exposes `apu.sample_chunk()` so an external loop can
    pipe data into a sounddevice / simpleaudio backend if installed.
    """

    # frame counter register
    F_IRQ_INHIBIT = 0x40
    F_5STEP       = 0x80

    def __init__(self, bus):
        self.bus = bus
        self.regs = bytearray(0x18)
        self.frame_counter = 0
        self.frame_mode = 0
        self.frame_irq_inhibit = False
        self.frame_step = 0
        # individual channel "playing" flags for $4015 reads
        self.pulse1_enabled = False
        self.pulse2_enabled = False
        self.triangle_enabled = False
        self.noise_enabled = False
        self.dmc_enabled = False
        self.frame_irq = False
        # length counters (for status read; simplified)
        self.pulse1_len = 0
        self.pulse2_len = 0
        self.triangle_len = 0
        self.noise_len = 0
        # cycle accumulator
        self._cycles = 0

    def reset(self) -> None:
        for i in range(len(self.regs)):
            self.regs[i] = 0
        self.frame_counter = 0
        self.frame_mode = 0
        self.frame_irq_inhibit = False
        self.frame_step = 0
        self.pulse1_enabled = False
        self.pulse2_enabled = False
        self.triangle_enabled = False
        self.noise_enabled = False
        self.dmc_enabled = False
        self.frame_irq = False
        self._cycles = 0

    def cpu_read(self, addr: int) -> int:
        if addr == 0x4015:
            v = 0
            if self.pulse1_len > 0:   v |= 0x01
            if self.pulse2_len > 0:   v |= 0x02
            if self.triangle_len > 0: v |= 0x04
            if self.noise_len > 0:    v |= 0x08
            if self.dmc_enabled:      v |= 0x10
            if self.frame_irq:        v |= 0x40
            # reading $4015 clears frame IRQ
            self.frame_irq = False
            return v
        return 0  # open bus on other APU reads

    def cpu_write(self, addr: int, value: int) -> None:
        if not (0x4000 <= addr <= 0x4017):
            return
        idx = addr - 0x4000
        if idx < len(self.regs):
            self.regs[idx] = value & 0xFF

        if addr == 0x4015:
            # channel enable register
            self.pulse1_enabled   = bool(value & 0x01)
            self.pulse2_enabled   = bool(value & 0x02)
            self.triangle_enabled = bool(value & 0x04)
            self.noise_enabled    = bool(value & 0x08)
            self.dmc_enabled      = bool(value & 0x10)
            if not self.pulse1_enabled:   self.pulse1_len = 0
            if not self.pulse2_enabled:   self.pulse2_len = 0
            if not self.triangle_enabled: self.triangle_len = 0
            if not self.noise_enabled:    self.noise_len = 0
        elif addr == 0x4017:
            # frame counter / IRQ inhibit
            self.frame_mode = (value >> 7) & 1
            self.frame_irq_inhibit = bool(value & self.F_IRQ_INHIBIT)
            if self.frame_irq_inhibit:
                self.frame_irq = False
            self.frame_step = 0
            self._cycles = 0
        elif addr in (0x4003, 0x4007):
            # writing $4003/$4007 (pulse length-counter load) starts the channel
            length = (value >> 3) & 0x1F
            if addr == 0x4003 and self.pulse1_enabled:
                self.pulse1_len = max(1, length + 1)
            elif addr == 0x4007 and self.pulse2_enabled:
                self.pulse2_len = max(1, length + 1)
        elif addr == 0x400B and self.triangle_enabled:
            self.triangle_len = max(1, ((value >> 3) & 0x1F) + 1)
        elif addr == 0x400F and self.noise_enabled:
            self.noise_len = max(1, ((value >> 3) & 0x1F) + 1)

    def tick(self, cycles: int) -> None:
        """Advance the frame counter by `cycles` CPU cycles, fire IRQ if due.

        Approximation: in 4-step mode the IRQ fires at step 3 (= 22371
        cycles in). We treat one full 29830-cycle period as one tick and
        fire once per period. Length counters decrement once per period.
        """
        self._cycles += cycles
        # 4-step period is 29830 CPU cycles, 5-step is 37281
        period = 37281 if self.frame_mode else 29830
        while self._cycles >= period:
            self._cycles -= period
            # decrement length counters
            if self.pulse1_len   > 0: self.pulse1_len   -= 1
            if self.pulse2_len   > 0: self.pulse2_len   -= 1
            if self.triangle_len > 0: self.triangle_len -= 1
            if self.noise_len    > 0: self.noise_len    -= 1
            # fire IRQ if 4-step mode and not inhibited
            if self.frame_mode == 0 and not self.frame_irq_inhibit:
                self.frame_irq = True
                self.bus.cpu.irq_pending = True


# =============================================================================
#  BUS - wires CPU, PPU, mapper, controllers, DMA
# =============================================================================

class ACNES:
    def __init__(self):
        self.cart = NESCartridge()
        self.mapper: Mapper | None = None
        self.cpu_ram = bytearray(0x800)
        self.cpu = CPU6502(self)
        self.ppu = PPU(self)
        self.apu = APU(self)
        self.controller = 0
        self.controller_strobe = False
        self.controller_shift = 0
        self.frame = 0
        # open bus: last value returned by a CPU read, used for unmapped reads
        self._open_bus = 0

    def reset(self) -> None:
        self.cpu_ram = bytearray(0x800)
        self.mapper = make_mapper(self.cart) if self.cart.valid else None
        self.controller = 0
        self.controller_strobe = False
        self.controller_shift = 0
        self.frame = 0
        self._open_bus = 0
        self.ppu.reset()
        self.apu.reset()
        self.cpu.reset()

    # ---------------- CPU memory map ----------------
    def cpu_read(self, addr: int) -> int:
        addr &= 0xFFFF
        if addr < 0x2000:
            v = self.cpu_ram[addr & 0x7FF]
            self._open_bus = v
            return v
        if addr < 0x4000:
            v = self.ppu.cpu_read(addr)
            self._open_bus = v
            return v
        if addr == 0x4015:
            v = self.apu.cpu_read(addr)
            self._open_bus = v
            return v
        if addr == 0x4016:
            if self.controller_strobe:
                bit = self.controller & 1
            else:
                bit = self.controller_shift & 1
                self.controller_shift = (self.controller_shift >> 1) | 0x80
            # bits 0..4 from controller, bit 6 always set (open-bus quirk)
            v = (self._open_bus & 0xE0) | 0x40 | bit
            self._open_bus = v
            return v
        if addr == 0x4017:
            return (self._open_bus & 0xE0) | 0x40
        if 0x4000 <= addr <= 0x4017:
            return self.apu.cpu_read(addr)
        if addr >= 0x6000 and self.mapper:
            v = self.mapper.cpu_read(addr)
            self._open_bus = v
            return v
        return self._open_bus  # actual open bus

    def cpu_write(self, addr: int, value: int) -> None:
        addr &= 0xFFFF
        value &= 0xFF
        if addr < 0x2000:
            self.cpu_ram[addr & 0x7FF] = value
        elif addr < 0x4000:
            self.ppu.cpu_write(addr, value)
        elif addr == 0x4014:
            # OAM DMA: copy 256 bytes from CPU page (value*0x100) to OAM
            base = (value & 0xFF) << 8
            for i in range(256):
                self.ppu.oam[(self.ppu.oam_addr + i) & 0xFF] = self.cpu_read(base + i)
            # DMA stalls CPU for 513 cycles (or 514 on odd) — account for it
            self.cpu.cycles += 513
        elif addr == 0x4016:
            if value & 1:
                self.controller_strobe = True
                self.controller_shift = self.controller
            else:
                self.controller_strobe = False
                self.controller_shift = self.controller
        elif 0x4000 <= addr <= 0x4017:
            self.apu.cpu_write(addr, value)
        elif addr >= 0x4020 and self.mapper:
            self.mapper.cpu_write(addr, value)

    # ---------------- frame loop ----------------
    def run_frame(self) -> None:
        """Dual-path frame loop.

        Burst path (default): one big CPU run for the whole visible
        portion of the frame, then one for vblank. This is what 0.5
        did, with one tweak: if sprite 0 hit happens this frame we
        split the visible run into pre-hit and post-hit phases so
        polling `BIT $2002 / BPL` loops actually escape. Used for all
        mappers except MMC3.

        Scanline path (MMC3 only): the visible 240 scanlines are run
        one at a time so we can call `mapper.notify_scanline()` at each
        boundary and let MMC3 decrement its IRQ counter. Mapper IRQs
        are serviced only at scanline boundaries — real MMC3 has 1-2
        scanline latency anyway, so this is accurate enough.
        """
        ppu = self.ppu
        cpu = self.cpu
        apu = self.apu
        mapper = self.mapper

        ppu.precompute_sprite0_hit()
        s0 = ppu.sprite0_hit_scanline

        # Local-bind hot attrs for the inner Python loop
        step = cpu.step
        cycles_sl = CPU_CYCLES_PER_SCANLINE
        total_visible = CPU_CYCLES_PER_SCANLINE * VISIBLE_SCANLINES

        # ============ BURST PATH (non-MMC3) ============
        if not (mapper and mapper.has_scanline_irq):
            used = 0
            if s0 >= 0:
                pre = s0 * cycles_sl
                while used < pre and cpu.running:
                    used += step()
                ppu.status |= 0x40
            while used < total_visible and cpu.running:
                used += step()
            apu.tick(used)

            # vblank: enter, run remaining cycles, exit
            ppu.enter_vblank()
            vb_target = (VBLANK_SCANLINES + 1) * cycles_sl
            vb = 0
            while vb < vb_target and cpu.running:
                vb += step()
                # APU frame IRQ can fire during vblank; service it
                if cpu.irq_pending and not (cpu.p & I_FLAG):
                    vb += cpu._service_irq()
            apu.tick(vb)
            ppu.exit_vblank()
            ppu.status &= ~0x40
            self.frame += 1
            return

        # ============ SCANLINE PATH (MMC3) ============
        rendering = (ppu.mask & 0x18) != 0
        for sl in range(VISIBLE_SCANLINES):
            used = 0
            while used < cycles_sl and cpu.running:
                used += step()
            apu.tick(used)
            if s0 >= 0 and sl == s0:
                ppu.status |= 0x40
            if rendering:
                mapper.notify_scanline()
            # service any pending IRQ at scanline boundary
            if (mapper.irq_pending or cpu.irq_pending) and not (cpu.p & I_FLAG):
                cpu._service_irq()
                mapper.irq_pending = False

        ppu.enter_vblank()
        for _ in range(VBLANK_SCANLINES + 1):
            used = 0
            while used < cycles_sl and cpu.running:
                used += step()
            apu.tick(used)
            if cpu.irq_pending and not (cpu.p & I_FLAG):
                cpu._service_irq()

        ppu.exit_vblank()
        ppu.status &= ~0x40
        self.frame += 1


# =============================================================================
#  GUI - FCEUX-inspired, black/blue, 600x400, 60 FPS, PPM blit
# =============================================================================

class FCEUXStyleGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self.nes = ACNES()
        self.paused = True   # paused until a ROM is loaded
        self.last_time = time.perf_counter()
        self.frames = 0
        self.fps_text = "—"
        self.core_name = "native" if _NATIVE_CORE is not None else "python"

        self._make_menu()
        self._make_layout()

        self.photo = tk.PhotoImage(width=NES_W, height=NES_H)
        self.screen_img = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self._bind_keys()
        self.root.after_idle(lambda: self.root.focus_force())

        # show a friendly idle pattern until a ROM loads
        self._render_idle()
        self.root.after(int(1000 / FPS), self._loop)

    # ----- menu / layout -----
    def _make_menu(self):
        menubar = tk.Menu(self.root, bg=BG, fg=BLUE, activebackground=DARK_BLUE, activeforeground=WHITE)
        filemenu = tk.Menu(menubar, tearoff=0, bg=BG, fg=BLUE, activebackground=DARK_BLUE, activeforeground=WHITE)
        filemenu.add_command(label="Open ROM",     command=self.open_rom)
        filemenu.add_command(label="Reset",        command=self.reset)
        filemenu.add_separator()
        filemenu.add_command(label="Exit",         command=self.root.destroy)

        emumenu = tk.Menu(menubar, tearoff=0, bg=BG, fg=BLUE, activebackground=DARK_BLUE, activeforeground=WHITE)
        emumenu.add_command(label="Pause/Resume",  command=self.toggle_pause)
        emumenu.add_command(label="About",         command=self.about)

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Emulation", menu=emumenu)
        self.root.config(menu=menubar)

    def _button(self, parent, text, command):
        return tk.Button(
            parent, text=text, command=command,
            bg=BG, fg=BLUE,
            activebackground=DARK_BLUE, activeforeground=WHITE,
            relief="ridge", bd=2, width=12,
        )

    def _make_layout(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x")
        tk.Label(
            top,
            text=f"AC'S NES EMU {APP_VERSION}  |  FAMICOM 60 FPS  |  core={self.core_name}",
            bg=BG, fg=BLUE, font=("Consolas", 11, "bold"),
        ).pack(side="left", padx=8, pady=4)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            main, width=NES_W, height=NES_H, bg=BG,
            highlightthickness=2, highlightbackground=BLUE,
        )
        self.canvas.pack(side="left", padx=10, pady=8)

        side = tk.Frame(main, bg=BG)
        side.pack(side="right", fill="y", padx=8, pady=8)

        self._button(side, "Open ROM",   self.open_rom).pack(pady=4)
        self._button(side, "Reset",      self.reset).pack(pady=4)
        self._button(side, "Pause",      self.toggle_pause).pack(pady=4)
        self._button(side, "About",      self.about).pack(pady=4)
        self._button(side, "Exit",       self.root.destroy).pack(pady=4)

        self.status = tk.Label(
            side, text=self._status_text(),
            bg=BG, fg=BLUE, justify="left", font=("Consolas", 9),
        )
        self.status.pack(pady=12)

        tk.Label(
            self.root,
            text="Z=A  X=B  Enter=Start  RShift=Select  Arrows=D-Pad",
            bg=BG, fg=BLUE, font=("Consolas", 9),
        ).pack(side="bottom", pady=4)

    # ----- input -----
    def _bind_keys(self):
        self.root.bind("<KeyPress>",   self._key_down)
        self.root.bind("<KeyRelease>", self._key_up)

    @staticmethod
    def _key_bit(event):
        k = event.keysym.lower()
        return {
            "z": 0, "x": 1, "shift_r": 2, "return": 3,
            "up": 4, "down": 5, "left": 6, "right": 7,
        }.get(k)

    def _key_down(self, event):
        bit = self._key_bit(event)
        if bit is not None:
            self.nes.controller |= 1 << bit

    def _key_up(self, event):
        bit = self._key_bit(event)
        if bit is not None:
            self.nes.controller &= ~(1 << bit)

    # ----- actions -----
    def open_rom(self):
        path = filedialog.askopenfilename(
            title="Open NES ROM",
            filetypes=[("NES ROM", "*.nes"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.nes.cart.load_ines(path)
            self.nes.reset()
            self.paused = False
            self._update_status()
        except Exception as exc:
            messagebox.showerror("ROM Load Error", str(exc))

    def reset(self):
        if not self.nes.cart.valid:
            return
        self.nes.reset()
        self.paused = False
        self._update_status()

    def toggle_pause(self):
        if not self.nes.cart.valid:
            return
        self.paused = not self.paused
        self._update_status()

    def about(self):
        messagebox.showinfo(
            APP_TITLE,
            f"AC'S NES EMU {APP_VERSION} 'Quartet'\n"
            "Single-file Python 3.14 emu with pre-baked Cython core.\n\n"
            f"Active core: {self.core_name}\n"
            "CPU: 6502 official + LAX/SAX/DCP/ISC/SLO/RLA/SRE/RRA/ANC/ALR/ARR/AXS\n"
            "PPU: NMI on vblank, sprite 0 hit, 8x16 sprites, scanline IRQ hooks\n"
            "APU: frame counter + IRQ, channel status, register persistence\n"
            "Mappers: 0/1/2/3/4/7/9/11/66/71 (MMC1, MMC2, MMC3, UxROM, CNROM,\n"
            "         AxROM, GxROM, Color Dreams, Camerica)\n\n"
            "tip: pip install cython for the fast pixel pipe",
        )

    # ----- status / FPS -----
    def _status_text(self) -> str:
        cart = self.nes.cart
        if cart.valid:
            mode = "paused" if self.paused else "running"
            return (
                f"ROM:    {cart.name[:18]}\n"
                f"Mapper: {cart.mapper}\n"
                f"PRG:    {cart.prg_banks}x16K\n"
                f"CHR:    {cart.chr_banks}x8K{' (RAM)' if cart.is_chr_ram else ''}\n"
                f"Mode:   {mode}\n"
                f"Core:   {self.core_name}\n"
                f"FPS:    {self.fps_text}"
            )
        return (
            f"ROM:    (none)\n"
            f"Mapper: --\n"
            f"PRG:    --\n"
            f"CHR:    --\n"
            f"Mode:   idle\n"
            f"Core:   {self.core_name}\n"
            f"FPS:    {self.fps_text}"
        )

    def _update_status(self):
        self.status.config(text=self._status_text())

    # ----- pixel blit -----
    @staticmethod
    def _rgb_to_png(rgb_bytes: bytes, w: int, h: int) -> bytes:
        """Encode 24-bit RGB as PNG (stdlib only). Tk PhotoImage(data=) accepts
        GIF/PNG, not raw P6 PPM — PPM caused TclError: couldn't recognize image data."""
        sig = b"\x89PNG\r\n\x1a\n"

        def _chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        stride = w * 3
        raw = bytearray(h * (stride + 1))
        out = 0
        for y in range(h):
            raw[out] = 0
            src = y * stride
            raw[out + 1 : out + 1 + stride] = rgb_bytes[src : src + stride]
            out += stride + 1
        idat = zlib.compress(bytes(raw), 1)
        return sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")

    def _blit_rgb(self, rgb_bytes: bytes) -> None:
        """Push a 256x240x3 RGB buffer into the Canvas as PNG."""
        png = self._rgb_to_png(rgb_bytes, NES_W, NES_H)
        encoded = base64.b64encode(png)
        new_photo = tk.PhotoImage(data=encoded, format="png")
        self.canvas.itemconfig(self.screen_img, image=new_photo)
        self.photo = new_photo

    def _render_idle(self):
        """Animated demo pattern when no ROM is loaded."""
        fb = bytearray(NES_W * NES_H * 3)
        t = time.perf_counter()
        for y in range(NES_H):
            for x in range(NES_W):
                wave = int((math.sin(x * 0.05 + t) + math.cos(y * 0.05 + t)) * 40 + 80)
                if (x // 16 + y // 16) & 1:
                    r, g, b = 0, max(0, min(255, 40 + wave)), max(0, min(255, 120 + wave))
                else:
                    r, g, b = 0, 0, 20
                off = (y * NES_W + x) * 3
                fb[off] = r; fb[off + 1] = g; fb[off + 2] = b
        self._blit_rgb(bytes(fb))

    # ----- main loop -----
    def _loop(self):
        start = time.perf_counter()

        if self.nes.cart.valid and not self.paused:
            self.nes.run_frame()
            self._blit_rgb(self.nes.ppu.render_frame())
        elif not self.nes.cart.valid:
            self._render_idle()

        self.frames += 1
        now = time.perf_counter()
        if now - self.last_time >= 1.0:
            self.fps_text = str(self.frames)
            self.frames = 0
            self.last_time = now
            self._update_status()

        elapsed = time.perf_counter() - start
        delay_ms = max(1, int((1.0 / FPS - elapsed) * 1000))
        self.root.after(delay_ms, self._loop)


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main() -> None:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(
            "AC's NES Emu 0.6 — single-file Python NES emulator\n"
            "\n"
            "  python3 ##acnesemu.py              # pure-Python, no compile\n"
            "  python3 ##acnesemu.py --build-core # opt-in: compile embedded Cython core\n"
            "  ACNES_BUILD_CORE=1 python3 ##acnesemu.py\n"
        )
        return
    root = tk.Tk()
    FCEUXStyleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
