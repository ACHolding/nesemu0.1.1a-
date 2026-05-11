# ac_nes_emu_0_1.py
# AC'S NES EMU 0.1
# Python 3.14 single-file starter NES/Famicom emulator shell
# GUI: FCEUX-inspired, black background, blue text/buttons, 600x400, 60 FPS

import math
import time
import struct
import tkinter as tk
from tkinter import filedialog, messagebox

APP_TITLE = "ac's nes emu 0.1"
WIDTH, HEIGHT = 600, 400
NES_W, NES_H = 256, 240
FPS = 60

BG = "#000000"
BLUE = "#1e90ff"
DARK_BLUE = "#003366"
WHITE = "#ffffff"
GRAY = "#222222"


class NESCartridge:
    def __init__(self):
        self.prg = bytearray()
        self.chr = bytearray()
        self.mapper = 0
        self.mirror = 0
        self.valid = False
        self.name = "No ROM"

    def load_ines(self, path):
        with open(path, "rb") as f:
            data = f.read()

        if len(data) < 16 or data[:4] != b"NES\x1a":
            raise ValueError("Not a valid iNES ROM.")

        prg_banks = data[4]
        chr_banks = data[5]
        flag6 = data[6]
        flag7 = data[7]

        self.mapper = (flag6 >> 4) | (flag7 & 0xF0)
        self.mirror = flag6 & 1

        has_trainer = bool(flag6 & 0x04)
        pos = 16 + (512 if has_trainer else 0)

        prg_size = prg_banks * 16384
        chr_size = chr_banks * 8192

        self.prg = bytearray(data[pos:pos + prg_size])
        pos += prg_size

        if chr_size > 0:
            self.chr = bytearray(data[pos:pos + chr_size])
        else:
            self.chr = bytearray(8192)

        if self.mapper != 0:
            raise ValueError(
                f"Mapper {self.mapper} detected. AC'S NES EMU 0.1 currently supports mapper 0 / NROM only."
            )

        self.name = path.split("/")[-1]
        self.valid = True


class CPU6502:
    def __init__(self, nes):
        self.nes = nes
        self.a = 0
        self.x = 0
        self.y = 0
        self.sp = 0xFD
        self.pc = 0x8000
        self.p = 0x24
        self.cycles = 0
        self.running = True

    def reset(self):
        self.a = 0
        self.x = 0
        self.y = 0
        self.sp = 0xFD
        self.p = 0x24
        lo = self.read(0xFFFC)
        hi = self.read(0xFFFD)
        self.pc = (hi << 8) | lo
        if self.pc == 0:
            self.pc = 0x8000
        self.cycles = 0
        self.running = True

    def read(self, addr):
        return self.nes.cpu_read(addr & 0xFFFF)

    def write(self, addr, value):
        self.nes.cpu_write(addr & 0xFFFF, value & 0xFF)

    def push(self, v):
        self.write(0x100 + self.sp, v)
        self.sp = (self.sp - 1) & 0xFF

    def pull(self):
        self.sp = (self.sp + 1) & 0xFF
        return self.read(0x100 + self.sp)

    def fetch8(self):
        v = self.read(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        return v

    def fetch16(self):
        lo = self.fetch8()
        hi = self.fetch8()
        return lo | (hi << 8)

    def set_zn(self, v):
        v &= 0xFF
        if v == 0:
            self.p |= 0x02
        else:
            self.p &= ~0x02

        if v & 0x80:
            self.p |= 0x80
        else:
            self.p &= ~0x80

    def flag_c(self):
        return 1 if self.p & 1 else 0

    def set_c(self, cond):
        if cond:
            self.p |= 1
        else:
            self.p &= ~1

    def adc(self, v):
        total = self.a + v + self.flag_c()
        self.set_c(total > 0xFF)
        result = total & 0xFF

        if (~(self.a ^ v) & (self.a ^ result) & 0x80):
            self.p |= 0x40
        else:
            self.p &= ~0x40

        self.a = result
        self.set_zn(self.a)

    def sbc(self, v):
        self.adc(v ^ 0xFF)

    def branch(self, cond):
        off = self.fetch8()
        if off & 0x80:
            off -= 0x100
        if cond:
            self.pc = (self.pc + off) & 0xFFFF

    def step(self):
        if not self.running:
            return 1

        op = self.fetch8()

        # NOP / BRK
        if op == 0xEA:
            return 2

        if op == 0x00:
            self.running = False
            return 7

        # LDA
        if op == 0xA9:
            self.a = self.fetch8()
            self.set_zn(self.a)
            return 2
        if op == 0xA5:
            self.a = self.read(self.fetch8())
            self.set_zn(self.a)
            return 3
        if op == 0xAD:
            self.a = self.read(self.fetch16())
            self.set_zn(self.a)
            return 4
        if op == 0xBD:
            self.a = self.read((self.fetch16() + self.x) & 0xFFFF)
            self.set_zn(self.a)
            return 4
        if op == 0xB9:
            self.a = self.read((self.fetch16() + self.y) & 0xFFFF)
            self.set_zn(self.a)
            return 4

        # LDX
        if op == 0xA2:
            self.x = self.fetch8()
            self.set_zn(self.x)
            return 2
        if op == 0xA6:
            self.x = self.read(self.fetch8())
            self.set_zn(self.x)
            return 3
        if op == 0xAE:
            self.x = self.read(self.fetch16())
            self.set_zn(self.x)
            return 4

        # LDY
        if op == 0xA0:
            self.y = self.fetch8()
            self.set_zn(self.y)
            return 2
        if op == 0xA4:
            self.y = self.read(self.fetch8())
            self.set_zn(self.y)
            return 3
        if op == 0xAC:
            self.y = self.read(self.fetch16())
            self.set_zn(self.y)
            return 4

        # STA/STX/STY
        if op == 0x85:
            self.write(self.fetch8(), self.a)
            return 3
        if op == 0x8D:
            self.write(self.fetch16(), self.a)
            return 4
        if op == 0x9D:
            self.write((self.fetch16() + self.x) & 0xFFFF, self.a)
            return 5
        if op == 0x99:
            self.write((self.fetch16() + self.y) & 0xFFFF, self.a)
            return 5
        if op == 0x86:
            self.write(self.fetch8(), self.x)
            return 3
        if op == 0x8E:
            self.write(self.fetch16(), self.x)
            return 4
        if op == 0x84:
            self.write(self.fetch8(), self.y)
            return 3
        if op == 0x8C:
            self.write(self.fetch16(), self.y)
            return 4

        # TAX/TAY/TXA/TYA/TSX/TXS
        if op == 0xAA:
            self.x = self.a
            self.set_zn(self.x)
            return 2
        if op == 0xA8:
            self.y = self.a
            self.set_zn(self.y)
            return 2
        if op == 0x8A:
            self.a = self.x
            self.set_zn(self.a)
            return 2
        if op == 0x98:
            self.a = self.y
            self.set_zn(self.a)
            return 2
        if op == 0xBA:
            self.x = self.sp
            self.set_zn(self.x)
            return 2
        if op == 0x9A:
            self.sp = self.x
            return 2

        # INX/INY/DEX/DEY
        if op == 0xE8:
            self.x = (self.x + 1) & 0xFF
            self.set_zn(self.x)
            return 2
        if op == 0xC8:
            self.y = (self.y + 1) & 0xFF
            self.set_zn(self.y)
            return 2
        if op == 0xCA:
            self.x = (self.x - 1) & 0xFF
            self.set_zn(self.x)
            return 2
        if op == 0x88:
            self.y = (self.y - 1) & 0xFF
            self.set_zn(self.y)
            return 2

        # ADC/SBC
        if op == 0x69:
            self.adc(self.fetch8())
            return 2
        if op == 0xE9:
            self.sbc(self.fetch8())
            return 2

        # AND/ORA/EOR
        if op == 0x29:
            self.a &= self.fetch8()
            self.set_zn(self.a)
            return 2
        if op == 0x09:
            self.a |= self.fetch8()
            self.set_zn(self.a)
            return 2
        if op == 0x49:
            self.a ^= self.fetch8()
            self.set_zn(self.a)
            return 2

        # CMP/CPX/CPY immediate
        if op == 0xC9:
            v = self.fetch8()
            r = (self.a - v) & 0x1FF
            self.set_c(self.a >= v)
            self.set_zn(r & 0xFF)
            return 2
        if op == 0xE0:
            v = self.fetch8()
            r = (self.x - v) & 0x1FF
            self.set_c(self.x >= v)
            self.set_zn(r & 0xFF)
            return 2
        if op == 0xC0:
            v = self.fetch8()
            r = (self.y - v) & 0x1FF
            self.set_c(self.y >= v)
            self.set_zn(r & 0xFF)
            return 2

        # JMP/JSR/RTS
        if op == 0x4C:
            self.pc = self.fetch16()
            return 3
        if op == 0x6C:
            ptr = self.fetch16()
            lo = self.read(ptr)
            hi = self.read((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
            self.pc = lo | (hi << 8)
            return 5
        if op == 0x20:
            addr = self.fetch16()
            ret = (self.pc - 1) & 0xFFFF
            self.push((ret >> 8) & 0xFF)
            self.push(ret & 0xFF)
            self.pc = addr
            return 6
        if op == 0x60:
            lo = self.pull()
            hi = self.pull()
            self.pc = ((hi << 8) | lo) + 1
            self.pc &= 0xFFFF
            return 6

        # Branches
        if op == 0xD0:
            self.branch(not (self.p & 0x02))
            return 2
        if op == 0xF0:
            self.branch(self.p & 0x02)
            return 2
        if op == 0x90:
            self.branch(not (self.p & 0x01))
            return 2
        if op == 0xB0:
            self.branch(self.p & 0x01)
            return 2
        if op == 0x10:
            self.branch(not (self.p & 0x80))
            return 2
        if op == 0x30:
            self.branch(self.p & 0x80)
            return 2

        # Flags
        if op == 0x18:
            self.p &= ~1
            return 2
        if op == 0x38:
            self.p |= 1
            return 2
        if op == 0x58:
            self.p &= ~0x04
            return 2
        if op == 0x78:
            self.p |= 0x04
            return 2
        if op == 0xB8:
            self.p &= ~0x40
            return 2

        # Stack
        if op == 0x48:
            self.push(self.a)
            return 3
        if op == 0x68:
            self.a = self.pull()
            self.set_zn(self.a)
            return 4
        if op == 0x08:
            self.push(self.p | 0x30)
            return 3
        if op == 0x28:
            self.p = self.pull()
            return 4

        # Unknown opcode: act like NOP so the GUI does not crash.
        return 2


class ACNES:
    def __init__(self):
        self.cart = NESCartridge()
        self.cpu_ram = bytearray(2048)
        self.ppu_regs = bytearray(8)
        self.controller = 0
        self.cpu = CPU6502(self)
        self.frame = 0

    def reset(self):
        self.cpu_ram = bytearray(2048)
        self.ppu_regs = bytearray(8)
        self.cpu.reset()
        self.frame = 0

    def cpu_read(self, addr):
        addr &= 0xFFFF

        if addr < 0x2000:
            return self.cpu_ram[addr & 0x07FF]

        if addr < 0x4000:
            return self.ppu_regs[addr & 7]

        if addr == 0x4016:
            return self.controller & 1

        if addr >= 0x8000 and self.cart.valid:
            if len(self.cart.prg) == 16384:
                return self.cart.prg[(addr - 0x8000) & 0x3FFF]
            return self.cart.prg[(addr - 0x8000) & 0x7FFF]

        return 0

    def cpu_write(self, addr, value):
        addr &= 0xFFFF
        value &= 0xFF

        if addr < 0x2000:
            self.cpu_ram[addr & 0x07FF] = value
        elif addr < 0x4000:
            self.ppu_regs[addr & 7] = value
        elif addr == 0x4016:
            pass

    def run_frame(self):
        # Real NTSC NES CPU is ~1.79 MHz, about 29,780 CPU cycles per frame.
        # This starter runs a safe chunk so the Python/Tkinter GUI stays smooth.
        target_cycles = 29780
        used = 0
        while used < target_cycles and self.cpu.running:
            used += self.cpu.step()
        self.frame += 1

    def render_pixels(self):
        pixels = []

        if not self.cart.valid:
            t = self.frame * 0.06
            for y in range(NES_H):
                row = []
                for x in range(NES_W):
                    wave = int((math.sin(x * 0.05 + t) + math.cos(y * 0.05 + t)) * 40 + 80)
                    if (x // 16 + y // 16 + self.frame // 20) % 2 == 0:
                        row.append((0, 40 + wave, 120 + wave))
                    else:
                        row.append((0, 0, 20))
                pixels.append(row)
            return pixels

        # Simplified CHR/pattern-table viewer.
        # This displays the ROM graphics tiles, not a full PPU background/sprite renderer yet.
        chr_data = self.cart.chr
        palette = [
            (0, 0, 0),
            (40, 90, 180),
            (110, 170, 255),
            (240, 248, 255),
        ]

        for y in range(NES_H):
            row = []
            for x in range(NES_W):
                tile_x = x // 8
                tile_y = y // 8
                px = x & 7
                py = y & 7
                tile_index = (tile_y * 32 + tile_x) % 256
                base = tile_index * 16
                if base + 15 < len(chr_data):
                    lo = chr_data[base + py]
                    hi = chr_data[base + py + 8]
                    bit = 7 - px
                    color_id = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
                    row.append(palette[color_id])
                else:
                    row.append((0, 0, 0))
            pixels.append(row)

        return pixels


class FCEUXStyleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(f"{WIDTH}x{HEIGHT}")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self.nes = ACNES()
        self.paused = False
        self.last_time = time.perf_counter()
        self.frames = 0
        self.fps_text = "60"

        self.make_menu()
        self.make_layout()

        self.photo = tk.PhotoImage(width=NES_W, height=NES_H)
        self.screen_img = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.bind_keys()
        self.loop()

    def make_menu(self):
        menubar = tk.Menu(self.root, bg=BG, fg=BLUE, activebackground=DARK_BLUE, activeforeground=WHITE)
        filemenu = tk.Menu(menubar, tearoff=0, bg=BG, fg=BLUE, activebackground=DARK_BLUE, activeforeground=WHITE)
        filemenu.add_command(label="Open ROM", command=self.open_rom)
        filemenu.add_command(label="Reset", command=self.reset)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.destroy)

        emumenu = tk.Menu(menubar, tearoff=0, bg=BG, fg=BLUE, activebackground=DARK_BLUE, activeforeground=WHITE)
        emumenu.add_command(label="Pause/Resume", command=self.toggle_pause)
        emumenu.add_command(label="About", command=self.about)

        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Emulation", menu=emumenu)
        self.root.config(menu=menubar)

    def button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BG,
            fg=BLUE,
            activebackground=DARK_BLUE,
            activeforeground=WHITE,
            relief="ridge",
            bd=2,
            width=12,
        )

    def make_layout(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x")

        title = tk.Label(
            top,
            text="AC'S NES EMU 0.1  |  FAMICOM SPEED 60 FPS",
            bg=BG,
            fg=BLUE,
            font=("Consolas", 12, "bold"),
        )
        title.pack(side="left", padx=8, pady=4)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            main,
            width=NES_W,
            height=NES_H,
            bg=BG,
            highlightthickness=2,
            highlightbackground=BLUE,
        )
        self.canvas.pack(side="left", padx=10, pady=8)

        side = tk.Frame(main, bg=BG)
        side.pack(side="right", fill="y", padx=8, pady=8)

        self.button(side, "Open ROM", self.open_rom).pack(pady=4)
        self.button(side, "Reset", self.reset).pack(pady=4)
        self.button(side, "Pause", self.toggle_pause).pack(pady=4)
        self.button(side, "About", self.about).pack(pady=4)
        self.button(side, "Exit", self.root.destroy).pack(pady=4)

        self.status = tk.Label(
            side,
            text="ROM: none\nMapper: N/A\nMode: demo\nFPS: 60",
            bg=BG,
            fg=BLUE,
            justify="left",
            font=("Consolas", 9),
        )
        self.status.pack(pady=12)

        controls = tk.Label(
            self.root,
            text="Controls: Arrow Keys = D-Pad | Z = A | X = B | Enter = Start | Right Shift = Select",
            bg=BG,
            fg=BLUE,
            font=("Consolas", 9),
        )
        controls.pack(side="bottom", pady=4)

    def bind_keys(self):
        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)

    def key_bit(self, event):
        k = event.keysym.lower()
        if k == "z":
            return 0
        if k == "x":
            return 1
        if k == "shift_r":
            return 2
        if k == "return":
            return 3
        if k == "up":
            return 4
        if k == "down":
            return 5
        if k == "left":
            return 6
        if k == "right":
            return 7
        return None

    def key_down(self, event):
        bit = self.key_bit(event)
        if bit is not None:
            self.nes.controller |= 1 << bit

    def key_up(self, event):
        bit = self.key_bit(event)
        if bit is not None:
            self.nes.controller &= ~(1 << bit)

    def open_rom(self):
        path = filedialog.askopenfilename(
            title="Open NES ROM",
            filetypes=[("NES ROM", "*.nes"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.nes.cart.load_ines(path)
            self.nes.reset()
            self.paused = False
            self.update_status()
        except Exception as e:
            messagebox.showerror("ROM Load Error", str(e))

    def reset(self):
        self.nes.reset()
        self.paused = False
        self.update_status()

    def toggle_pause(self):
        self.paused = not self.paused
        self.update_status()

    def about(self):
        messagebox.showinfo(
            APP_TITLE,
            "AC'S NES EMU 0.1\n"
            "Python 3.14 single-file starter emulator.\n\n"
            "Supports basic iNES loading and mapper 0 / NROM.\n"
            "CPU core is partial. PPU is currently a CHR tile viewer.\n"
            "GUI style: FCEUX-inspired black/blue."
        )

    def update_status(self):
        cart = self.nes.cart
        mode = "paused" if self.paused else ("running" if cart.valid else "demo")
        mapper = cart.mapper if cart.valid else "N/A"
        self.status.config(
            text=f"ROM: {cart.name}\nMapper: {mapper}\nMode: {mode}\nFPS: {self.fps_text}"
        )

    def draw_pixels(self, pixels):
        # Fast enough for this small starter emulator.
        lines = []
        for row in pixels:
            line = "{" + " ".join(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in row) + "}"
            lines.append(line)
        self.photo.put(" ".join(lines), to=(0, 0))

    def loop(self):
        start = time.perf_counter()

        if not self.paused:
            if self.nes.cart.valid:
                self.nes.run_frame()
            else:
                self.nes.frame += 1

        pixels = self.nes.render_pixels()
        self.draw_pixels(pixels)

        self.frames += 1
        now = time.perf_counter()
        if now - self.last_time >= 1.0:
            self.fps_text = str(self.frames)
            self.frames = 0
            self.last_time = now
            self.update_status()

        elapsed = time.perf_counter() - start
        delay_ms = max(1, int((1.0 / FPS - elapsed) * 1000))
        self.root.after(delay_ms, self.loop)


def main():
    root = tk.Tk()
    FCEUXStyleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
