# FPGA Remote — User Manual

FPGA Remote is a browser-based interface for controlling a Genesys 2 FPGA board remotely.

---

## Layout

\[ Toolbar \]          — buttons and status

\[ LEDs | Switches \]  — indicator bar

\[ Terminal | Trace \] — tabbed main panel

The **status indicator** (top-right of the toolbar) shows the current state: Idle, Loading, Flashing, Capturing, Done, or Error. The active bitstream and loaded image are shown next to it, along with a connected-user count.

---

## Power

Click **Power: OFF** to turn the board on. The button label toggles to **Power: ON**. Most other actions require power to be on.

---

## Bitstreams

Click **Bitstreams** to open the Bitstream Manager.

| Action | How |
| :---- | :---- |
| Upload a `.bit` file | Select a file, optionally set a name, click **Upload** |
| Program FPGA (volatile) | Select a file, click **Flash volatile** — survives until power-off |
| Program FPGA (non-volatile) | Select a file, click **Flash non-volatile** — persists across power cycles via SPI flash |
| Delete / Rename | Select a file, click **Delete** or **Rename** |

The current bitstream name is shown in the toolbar status area after a successful flash.

---

## Boot Images

Click **Boot Images** to open the Image Manager.

| Action | How |
| :---- | :---- |
| Upload a `.bin` image | Select a file, optionally set a name, click **Upload** |
| Load an image | Select a file, click **Load** |
| Delete / Rename | Select a file, click **Delete** or **Rename** |

Loading transfers the image to `0x80000000` over JTAG using OpenOCD \+ GDB. A 15 MB image takes roughly 2+ minutes. The toolbar status shows **Loading…** while in progress.

---

## Terminal

The terminal panel streams UART output from the board and accepts keyboard input.

- **Click** (or tap on mobile) to focus, then type normally.
- **Ctrl+C / Ctrl+D / Ctrl+Z** and other control sequences work as expected.
- **Arrow keys**, Home, End, PageUp, PageDown, Tab, Escape, Backspace, Delete are mapped to VT100 sequences.
- **Paste** is supported (Ctrl+V or right-click paste on desktop; long-press on mobile).
- Click **Clear** in the toolbar to wipe the terminal buffer.

---

## Reset

Click **Reset** to send a board reset. The button is enabled only while power is on and no operation is in progress.

---

## Virtual LEDs

The **LEDs** section (left side of the indicator bar) shows the live state of 8 LEDs (0–7). Green \= asserted. Updated at 10 Hz.

---

## Virtual Switches

The **Switches** section (right side of the indicator bar) shows 8 switches (0–7). Click any switch to toggle it. Changes are sent to the board immediately and reflected for all connected users.

---

## Trace Dump

The Trace feature captures CVA6 Capstone tracer frames from UART.

1. Click **Trace Dump** in the toolbar. The button changes to **Cancel Trace** and the status shows **Capturing…**.
2. Trigger a trace dump from the running software.
3. When the end-of-dump frame arrives, the parsed trace appears automatically in the **Trace** tab.
4. To stop early, click **Cancel Trace**.

Switch between the terminal and the captured trace using the **Terminal** / **Trace** tabs below the indicator bar.

---

## Multiple Users

If more than one person is connected, the user count in the toolbar turns teal. All users share the same board state — switch toggles and terminal output are visible to everyone.

---

## Timeout

By default, the the power of the board is shut down with 10 minutes of inactivity. This timeout can be adjusted in settings. One can also toggle the lock to temporarily disable the shutdown behaviour.

# Tracing on CapliFive RTL

## Overview

The tracer has a 256-entry buffer that logs interesting events during the execution. Overall, the way tracing works is:

1. Tracing starts
2. The workload starts running. During the execution, the tracer logs events in the buffer
3. The workload finishes
4. The user interacts with the board to dump the buffer from the FPGA so the trace can be viewed

## Enable Tracing

Upon reset, tracing is initially disabled.  
Software can enable tracing for selected events. The events are selected by groups. See `tracer.sv` for the detailed grouping. Enablement of each group is controlled by one bit in the `0x810` CSR.

### Debug Print

One of the special events is debug print. When enabled, writes to the `0x800` CSR are logged, effectively allowing debug print.

### Watchpoint Triggers

Another special event is watchpoint triggers. The watchpoint is a physical address that is watched. Any write to the physical address triggers the watchpoint, and will be logged if the watchpoint trigger event tracing is enabled. The watchpoint address is configured through the `0x811` CSR.

## Replacement Policy

Since the tracer buffer is of limited size, it can easily become full. Switch 2 can configure what the tracer does when the buffer is already full:

* On: The oldest entry gets replaced by a new entry
* Off: The new entry gets dropped

## Dumping

To view the trace, it needs to be dumped first. As dumping is done through UART, which is multiplexed with the terminal, it is necessary to first detach it from the terminal lest the terminal output pollutes the dumped trace. This can be done by flipping switch 0 on. Next, click the "Trace Dump" button, which starts the dumping process: a thread will wait on UART for the trace data. Finally, flip on switch 1, which triggers the tracer to dump data in its buffer to UART. The trace should now be viewable in the trace pane.

# LED-based debugging

There are 8 LED indicators. These indicators can reflect some interesting signals in the RTL design. These are specified in `cva6.sv`.   
Then there are also 8 switches. They can serve as a special way to provide input to the RTL design. They are also accessible from [`cva6.sv`](http://cva6.sv).  
Currently, beyond a few special things associated with tracing (see the tracing tab), the switches are largely used to select signals to view on the LED indicators.

Note that to get these indicators and switches to work, it is necessary to use a pin mapping that's different from the original cva6. Cherrypick/merge this commit [https://github.com/project-starch/capstone-ariane/commit/aef4bd4445818eb558d1251d787a413f54315874](https://github.com/project-starch/capstone-ariane/commit/aef4bd4445818eb558d1251d787a413f54315874).

# GDB debugging

GDB-based debugging is also supported. To use it, go to the GDB tab and click on "start GDB." This pauses the CPU execution and attaches a GDB instance to it, allowing for single-stepping and viewing of CPU/memory states. After finishing with GDB debugging, click the "stop GDB" button.

Note that this functionality would require the CPU configured on the FPGA to be largely working.  
