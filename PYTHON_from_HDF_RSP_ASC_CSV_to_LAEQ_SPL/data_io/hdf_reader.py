import pandas as pd
import numpy as np
import re


def extract_mics_from_hdf(filepath, logger, save_to_disk=False):

    # ================= HEADER =================
    with open(filepath, 'rb') as f:
        raw_head = f.read(200000)
        header_text = raw_head.decode('ascii', errors='ignore')

    data_start_idx = int(re.search(r'start of data\s*:\s*(\d+)', header_text).group(1))

    # Scan mode
    scan_mode_match = re.search(r'scan mode\s*:\s*(.+)', header_text)
    scan_mode = scan_mode_match.group(1).strip().lower() if scan_mode_match else ""

    # Sampling
    delta = float(re.search(r'delta value\s*:\s*([0-9.eE+-]+)', header_text).group(1))
    hw_fs = 1.0 / delta

    logger(f"Scan Mode: {scan_mode}")
    logger(f"Hardware FS: {hw_fs:.2f} Hz")

    # ================= CHANNEL PARSE =================
    ch_specs = {}
    channel_names = {}

    blocks = header_text.split('channel definition:')

    for block in blocks[1:]:
        try:
            ch_id = int(re.search(r'^\s*(\d+)', block).group(1))

            # Name
            name_match = re.search(r'name str\s*:\s*(.+)', block)
            name = name_match.group(1).strip() if name_match else f"Ch{ch_id}"
            channel_names[ch_id] = name

            # Type
            impl_match = re.search(r'implementation type\s*:\s*(.+)', block)
            impl = impl_match.group(1).strip() if impl_match else 'INT24'

            # Map factor
            map_match = re.search(r'map factor\s*:\s*([0-9.eE+-]+)', block)
            map_factor = float(map_match.group(1)) if map_match else 1.0

            # ===== REAL SENSITIVITY =====
            channel_index = ch_id - 1

            sens_match = re.search(
                rf'\[Channel{channel_index}\].*?_Sensitivity\s*=\s*double\s*\(\s*([0-9.eE+-]+)',
                header_text,
                re.DOTALL
            )

            sensitivity = float(sens_match.group(1)) if sens_match else None

            ch_specs[ch_id] = {
                'type': impl,
                'bytes': 3 if impl == 'INT24' else 4,
                'map': map_factor,
                'sens': sensitivity
            }

        except Exception as e:
            logger(f"Channel parse error: {e}")
            continue

    # ================= CHANNEL ORDER =================
    ch_order_match = re.search(r'ch order\s*:\s*([0-9,\s]+)', header_text)
    raw_seq = [int(x) for x in re.findall(r'\d+', ch_order_match.group(1))]

    # Detect repeating cycle
    macro_cycle = raw_seq
    for i in range(1, len(raw_seq)//2):
        if raw_seq[:i] == raw_seq[i:2*i]:
            macro_cycle = raw_seq[:i]
            break

    # ================= BYTE OFFSETS =================
    offsets = {ch: [] for ch in set(macro_cycle)}
    pos = 0

    for ch in macro_cycle:
        size = ch_specs[ch]['bytes']
        offsets[ch].append((pos, pos + size))
        pos += size

    cycle_bytes = pos

    # ================= READ DATA =================
    with open(filepath, 'rb') as f:
        f.seek(data_start_idx)
        raw = f.read()

    num_blocks = len(raw) // cycle_bytes
    data = np.frombuffer(raw[:num_blocks * cycle_bytes], dtype=np.uint8)
    data = data.reshape(num_blocks, cycle_bytes)

    logger("Unpacking multiplexed data...")

    # ================= EXTRACT MIC CHANNELS =================
    mic_data = []
    mic_names = []
    mic_sens = {}

    exclude_keywords = ["can", "rpm", "eng", "speed"]

    for ch in sorted(offsets):
        name = channel_names.get(ch, f"Ch{ch}")
        name_lower = name.lower()

        # Skip non-acoustic channels
        if any(k in name_lower for k in exclude_keywords):
            logger(f"Skipping non-mic channel: {name}")
            continue

        # 🔥 PRINT SENSITIVITY IN SAME LINE
        sens = ch_specs[ch].get("sens", None)

        if sens:
            logger(f"Detected MIC channel: {name} → Sens = {sens:.6f} V/Pa")
        else:
            logger(f"Detected MIC channel: {name} → Sens = N/A")

        mic_names.append(name)
        mic_sens[name] = sens

        chunks = []
        for start, end in offsets[ch]:
            sb = data[:, start:end]

            if ch_specs[ch]['type'] == 'INT24':
                sign = np.where(sb[:, 2] >= 128, 255, 0).astype(np.uint8)
                vals = np.column_stack((sb, sign)).view('<i4').flatten()
            else:
                vals = sb.view('<f4').flatten()

            chunks.append(vals)

        raw_vals = np.column_stack(chunks).flatten()

        # ✅ FINAL CORRECT SCALING
        pascals = raw_vals * ch_specs[ch]['map']

        mic_data.append(pascals)

    # ================= FS CORRECTION =================
    first_mic = list(offsets.keys())[0]

    if "simultaneous" in scan_mode:
        fs_eff = hw_fs
    else:
        fs_eff = hw_fs * (len(offsets[first_mic]) / len(macro_cycle))

    logger(f"Effective FS: {fs_eff:.2f} Hz")

    # ================= TIME =================
    time = np.arange(len(mic_data[0])) / fs_eff

    df = pd.DataFrame(np.column_stack([time] + mic_data))
    df.columns = ["Time"] + mic_names

    # Attach sensitivity for GUI use
    df.sensitivities = mic_sens

    return df