import pandas as pd
import numpy as np
import re
import os

from data_io.hdf_reader import extract_mics_from_hdf


# ================= MAIN ENTRY =================
def read_file(file_path, logger, keep_asc=False):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".hdf":
        return extract_mics_from_hdf(file_path, logger, save_to_disk=keep_asc)

    elif ext in (".rsp", ".rpc"):
        logger("  [RPC/RSP Detected] Reading file...")
        return read_rpc_file(file_path)

    elif ext in (".asc", ".csv"):
        logger("  [ASCII/CSV Detected] Reading table...")
        return read_ascii_csv(file_path)

    else:
        raise ValueError(f"Unsupported file: {ext}")


# ================= RSP / RPC =================
def read_rpc_file(file_path):
    header_info, chan_scales, chan_desc, chan_units = {}, {}, {}, {}
    num_header_blocks, int_full_scale = 2, 32752.0
    data_format = 'SHORT'
    
    with open(file_path, 'rb') as f:
        lin = 0
        while lin < (num_header_blocks * 4) or (lin % 4 != 0):
            chunk = f.read(128)
            if not chunk:
                break
            lin += 1

            chunk_str = chunk.replace(b'\x00', b' ').decode('latin-1', errors='ignore').strip()
            if not chunk_str:
                continue

            name = chunk_str[:32].strip().upper()
            value = chunk_str[32:].strip()

            if name == 'NUM_HEADER_BLOCKS':
                num_header_blocks = int(value)
            elif name == 'INT_FULL_SCALE':
                int_full_scale = float(value)
            elif name == 'FORMAT':
                data_format = value.upper()
            elif name == 'CHANNELS':
                header_info['channels'] = int(value)
            elif name == 'PTS_PER_GROUP':
                header_info['pts_per_group'] = int(value)
            elif name == 'FRAMES':
                header_info['frames'] = int(value)
            elif name == 'DELTA_T':
                header_info['delta_t'] = float(value)
            elif name.startswith('SCALE.CHAN_'):
                chan_scales[int(name[11:15].strip())] = float(value)
            elif name.startswith('DESC.CHAN_'):
                chan_desc[int(name[10:14].strip())] = value
            elif name.startswith('UNITS.CHAN_'):
                chan_units[int(name[11:15].strip())] = value

        if 'channels' not in header_info:
            raise ValueError("Invalid RPC file")

        # ===== DATA =====
        if 'FLOAT' in data_format or 'REAL' in data_format:
            raw_data = np.fromfile(f, dtype=np.float32)
        else:
            raw_data = np.fromfile(f, dtype=np.int16)

    num_channels = header_info['channels']
    num_ppg = header_info.get('pts_per_group', 1)

    total_pts = num_channels * num_ppg
    frames = len(raw_data) // total_pts

    raw_data = raw_data[:frames * total_pts]
    reshaped = raw_data.reshape((frames, num_channels, num_ppg))

    time = np.arange(frames * num_ppg) * header_info.get('delta_t', 0.001)

    data_dict = {"Time": time}

    for i in range(num_channels):
        ch_idx = i + 1
        scale = chan_scales.get(ch_idx, 1.0)

        if 'FLOAT' not in data_format:
            scale *= (32768.0 / int_full_scale)

        values = reshaped[:, i, :].flatten() * scale

        name = chan_desc.get(ch_idx, f"Mic {ch_idx}")
        unit = chan_units.get(ch_idx, "Pa")

        data_dict[f"{name} [{unit}]"] = values

    return pd.DataFrame(data_dict)


# ================= ASC / CSV =================
def read_ascii_csv(file_path):

    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
        lines = f.readlines()

    table_data = []

    for line in lines:
        nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+', line)
        if len(nums) >= 2:
            table_data.append([float(x) for x in nums])

    df = pd.DataFrame(table_data)

    # Assign column names
    df.columns = ["Time"] + [f"Mic {i}" for i in range(1, df.shape[1])]

    return df