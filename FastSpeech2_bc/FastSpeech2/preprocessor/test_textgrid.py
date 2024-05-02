import tgt
import numpy as np



sampling_rate = 22050
hop_length = 1024

def get_alignment(tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            if p == "":
                p="sp"

            # Trim leading silences
            if phones == []:
                if p == "":
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * sampling_rate / hop_length)
                    - np.round(s * sampling_rate / hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

import ptvsd

ptvsd.enable_attach(("0.0.0.0",5678))


print("Waiting for debuger to attach...")
ptvsd.wait_for_attach()


tg_path = "/home/pc/mfa_data/bc2021_aligned/00006403.TextGrid"

textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
phone, duration, start, end = get_alignment(
            textgrid.get_tier_by_name("phones")
)

print(phone)