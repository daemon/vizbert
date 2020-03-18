from pathlib import Path
import argparse

from tqdm import tqdm
import torch

from vizbert.inject import ProbeDirectionRemovalModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-path', type=str, required=True)
    parser.add_argument('--hidden-path', '-hp', type=Path, required=True)
    parser.add_argument('--output-path', '-o', type=Path, required=True)
    args = parser.parse_args()

    probe_weight = torch.load(args.probe_path)['b']
    remove_mod = ProbeDirectionRemovalModule(probe_weight, use_complement=False, strength=1, normalize=True).cuda()
    hidden = torch.load(str(args.hidden_path))
    removed_hids = []

    for hid in tqdm(hidden['hidden_state']):
        with torch.no_grad():
            hid = remove_mod(hid.cuda())
            removed_hids.append(hid.cpu().detach())
    torch.save(dict(hidden_state=removed_hids), args.output_path)


if __name__ == '__main__':
    main()
