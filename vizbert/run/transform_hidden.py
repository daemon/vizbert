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
    parser.add_argument('--new-key', type=str, default='hidden_state')
    args = parser.parse_args()

    probe_weight = torch.load(args.probe_path)['b']
    remove_mod = ProbeDirectionRemovalModule(probe_weight, use_complement=False, strength=1, normalize=True).cuda()
    data_dict = torch.load(str(args.hidden_path))
    transformed_hids = []

    for hid in tqdm(data_dict['hidden_state']):
        with torch.no_grad():
            hid = remove_mod(hid.cuda())
            transformed_hids.append(hid.cpu().detach())
    data_dict[args.new_key] = transformed_hids
    torch.save(data_dict, args.output_path)


if __name__ == '__main__':
    main()
