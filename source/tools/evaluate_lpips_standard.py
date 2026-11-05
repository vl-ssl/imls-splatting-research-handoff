"""Recompute LPIPS-VGG on [-1,1] images without altering legacy metrics."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from lpipsPyTorch.modules.lpips import LPIPS


def evaluate(test_dir, model):
    render_dir, gt_dir = test_dir / 'renders', test_dir / 'gt'
    names = sorted(p.name for p in render_dir.glob('*.png'))
    if not names or any(not (gt_dir / name).is_file() for name in names):
        raise RuntimeError(f'Unmatched rendered/GT views: {test_dir}')
    results = {}
    with torch.no_grad():
        for name in names:
            with Image.open(render_dir / name) as render_image:
                rendered = to_tensor(render_image.convert('RGB'))[None].cuda()
            with Image.open(gt_dir / name) as gt_image:
                reference = to_tensor(gt_image.convert('RGB'))[None].cuda()
            results[name] = float(model(2 * rendered - 1, 2 * reference - 1).item())
    output = {
        'metric': 'LPIPS-VGG v0.1', 'input_range': '[-1,1]',
        'count': len(results), 'mean': sum(results.values()) / len(results),
        'per_view': results,
    }
    (test_dir / 'standard_lpips_vgg.json').write_text(json.dumps(output, indent=2))
    print(test_dir, output['count'], output['mean'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    model = LPIPS('vgg', '0.1').cuda().eval()
    for test_dir in args.test_dirs:
        evaluate(test_dir, model)


if __name__ == '__main__':
    main()
