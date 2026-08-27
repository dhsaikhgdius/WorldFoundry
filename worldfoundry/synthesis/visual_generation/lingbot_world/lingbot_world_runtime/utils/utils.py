
import argparse
import binascii
import logging
import os
import os.path as osp
import shutil

import imageio
import torch
import torchvision

from worldfoundry.core.process import read_text_tail, run_logged_subprocess

__all__ = ['save_video', 'save_image', 'str2bool', 'merge_video_audio']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"
    command = [
        'ffmpeg',
        '-y',
        '-i',
        video_path,
        '-i',
        audio_path,
        '-c:v',
        'copy',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-map',
        '0:v:0',
        '-map',
        '1:a:0',
        '-shortest',
        temp_output
    ]

    logging.info("Start merging video and audio...")
    stdout_path = f"{base}_merge_audio.stdout.log"
    stderr_path = f"{base}_merge_audio.stderr.log"
    result = run_logged_subprocess(
        command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        start_new_session=False,
    )
    if result.returncode != 0:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        tail = read_text_tail(stderr_path)
        raise RuntimeError(
            f"FFmpeg execute failed with code {result.returncode}. "
            f"See {stdout_path} and {stderr_path}.\n{tail}"
        )
    shutil.move(temp_output, video_path)
    logging.info(f"Merge completed, saved to {video_path}")


def save_video(tensor,
               save_file=None,
               fps=30,
               suffix='.mp4',
               nrow=8,
               normalize=True,
               value_range=(-1, 1)):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    tensor = tensor.clamp(min(value_range), max(value_range))
    tensor = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor.unbind(2)
    ],
                         dim=1).permute(1, 2, 3, 0)
    tensor = (tensor * 255).type(torch.uint8).cpu()

    writer = imageio.get_writer(
        cache_file, fps=fps, codec='libx264', quality=8)
    for frame in tensor.numpy():
        writer.append_data(frame)
    writer.close()


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    tensor = tensor.clamp(min(value_range), max(value_range))
    torchvision.utils.save_image(
        tensor,
        save_file,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range)
    return save_file


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
