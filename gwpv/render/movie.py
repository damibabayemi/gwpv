import os
import subprocess


def render_movie(output_filename, frames_dir, frame_rate):
    subprocess.call([
        'ffmpeg', '-vcodec', 'png', '-framerate',
        str(frame_rate), '-i',
        os.path.join(frames_dir, r'frame.%06d.png'), '-pix_fmt', 'yuv420p',
        '-vcodec', 'mpeg4', '-threads', '0',
        '-y', output_filename + '.mp4'
    ])
