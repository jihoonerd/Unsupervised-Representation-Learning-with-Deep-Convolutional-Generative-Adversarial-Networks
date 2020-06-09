import imageio
import os


def gify(path: str, outname: str, fps: int):
    files = os.listdir(path)
    files.sort()
    frames = []
    for image in files:
        assert image.endswith("png")
        frames.append(imageio.imread(path + image))
    imageio.mimsave("./assets/" + outname, frames, fps=fps)
    return None
