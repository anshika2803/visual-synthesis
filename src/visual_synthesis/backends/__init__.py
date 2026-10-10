"""Model backends: image generation (image.py) and NSFW detection (detectors.py).

torch, diffusers and nudenet are imported only when a backend is loaded, so the
data stages run on a machine that has none of them.
"""
