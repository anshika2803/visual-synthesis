"""The six pipeline stages, one module each.

Every stage module has the same shape:

    NAME, SLUG, HELP, NEEDS_GPU     what the stage is
    run(cfg, run, source)           do the work, return a summary dict

``cfg`` is the resolved config, ``run`` is the run folder (see runs.py) and
``source`` is the stage's input: a file or the previous stage's run folder.
"""
