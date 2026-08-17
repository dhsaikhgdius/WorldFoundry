"""Visualization helpers (action overlays) for generated video frames.

This ``__init__`` exists so setuptools package discovery includes
``worldfoundry.core.visualization`` in built distributions; without it the
subpackage imports from a source checkout (implicit namespace package) but is
silently dropped from wheels. Import the concrete helpers from
``worldfoundry.core.visualization.action_overlay``.
"""
