I'd like to integrate a new renderer using the ovrtx library.

You can find ovrtx Python bindings at /home/ncournia/dev/kit.0/rendering/source/bindings/python.

The code should follow the pattern used to integrate the Newton warp renderer.

When running Python code, the following environment variables should be used:

export OVRTX_SKIP_USD_CHECK=1
export LD_LIBRARY_PATH=/home/ncournia/dev/kit.0/rendering/_build/linux-x86_64/release:$LD_LIBRARY_PATH
export PYTHONPATH=/home/ncournia/dev/kit.0/rendering/source/bindings/python:$PYTHONPATH
env LD_PRELOAD=~/dev/kit.0/kit/_build/linux-x86_64/release/libcarb.so
