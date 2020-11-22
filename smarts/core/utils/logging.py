# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import logging
from time import time
from contextlib import contextmanager


@contextmanager
def timeit(name: str):
    start = time()
    yield
    ellapsed_time = (time() - start) * 1000

    logging.info(f'"{name}" took: {ellapsed_time:4f}ms')


# Reference: https://gist.github.com/simon-weber/7853144
@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages triggered during the
    body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # HACK: If can't get the highest logging level in effect then delegate to the user.
    #       If can't get the current module-level override then use an undocumented
    #       (but non-private!) interface.

    previous_level = logging.root.manager.disable
    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
