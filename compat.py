try:
    from PIL import Image, ImageFilter
except ImportError:
    import Image
    import ImageFilter


import numpy
import tempfile
from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,

                   transpose, iscomplexobj, uint8, issubdtype, array)


if not hasattr(Image, 'frombytes'):
    Image.frombytes = Image.fromstring

def imresize(arr, size, interp='bilinear', mode=None):
    """

    Resize an image.



    This function is only available if Python Imaging Library (PIL) is installed.



    .. warning::



        This function uses `bytescale` under the hood to rescale images to use

        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.

        It will also cast data for 2-D images to ``uint32`` for ``mode=None``

        (which is the default).



    Parameters

    ----------

    arr : ndarray

        The array of image to be resized.

    size : int, float or tuple

        * int   - Percentage of current size.

        * float - Fraction of current size.

        * tuple - Size of the output image (height, width).



    interp : str, optional

        Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',

        'bicubic' or 'cubic').

    mode : str, optional

        The PIL image mode ('P', 'L', etc.) to convert `arr` before resizing.

        If ``mode=None`` (the default), 2-D images will be treated like

        ``mode='L'``, i.e. casting to long integer.  For 3-D and 4-D arrays,

        `mode` will be set to ``'RGB'`` and ``'RGBA'`` respectively.



    Returns

    -------

    imresize : ndarray

        The resized array of image.



    See Also

    --------

    toimage : Implicitly used to convert `arr` according to `mode`.

    scipy.ndimage.zoom : More generic implementation that does not use PIL.



    """

    im = Image.fromarray(arr, mode=mode)

    ts = type(size)

    if issubdtype(ts, numpy.signedinteger):

        percent = size / 100.0

        size = tuple((array(im.size)*percent).astype(int))

    elif issubdtype(type(size), numpy.floating):

        size = tuple((array(im.size)*size).astype(int))

    else:

        size = (size[1], size[0])

    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}

    imnew = im.resize(size, resample=func[interp])

    return numpy.asarray(imnew)

import scipy.misc as misc
misc.imresize = imresize