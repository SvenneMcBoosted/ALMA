from astropy.visualization import ZScaleInterval
from astropy.io import fits

def fits2png(file_path):
    data = fits.getdata(file_path)
    zscale = ZScaleInterval()
    data = zscale(data).squeeze()
    return data