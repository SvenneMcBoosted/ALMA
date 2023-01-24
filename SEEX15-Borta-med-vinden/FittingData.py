from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from alminer_extensions import get_molecule, get_frequencies
import scipy.optimize as opt
from astropy.wcs import WCS
import warnings
import os
from heapq import heappop, heappush, heapify

warnings.filterwarnings("ignore")

freqs = get_frequencies('molecules.csv')

##############################################
# Math and helper functions
##############################################

def twoDimensionalEllipticalGauss(xDataTuple, amplitude, xCenter, yCenter, sigmaX, sigmaY, theta, offset):
    """
    Evaluates the two-dimensional Gaussian function at some coordinates given the parameters.
    ----------
    xDataTuple : tuple of (list of) floats
         The coordinates where the function is evaluated
    amplitude : float
         The amplitude of the Gaussian
    xCenter : float
         The x-coordinate for the center point of the Gaussian
    yCenter : float
         The y-coordinate for the center point of the Gaussian
    sigmaX : float
         The standard deviation of the Gaussian in the x-direction
    sigmaY : float
         The standard deviation of the Gaussian in the y-direction
    theta : float
         The counterclockwise rotation of the Gaussian
    offset : float
         The offset of the Gaussian (shift in the z-direction)
    Returns
    -------
    A 1-dimensional array containing the function evaluated at every given coordinate, in row-major order.
    """
    x, y = xDataTuple
    xCenter = float(xCenter)
    yCenter = float(yCenter)
    a = np.cos(theta) ** 2 / (2 * sigmaX ** 2) + np.sin(theta) ** 2 / (2 * sigmaY ** 2)
    b = -np.sin(2 * theta) / (4 * sigmaX ** 2) + np.sin(2 * theta) / (4 * sigmaY ** 2)
    c = np.sin(theta) ** 2 / (2 * sigmaX ** 2) + np.cos(theta) ** 2 / (2 * sigmaY ** 2)
    z = offset + amplitude * np.exp(- (a * ((x - xCenter) ** 2) + 2 * b * (x - xCenter) * (y - yCenter)
                                       + c * ((y - yCenter) ** 2)))
    return z.ravel()


def oneDimensionalGaussian(x, amplitude, center, sigma, offset=0):
    """
    Evaluates the one dimensional Gaussian function at a value x given parameters.
    ----------
    x : float
         The location to compute the function.
    amplitude : float
         The amplitude of the gaussian(max value)
    center : float
         The center of the gaussian(location of max)
    sigma : float
         The standard deviation of the gaussian(in "space")
    offset : float
         The offset of the gaussian(shift in y-direction)
    Returns
    -------
    The value of the specified gaussian at point x.
    """
    return offset + abs(amplitude) * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def sumOfTwoGauss(x, amplitude1, center1, sigma1, offset1, amplitude2, center2, sigma2, offset2):
    """
    Evaluates the sum of two one dimensional Gaussians at a point x given parameters
    ----------
    x : float
         The location to compute the function.
    amplitude : float
         The amplitude of the gaussian(max value)
    center : float
         The center of the gaussian(location of max)
    sigma : float
         The standard deviation of the gaussian(in "space")
    offset : float
         The offset of the gaussian(shift in y-direction)
    Returns
    -------
    The value of the specified gaussian at point x.
    """
    return oneDimensionalGaussian(x, amplitude1, center1, sigma1, offset1) + oneDimensionalGaussian(x, amplitude2,
                                                                                                    center2, sigma2,
                                                                                                    offset2)


def sumOfThreeGauss(x, a1, c1, s1, o1, a2, c2, s2, o2, a3, c3, s3, o3):
    """
    Evaluates the sum of three one dimensional Gaussians (where the third has negative amplitude) at a point x given parameters
    ----------
    x : float
         The location to compute the function.
    a : float
         The amplitude of the gaussian(max value)
    c : float
         The center of the gaussian(location of max)
    s : float
         The standard deviation of the gaussian(in "space")
    o : float
         The offset of the gaussian(shift in y-direction)
    Returns
    -------
    The value of the specified gaussian at point x.
    """
    return oneDimensionalGaussian(x, a1, c1, s1, o1) + oneDimensionalGaussian(x, a2, c2, s2,
                                                                              o2) - oneDimensionalGaussian(x, a3, c3,
                                                                                                           s3, o3)


def linFunc(x, k, m):
    """Computes value of the line function y = kx+m at x value"""
    return k * x + m


def fitWrapper(coeffs, *args):
    """Wrapper function that allows us to weight a line function"""
    xdata, ydata, prio = args
    return prio * (linFunc(xdata, *coeffs) - ydata)


def clearPlots(plotIndicies):
    """Clears and closes all plots"""
    for i in plotIndicies:
        plt.figure(i)
        plt.clf()
        plt.cla()
        plt.close()


def rms(matrix):
    """Computes the quadratic mean"""
    vals = np.ravel(matrix)
    rootmeansquared = np.sqrt(np.nanmean(vals ** 2))
    return rootmeansquared


def computeNoise(moment,
                 partitions=8):  # metod som beräknar brusnivån så som per illustrerade , känns som bruset blir "för lågt" med denna metod dock
    """
    Computes the noise in a moment map by subdividing the map and computing the average noise in each submap.
    ----------
    moment : matrix
        The moment map matrix
    partitions : integer
        The side length for the grid, i.e. we get a (partitions x partitions) grid
    Returns
    -------
    """
    means = []
    imageWidth = moment.shape[0]
    for submatrix in split(moment, imageWidth // partitions, imageWidth // partitions):
        submatrix = submatrix[~np.isnan(submatrix)]
        if len(submatrix) < 5:
            continue
        means.append(rms(submatrix))
    # print(means)
    # print(np.min(means))
    return np.min(means)


def split(array, nrows, ncols):
    """Helper method that splits a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))

def rotatePlane(x, y, theta):
    """
    Takes x and y coordinates and returns rotated coordinates
    """
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array((c, -s), (s, c))
    xy = np.array([[x],[y]])
    return map(round, (R*xy)[0])

# These three might be unnecessary...
def imageToNewZero(x, y, xcenter, ycenter):
    """Takes image coordinates and translates to coordinates zerod on xcenter and ycenter"""
    return (x-xcenter, y-ycenter)

def newZeroToImage(x, y, xcenter, ycenter):
    """Returns image coordinates from zerod coordinates"""
    return (x+xcenter, y+ycenter)

def outflowParabola(x, a, b):
    """Parabola for identifying outflows, might get scrapped depending
    on how the implementation turns out"""
    return a*x**2 + b

def hyperbolicParaboloildList(data, a, b, c, d):
    """Mapping of hyperbolic paraboloid function on lists of x and y values"""
    x, y = data
    #return [hyperbolicParaboloid(x, y, a, b, c, d) for x,y in list(zip(x, y))]
    z = -(((x-b)/a)**2+((y-d)/c)**2)+1.0
    return z.ravel()

def rotatingParabola(x, a, b, theta):
    """Outflow parabola but with added rotation"""
    y = outflowParabola(x, a, b)
    return rotatePlane(x, y, theta)

def sumOfParabola(x, a, b, theta, moment):
    """Sums the intensities """
    sum = 0
    for i in x:
        sum += moment[i][rotatingParabola(i, a, b, theta)]

    return sum

##############################################
# Data fitting
##############################################

def fit2DGaussianToContData(filename, createPlot=False, plotDistanceFromCenter=10):
    """
    Fits a two-dimensional Gaussian function to continuum data.
    ----------
    filename : String
         The location of the continuum data fits file.
    createPlot : bool, optional
         (Default value = False)
         Plots the gaussian fit to the continuum data.
    plotDistanceFromCenter : float, optional
         (Default value = 10)
         Determines how far out from the center the bounds of the plot are.
    Returns
    -------
    The fitted parameters to the Gaussian function.
    """
    with fits.open(filename) as hdul:
        fitsData = hdul[0].data
        contMatrix = np.squeeze(fitsData)  # Transforms matrix into correct shape
        imageWidth = contMatrix.shape[0]
        contMatrix = np.nan_to_num(contMatrix)
        x = np.linspace(0, imageWidth, imageWidth)
        y = np.linspace(0, imageWidth, imageWidth)
        x, y = np.meshgrid(x, y)
        initialGuess = [np.max(contMatrix), imageWidth / 2, imageWidth / 2, 1, 1, 0, 1]
        fittedValues, _ = opt.curve_fit(twoDimensionalEllipticalGauss, (x, y), contMatrix.ravel(), p0=initialGuess)
        if createPlot:
            fittedData = twoDimensionalEllipticalGauss((x, y), *fittedValues)
            plt.figure(1)
            wcs = WCS(filename)
            if wcs.naxis > 2:
                wcs = wcs.sub(2)
            plt.subplot(projection=wcs)
            plt.imshow(contMatrix, origin='lower')
            plt.colorbar(label=r"Intensity (Jy beam$^{-1}$)")
            plt.contour(x, y, fittedData.reshape(imageWidth, imageWidth), 2, colors="w")
            plt.xlabel("Right Ascension (J2000)")
            plt.ylabel("Declination (J2000)")
            plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Continuum")
            plt.axis([fittedValues[1] - plotDistanceFromCenter * fittedValues[3],  # Show plot in region around max
                      fittedValues[1] + plotDistanceFromCenter * fittedValues[3],
                      fittedValues[2] - plotDistanceFromCenter * fittedValues[4],
                      fittedValues[2] + plotDistanceFromCenter * fittedValues[4]])
            plt.savefig(filename.split(".cont")[0] + "_contFit" + ".pdf")
        del hdul[0].data
    return fittedValues

def fitParabolaToContData(filename):
    with fits.open(filename) as hdul:
        fitsData = hdul[0].data
        contMatrix = np.squeeze(fitsData)
        imageWidth = contMatrix.shape[0]
        contMatrix = np.nan_to_num(contMatrix)
        x = np.linspace(0, imageWidth, imageWidth)
        y = np.linspace(0, imageWidth, imageWidth)
        x, y = np.meshgrid(x, y)
        #print(x)
        #print(y)
        initialGuess = [1.5, 0.4, 1.5, 0.4]
        fittedValues, _ = opt.curve_fit(hyperbolicParaboloildList, (x, y), contMatrix.ravel(), p0=initialGuess)

        if True:
            fittedData = hyperbolicParaboloildList((x, y), *fittedValues)
            plt.figure(1)
            wcs = WCS(filename)
            if wcs.naxis > 2:
                wcs = wcs.sub(2)
            plt.subplot(projection=wcs)
            plt.imshow(contMatrix, origin='lower')
            plt.colorbar(label=r"Intensity (Jy beam$^{-1}$)")
            plt.contour(x, y, fittedData.reshape(imageWidth, imageWidth), 2, colors="w")
            plt.xlabel("Right Ascension (J2000)")
            plt.ylabel("Declination (J2000)")
            plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Continuum")
            plt.savefig(filename.split(".cont")[0] + "_contFit" + ".pdf")

        del hdul[0].data
    return fittedValues

def fitParabolaToMomentMap(moment, filename):
    imageWidth = moment.shape[0]
    x = np.linspace(0, imageWidth, imageWidth)
    y = np.linspace(0, imageWidth, imageWidth)
    x, y = np.meshgrid(x, y)
    initialGuess = [1.5, 0.4, 1.5, 0.4]
    fittedValues, _ = opt.curve_fit(hyperbolicParaboloildList, (x, y), moment.ravel(), p0=initialGuess)

    if True:
        fittedData = hyperbolicParaboloildList((x, y), *fittedValues)
        plt.figure(1)
        wcs = WCS(filename)
        if wcs.naxis > 2:
            wcs = wcs.sub(2)
        plt.subplot(projection=wcs)
        plt.imshow(moment, origin='lower')
        plt.colorbar(label=r"Intensity (Jy beam$^{-1}$)")
        plt.contour(x, y, fittedData.reshape(imageWidth, imageWidth), 2, colors="w")
        plt.xlabel("Right Ascension (J2000)")
        plt.ylabel("Declination (J2000)")
        #plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Continuum")
        plt.savefig(filename.split(".cont")[0] + "_contFit" + ".pdf")

    return fittedValues

def getPointsWithinGaussian(fittedValues, proportionOfMaximum=1 / 2, distanceFromCenter=10, createPlot=False):
    """
    Finds the points which are in an ellipse of "given size"
    ----------
    fittedValues : list of floats
         List of fitted parameters to the Gaussian function.
    proportionOfMaximum : float, optional
         (Default value = 1/2 {FWHM} )
         Cutoff for size of ellipse
    distanceFromCenter: float, optional
         (Default value = 10)
         How far from the center to search, (to skip iterating over all 1000x1000 pixels)
    createPlot : bool, optional
         (Default value = False)
         Plots the points.
    Returns
    -------
    A list of all coordinates contained within an ellipse of given size.
    """
    fMax = fittedValues[0] + fittedValues[6]
    xCenter = fittedValues[1]
    yCenter = fittedValues[2]
    xSigma = np.abs(fittedValues[3])
    ySigma = np.abs(fittedValues[4])
    coordinatesInEllipse = []
    for i in range(int(xCenter - distanceFromCenter * xSigma), int(xCenter + distanceFromCenter * xSigma)):
        for j in range(int(yCenter - distanceFromCenter * ySigma), int(yCenter + distanceFromCenter * ySigma)):
            if twoDimensionalEllipticalGauss((i, j), *fittedValues) > fMax * proportionOfMaximum:
                coordinatesInEllipse.append([i, j])
    if createPlot:
        x, y = np.array(coordinatesInEllipse).T
        # plt.figure(2)
        plt.scatter(x, y, c='black')
    return coordinatesInEllipse


def oneDGaussianMeanFit(means, createPlot=False):
    """
    Fits a one dimensional Gaussian to a list of values
    ----------
    means : list of floats
        The values to fit the Gaussian to.
    createPlot : bool, optional
         (Default value = False)
         Decides if the fit is plotted
    Returns
    -------
    The parameters of the fitted Gaussian.
    """
    n = len(means)
    x = np.linspace(1, n, n)
    sigma = len(means) / 30
    initialGuess = [np.max(means), np.argmax(means), sigma]
    parameters, _ = opt.curve_fit(oneDimensionalGaussian, x, means, p0=initialGuess)
    if createPlot:
        plt.figure(8)
        plt.plot(x, means, 'b+:', label='data')
        plt.plot(x, oneDimensionalGaussian(x, *parameters), 'ro:', label='fit')
    return parameters


def bimodalGaussianMeanFit(means, createPlot=False, createSubPlot=False):
    """
    Fits the sum of two one dimensional Gaussians to a list of values
    ----------
    means : list of floats
        The values to fit the Gaussian to.
    createPlot : bool, optional
         (Default value = False)
         Decides if the fit is plotted
    createSubPlot : bool, optional
         (Default value = False)
         Decides if the subfit is plotted
    Returns
    -------
    The parameters of the fitted Gaussians.
    """
    n = len(means)
    x = np.linspace(1, n, n)
    sigma = oneDGaussianMeanFit(means, createSubPlot)[2]
    initialGuess = [np.max(means), np.argmax(means), sigma, 0.001, np.max(means), np.argmax(means), sigma,
                    0.001]  # needs something smarter
    lowerBounds = [-2 * np.max(means), 0, -len(means), -1, -2 * np.max(means), 0, -len(means), -1]
    upperBounds = [2 * np.max(means), len(means), len(means), 1, 2 * np.max(means), len(means), len(means), 1]
    parameters, _ = opt.curve_fit(sumOfTwoGauss, x, means, p0=initialGuess, bounds=(lowerBounds, upperBounds))
    if createPlot:
        plt.figure(9)
        plt.plot(x, means, 'b+:', label='data')
        plt.plot(x, sumOfTwoGauss(x, *parameters), 'ro:', label='fit')
    return parameters


def trimodalGaussianMeanFit(means, createPlot=False, createSubPlot=False):
    """
    Fits the sum of three one dimensional Gaussians to a list of values
    ----------
    means : list of floats
        The values to fit the Gaussian to.
    createPlot : bool, optional
         (Default value = False)
         Decides if the fit is plotted
    createSubPlot : bool, optional
         (Default value = False)
         Decides if the subfit is plotted
    Returns
    -------
    The parameters of the fitted Gaussians.
    """
    n = len(means)
    x = np.linspace(1, n, n)
    sigma = oneDGaussianMeanFit(means, createSubPlot)[2]
    # sigma = len(means)/30
    initialGuess = [np.max(means), np.argmax(means), sigma, 0.001, np.max(means), np.argmin(means) + sigma, sigma,
                    0.001, np.min(means), np.argmin(means), sigma, 0.001]  # needs something smarter
    lowerBounds = [-2 * np.max(means), 0, -len(means), -1, -2 * np.max(means), 0, -len(means), -1, -2 * np.max(means),
                   0, -len(means), -1]
    upperBounds = [2 * np.max(means), len(means), len(means), 1, 2 * np.max(means), len(means), len(means), 1,
                   2 * np.max(means), len(means), len(means), 1]
    parameters, _ = opt.curve_fit(sumOfThreeGauss, x, means, p0=initialGuess, bounds=(lowerBounds, upperBounds))
    if createPlot:
        plt.figure(10)
        plt.plot(x, means, 'b+:', label='data')
        plt.plot(x, sumOfThreeGauss(x, *parameters), 'ro:', label='fit')
    return parameters


##############################################
# Intensity Profiles and Ranges
##############################################

def meanSpectralProfile(filename, coordinates, createPlot=False):
    """
    Computes the mean intensity in the fitted region for each frequency
    ----------
    filename : String
         The location of the cube data fits file.
    coordinates : ints
         List of coordinates within the given region.
    createPlot : bool, optional
         (Default value = False)
         Plots the spectral profile.
    Returns
    -------
    A list of the mean values for each frequency.
    """
    with fits.open(filename) as hdul:
        cube = hdul[0].data
        cubeData = np.squeeze(cube)
        means = []
        for i in range(0, cubeData.shape[0]):
            mean = 0
            for x, y in coordinates:
                mean += cubeData[i, x, y]
            mean = mean / len(coordinates)
            means.append(mean)

        rootMean = rms(means)
        print(rootMean)

        if createPlot:
            header = hdul[0].header
            x = np.linspace(header.get("CRVAL3"), header.get("CRVAL3") + header.get("CDELT3") * cubeData.shape[0],
                            cubeData.shape[0])
            if header.get("CDELT3") < 0:
                x = np.flip(x)
            plt.figure(3)
            plt.plot(x, means)
            plt.axhline(rootMean)
            col = "k"
            plt.axvline(x=header.get("RESTFRQ"), color=col)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(r"Mean intensity (Jy beam$^{-1}$)")
            plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Spectral Profile")
            plt.savefig(filename.split(".cube")[0] + "_spectralProfile" + ".pdf")
        del hdul[0].data
    return means


def findRangesByInflection(means, filename, createPlot=False):
    """
    Computes inflection points of the spectral profile and extracts ranges. Needs some work.
    ----------
    means : list of floats
         The mean values for each frequency
    createPlot : bool, optional
         (Default value = False)
         Plots the inflection points
    Returns
    -------
    A list of start and end values for use as bounds.
    """
    interpolatedMeans = sumOfThreeGauss(np.linspace(0, len(means), len(means)), *trimodalGaussianMeanFit(means))
    normalisedSquaredError = np.mean(((interpolatedMeans - means) / np.max(means)) ** 2)  # rms kanske istället
    print("error: ", normalisedSquaredError)
    if normalisedSquaredError > 0.05:
        return "break"

    interpolatedMeansDerivative = np.gradient(interpolatedMeans)
    interpolatedMeans2ndDerivative = np.gradient(interpolatedMeansDerivative)
    inflectionPoints = np.where(np.diff(np.sign(interpolatedMeans2ndDerivative + 1e-18)))[0]  # add small number to
    # avoid float precision errors when approaching zero
    extremumPoints = np.where(np.diff(np.sign(interpolatedMeansDerivative + 1e-18)))[0]

    # Finds smallest local minimum in the region between the two largest local maximums
    minPoints = {}
    maxPoints = {}
    for extremum in extremumPoints:
        if interpolatedMeans2ndDerivative[extremum] > 0:
            minPoints[extremum] = interpolatedMeans[extremum]
        elif interpolatedMeans2ndDerivative[extremum] < 0:
            maxPoints[extremum] = interpolatedMeans[extremum]

    sortedMax = dict(sorted(maxPoints.items(), key=lambda item: item[1]))
    twoLargestMaxima = sorted(list(sortedMax)[-2:])
    sortedMin = dict(sorted(minPoints.items(), key=lambda item: item[1]))
    for point in sortedMin:
        if twoLargestMaxima[1] >= point >= twoLargestMaxima[0]:
            minPoint = point
            break

    # minPoint = np.argmin(interpolatedMeans)

    lower = inflectionPoints[inflectionPoints <= minPoint]
    upper = inflectionPoints[inflectionPoints >= minPoint]
    if len(upper) == 1 or len(lower) == 1:
        raise Exception("Not enough inflection points")

    if createPlot:
        plt.figure(4)
        x = np.linspace(0, len(means) - 1, len(means))
        # plt.plot(x,interpolatedMeans2ndDerivative)
        plt.plot(x, means, 'ro:')
        plt.plot(x, interpolatedMeans)
        for inflectionPoint in inflectionPoints:
            1 + 1
            # plt.axvline(x=inflectionPoint, color='k')
        for extremum in extremumPoints:
            plt.axvline(x=extremum, color='k')
        for minPoint in minPoints:
            plt.axvline(x=minPoint, color='g')
        with fits.open(filename) as hdul:
            plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Ranges")
            plt.savefig(filename.split(".cube")[0] + "_ranges" + ".pdf")
            del hdul[0].data
    return [lower[0], lower[-1], upper[0], upper[-1]]


def findRangesByGaussianFit(means, createPlot=False, createSubPlot=False, modality=2, sigmaMult=1):
    """
    Finds the ranges to compute moment maps from by fitting Gaussians. (Does not work well for modality = 3)
    ----------
    means : list of floats
        The values to fit the Gaussian to.
    createPlot : bool, optional
         (Default value = False)
         Decides if the fit is plotted
    createSubPlot : bool, optional
         (Default value = False)
         Decides if the subfit is plotted
    modality : integer
         (Default value = 2)
         How many Gaussians to fit
    sigmaMult : float
         (Default value = 1)
         How many standard deviations from the peaks to include in the range.
    Returns
    -------
    The ranges to compute momentmaps from.
    """
    if modality == 2:
        parameters = bimodalGaussianMeanFit(means, createPlot, createSubPlot)
    elif modality == 3:
        parameters = trimodalGaussianMeanFit(means, createPlot, createSubPlot)
    else:
        raise Exception("Unsupported modality of Gaussian")
    center1 = parameters[1]
    center2 = parameters[5]
    sigma1 = abs(parameters[2])
    sigma2 = abs(parameters[6])
    ranges = [int(center1 - sigmaMult * sigma1), int(center1 + sigmaMult * sigma1), int(center2 - sigmaMult * sigma2),
              int(center2 + sigmaMult * sigma2)]
    return ranges


def findRangesByRMS(means):
    """Gets the indicies of all intensities larger than the rms in the spectral profile"""
    rootMean = rms(means)
    indicies = [i for i in range(len(means)) if means[i] > rootMean]
    return indicies


##############################################
# Moment maps
##############################################

def computeMoments(filename, ranges, createPlot=False):
    """
    Computes red- and blueshifted moments given a datacube and ranges and joins blue- and rightshifted sides.
    ----------
    filename : String
         The location of the cube data fits file.
    ranges : list of floats
         The start and endpoints of the ranges where moments are to be computed.
    createPlot : bool, optional
         (Default value = False)
         "Plots" the moment map
    Returns
    -------
    Two matricies with the "intensities" making up the blue- and redshifted moment maps.
    """
    with fits.open(filename) as hdul:
        cube = hdul[0].data
        cubeData = np.squeeze(cube)
        cubeSlab1 = cubeData[ranges[0]:ranges[1], :, :]
        cubeSlab2 = cubeData[ranges[2]:ranges[3], :, :]
        moment1 = np.sum(cubeSlab1, axis=0)
        moment2 = np.sum(cubeSlab2, axis=0)
        # moment3 = np.concatenate((moment1[:, :485], moment2[:, 485:]), axis=1)  # replace 485 with xCenter
        if createPlot:
            wcs = WCS(filename)
            if wcs.naxis > 2:
                wcs = wcs.sub(2)
            plt.figure(5)
            plt.subplot(projection=wcs)
            plt.imshow(moment1, origin='lower')
            # plt.colorbar(label=r"Intensity (Jy beam$^{-1}$)")
            plt.colorbar(label=r"Integrated Intensity (Jy beam$^{-1}$ km s$^{-1}$)")
            plt.xlabel("Right Ascension (J2000)")
            plt.ylabel("Declination (J2000)")
            plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Moment 0 Map")
            plt.savefig(filename.split(".cube")[0] + "_moment1" + ".pdf")
            plt.figure(6)
            plt.subplot(projection=wcs)
            plt.imshow(moment2, origin='lower')
            plt.colorbar(label=r"Integrated Intensity (Jy beam$^{-1}$ km s$^{-1}$)")
            plt.xlabel("Right Ascension (J2000)")
            plt.ylabel("Declination (J2000)")
            plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Moment 0 Map")
            plt.savefig(filename.split(".cube")[0] + "_moment2" + ".pdf")
            # plt.figure(7)
            # plt.imshow(moment3, cmap="inferno")
        del hdul[0].data
    return moment1, moment2


def maskedMoment(moment, factor=1):
    """Sets all values lower than factor*noise to 0"""
    moment[moment < factor * computeNoise(moment)] = 0
    plt.figure()
    plt.imshow(moment, origin='lower')
    return moment


def computeMomentsByMax(filename, indicies=[]):
    """Computes a "moment" by for each pixel summing the 10% largest pixels """
    with fits.open(filename) as hdul:
        cube = hdul[0].data
        cubeData = np.squeeze(cube)

        # cubeData = np.nan_to_num(cubeData) ###BLIR LITE KÖNSTIGT med konturerna

        # if len(indicies) > 0:
        #    cubeData = cubeData[indicies, :, :] # oklart om det är bra eller inte

        imageWidth = cubeData.shape[1]
        moment = np.zeros((imageWidth, imageWidth))

        for x in range(cubeData.shape[1]):
            for y in range(cubeData.shape[1]):
                pixelVals = list(-1*cubeData[:, x, y])
                heapify(pixelVals)
                val = 0
                for i in range(cubeData.shape[0] // 10):
                    val += -1*heappop(pixelVals)
                moment[x, y] = val

        wcs = WCS(filename)
        if wcs.naxis > 2:
            wcs = wcs.sub(2)
        plt.figure()
        plt.subplot(projection=wcs)
        plt.imshow(moment, origin='lower')
        # plt.colorbar(label=r"Intensity (Jy beam$^{-1}$)")
        plt.colorbar(label=r"Integrated Intensity (Jy beam$^{-1}$ km s$^{-1}$)")
        plt.xlabel("Right Ascension (J2000)")
        plt.ylabel("Declination (J2000)")
        plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Moment 0 Map (Max)")
        plt.savefig(filename.split(".cube")[0] + "_maxmoment" + ".pdf")
        del hdul[0].data
    return moment


def computeMomentByIndex(filename, indicies, createPlot=True):
    """Computes a moment by summing all frequency indicies"""
    with fits.open(filename) as hdul:
        cube = hdul[0].data
        cubeData = np.squeeze(cube)
        cubeSlab = cubeData[indicies, :, :]
        moment1 = np.sum(cubeSlab, axis=0)
        if createPlot:
            wcs = WCS(filename)
            if wcs.naxis > 2:
                wcs = wcs.sub(2)
            plt.figure()
            plt.subplot(projection=wcs)
            plt.imshow(moment1, origin='lower')
            plt.colorbar(label=r"Integrated Intensity (Jy beam$^{-1}$ km s$^{-1}$)")
            plt.xlabel("Right Ascension (J2000)")
            plt.ylabel("Declination (J2000)")
            plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Moment 0 Map (Index)")
            plt.savefig(filename.split(".cube")[0] + "_indexmoment" + ".pdf")
        del hdul[0].data
    return moment1


##############################################
# Angle finding and contours
##############################################

def refineContours(contours, xCenter, yCenter):
    """
    "Refines" contours by deleting all contours that do not surrond center
    ----------
    contours : contour object from plt.contour
        The contour object from plt.contour
    xCenter : float
        The x-coordinate for the center point of the disc
    xCenter : float
        The y-coordinate for the center point of the disc
    Returns
    -------
    """

    # First iteration to save all contours that surround center of disc
    contoursAroundCenter = []
    for level in contours.collections:
        for kp, path in reversed(list(enumerate(level.get_paths()))):  # loop in reverse since deletions
            if path.contains_point((xCenter, yCenter)):
                contoursAroundCenter.append([level, path])

    # Second iteration to remove all contours that are not within the above contours
    for level in contours.collections:
        for kp, path in reversed(list(enumerate(level.get_paths()))):
            if not path.contains_point((xCenter, yCenter)):
                isWithin = False
                for _, bigPath in contoursAroundCenter:
                    if bigPath.contains_path(path):
                        isWithin = True
                if not isWithin:
                    del (level.get_paths()[kp])

    plt.gcf().canvas.draw()  # uppdatera plotten
    return contours


def plotContours(moment1, moment2, fittedValues, filename, combinedPlot=True):
    """
    Plots the wanted contours by first removing noise, computing the contours and then refining them.
    ----------
    moment1 : matrix
        The first moment map matrix
    moment2 : matrix
        The second moment map matrix
    fittedValues : list of floats
        List of fitted parameters to the Gaussian function.
    combinedPlot : bool, optional
         (Default value = True)
         Plot both contours in same plot
    Returns
    -------
    """
    mmom1 = maskedMoment(moment1)
    mmom2 = maskedMoment(moment2)
    x, y = fittedValues[1:3]
    plt.figure(1337)
    contours1 = plt.contour(mmom1)
    refineContours(contours1, x, y)
    if not combinedPlot:
        plt.figure()
    contours2 = plt.contour(mmom2)
    refineContours(contours2, x, y)
    # findAngleOfOutflow(mmom1,fittedValues)
    # findAngleOfOutflow(mmom2,fittedValues)
    findAngleFromContour(contours1, contours2, fittedValues, filename)


def findAngleOfOutflow(moment, fittedValues,filename, extra="", coordinates=[], useDistance=True):
    """
    Find and plots the directions where intensities are present.
    ----------
    moment : matrix
        The moment map matrix
    fittedValues : list of floats
        List of fitted parameters to the Gaussian function.
    coordinates : list of list of ints
        (Default value: [] (i.e. none))
        Coordinates to ignore when calculating.
    useDistance : boolean
        (Default value = False)
        Whether or not to weight by distance (in the sense that intensities closer to the disc have more weight)

    Returns
    -------
    """
    xCenter, yCenter = fittedValues[1:3]
    imageWidth = moment.shape[0]
    angularIntensities = {}
    noise = computeNoise(moment, partitions=20)
    for i in range(-180, 180):
        angularIntensities[i] = 0
    newMoment = np.zeros((imageWidth, imageWidth))
    for x in range(0, imageWidth):
        for y in range(0, imageWidth):
            if len(coordinates) == 0 or [x, y] not in coordinates:
                if moment[x, y] > noise:
                    distanceFactor = 0
                    if useDistance:
                        distanceFactor = ((x - xCenter) ** 2 + (y - yCenter) ** 2) / ((imageWidth * 0.5) ** 2)
                        if distanceFactor > 1:
                            continue
                    index = np.floor(np.arctan2(x - xCenter, y - yCenter) * 180 / np.pi)
                    angularIntensities[index] += moment[x, y] * (1 - distanceFactor)
                    angularIntensities[index + 180 if index < 0 else -180 + ((180 + index) % 180)] += moment[x, y] * (
                            1 - distanceFactor)
                    newMoment[x, y] = moment[x, y]*(1-distanceFactor)

    # plt.figure()
    # plt.imshow(moment, origin="lower")
    # ¤moment[moment < 10*noise] = 0
    with fits.open(filename) as hdul:
        plt.figure()
        plt.imshow(newMoment, origin="lower")
        plt.scatter(xCenter, yCenter)
        plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Reduced Moment")
        plt.savefig(filename.split(".cube")[0] + "_redmoment" + extra + ".pdf")
        plt.figure()
        ax = plt.subplot(111, polar=True)
        bars = ax.bar((np.array(list(angularIntensities.keys()))) * np.pi / 180, angularIntensities.values(),
                      width=12 * np.pi / 180)
        plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Angular intensities")
        plt.savefig(filename.split(".cube")[0] + "_angles" + extra + ".pdf")
        #print(np.mean(np.array(list(angularIntensities.keys()))*np.array(list(angularIntensities.values()))))
        del hdul[0].data
        # print(angularIntensities)

# Double ended, motsatta sidor har båda summan av sig själv och motsatt sida
def findAngleOfOutflowDE(moment, fittedValues):
    xCenter, yCenter = fittedValues[1:3]
    angularIntensities = {}
    for i in range(-180, 180):
        angularIntensities[i] = 0
    for x in range(0, moment.shape[0]):
        for y in range(0, moment.shape[0]):
            index = np.floor(np.arctan2(y - yCenter, x - xCenter) * 180 / np.pi)
            angularIntensities[index] += (
                    np.nan_to_num(moment[x, y]) * (1 + (x - xCenter) ** 2 + (y - yCenter) ** 2) ** -0.5)
            angularIntensities[index + 180 if index < 0 else -180 + ((180 + index) % 180)] += (
                    np.nan_to_num(moment[x, y]) * (1 + (x - xCenter) ** 2 + (y - yCenter) ** 2) ** -0.5)

    plt.figure()
    # plt.plot(angularIntensities.keys(),angularIntensities.values())
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(angularIntensities.keys(), angularIntensities.values(), width=12 * np.pi / 180)

    # print(angularIntensities)

# Tries to find outflow by fitting a parabola to highest line intensity (WIP)
# Ideas: 
# - add center coordinates found via 2d gauss
def findOutflowWithParabola(filename):
    with fits.open(filename) as hdul:
        fitsData = hdul[0].data
        contMatrix = np.squeeze(fitsData)
        imageWidth = contMatrix.shape[0]
        contMatrix = np.nan_to_num(contMatrix)
        maxLineIntensity = 0

        # From 0 to 179 degrees try to fit parabola
        for i in range(180):
            maxLineIntensity = maxLineIntensity

        del hdul[0].data            

# Finds the angle given contours
def findAngleFromContour(contour1, contour2, fittedValues, filename):
    """From two contours find an angle."""
    xCenter, yCenter = fittedValues[1:3]
    coordinates = []
    for level in contour1.collections: #adds all points of the contours to an array.
        for path in level.get_paths():
            verts = np.array(path.vertices)
            n = len(verts)
            verts = verts[1::int(np.ceil(n / 8))]
            for x, y in verts:
                coordinates.append([x, y])
    for level in contour2.collections:
        for path in level.get_paths():
            verts = np.array(path.vertices)
            n = len(verts)
            verts = verts[1::int(np.ceil(n / 8))]
            for x, y in verts:
                coordinates.append([x, y])

    if len(coordinates) < 10:
        return

    coordinates = np.array(coordinates)
    distanceFromCenter = (coordinates[:, 0] - xCenter) ** 2 + (coordinates[:, 1] - yCenter) ** 2
    prio = np.ceil(100 * distanceFromCenter / np.max(distanceFromCenter))
    # prio = 1/prio # prio close vs prio far away, måste detektera vilken typ av utflöde vi har
    coordinates[0, :] = [xCenter, yCenter]
    prio[0] = 100 * 100 # prioritize center
    res = opt.least_squares(fitWrapper, x0=[1, 100], args=(coordinates[:, 0], coordinates[:, 1], prio))
    coeff = res.x

    # angle of contours
    angle = np.arctan(coeff[0])
    if angle < 0:
        angle += np.pi

    #angle of cont fit
    clockWiseContRotation = fittedValues[5] * 180 / np.pi % 360
    if fittedValues[4] > fittedValues[3]:
        clockWiseContRotation = (clockWiseContRotation + 90) % 360

    # y = kx + m
    # m = y - kx
    # k = tan(angle)
    k = np.tan(-clockWiseContRotation * np.pi / 180) # finds slope of cont line
    m = yCenter - k * xCenter
    print(fittedValues[3], fittedValues[4])

    angleBetweenLines = np.arctan((coeff[0] - k) / (1 + coeff[0] * k)) # finds the angle between the lines
    print(np.abs(angleBetweenLines * 180 / np.pi))
    with fits.open(filename) as hdul:
        imageWidth = np.squeeze(hdul[0].data).shape[1]
        plt.figure(1337)
        x = np.linspace(0, imageWidth, 500)  # replace with actual values
        y1 = np.polyval(coeff, x)
        plt.plot(x, y1, 'k')
        plt.plot(x, linFunc(x, k, m))
        plt.xlim([0, imageWidth])  # replace with actual values
        plt.ylim([0, imageWidth])  # replace with actual values
        plt.title(hdul[0].header.get("OBJECT").replace("_", " ") + " Contours")
        plt.savefig(filename.split(".cube")[0] + "_contours" + ".pdf")
        plt.figure(1)
        plt.plot(x, linFunc(x, k, m))
        del hdul[0].data


##############################################
# Complete analysis from fits files
##############################################

def findMoment(contFile, cubeFile, oneMillionPlots=False):
    """
    Finds and plots the momentmap and contours
    ----------
    contFile : file
        The continuum file of the observation
    cubeFile : file
        Datacube file from the observation
    oneMillionPlots : bool, optional
         (Default value = False)
         Whether to plot all the plots or not
    Returns
    -------
    """
    clearPlots([1, 2, 3, 4, 5, 6, 1337])
    fittedValues = fit2DGaussianToContData(contFile, oneMillionPlots)
    coordinates = getPointsWithinGaussian(fittedValues, 1 / 15, 20, False)
    means = meanSpectralProfile(cubeFile, coordinates, oneMillionPlots)
    ranges = findRangesByInflection(means, cubeFile, oneMillionPlots)
    moment1, moment2 = computeMoments(cubeFile, ranges, oneMillionPlots)
    # computeNoise(moment1, 8)
    # findAngleOfOutflow(moment1,fittedValues)
    # plt.figure(1337)
    plt.figure()
    plt.contour(moment1)
    plt.figure()
    # cont1 = plt.contour(maskedMoment(moment1))
    # cont2 = plt.contour(maskedMoment(moment2))
    plotContours(moment1, moment2, fittedValues, cubeFile)

    # findAngleFromContour(cont1,cont2,fittedValues)
    # wx, wy = w.wcs_pix2world(241.,241.,0) # RA convert to hours hh:mm:ss # DEC convert to degrees:mm:ss
    # TODOo: Fixa rätt koordinatsystem på plottar
    # print('{0} {1}'.format(wx, wy))
    plt.show()


def findMoment2(fittedValues, coordinates, cubeFile, oneMillionPlots=True):
    """Helper function to analyse cube files given cont files"""
    means = meanSpectralProfile(cubeFile, coordinates, oneMillionPlots)
    ranges = findRangesByInflection(means, cubeFile, oneMillionPlots)
    if ranges == "break":
        # plt.show()
        return
    moment1, moment2 = computeMoments(cubeFile, ranges, oneMillionPlots)
    plotContours(moment1, moment2, fittedValues, cubeFile)
    # plt.show()


# nåt sånt
def analyseDir(dir, allPlots=True):
    """Analyses an entire directory of fits files"""
    cubeFiles = []
    contFiles = []
    for filename in os.listdir(dir):  # TO  order cont files together with cube files that look at same object
        if filename.__contains__(".cont") and filename.__contains__(".fits"):
            contFiles.append(dir + "/" + filename)
        elif filename.__contains__(".cube") and filename.__contains__(".fits"):
            cubeFiles.append(dir + "/" + filename)

    if len(contFiles) == 0:
        return

    for contFile in contFiles:
        fittedValues = fit2DGaussianToContData(contFile, allPlots)
        coordinates = getPointsWithinGaussian(fittedValues, 1 / 15, 20, False)
        for cubeFile in cubeFiles:
            if cubeFile.__contains__(contFile.split("_sci")[0]):
                clearPlots([1, 2, 3, 4, 5, 6, 1337])
                try: 
                    findMoment2(fittedValues, coordinates, cubeFile, allPlots)
                except:
                    analasys_failed += 1
                    print("exception occured :(")
    print("done")

def analyseDir2(dir):
    cubeFiles = []
    contFiles = []
    for filename in os.listdir(dir):  # TO  order cont files together with cube files that look at same object
        if filename.__contains__(".cont") and filename.__contains__(".fits"):
            contFiles.append(dir + "/" + filename)
        elif filename.__contains__(".cube") and filename.__contains__(".fits"):
            cubeFiles.append(dir + "/" + filename)

    if len(contFiles) == 0:
        return

    for contFile in contFiles:
        fittedValues = fit2DGaussianToContData(contFile, True)
        coordinates = getPointsWithinGaussian(fittedValues, 1 / 15, 20, False)
        for cubeFile in cubeFiles:
            if cubeFile.__contains__(contFile.split("_sci")[0]):
                clearPlots([1, 2, 3, 4, 5, 6,7,8,9,10,11,1337])
                try:
                    means = meanSpectralProfile(cubeFile, coordinates, True)
                    indicies = findRangesByRMS(means)
                    indexMoment = computeMomentByIndex(cubeFile, indicies)
                    maxMoment = computeMomentsByMax(cubeFile, indicies)
                    findAngleOfOutflow(indexMoment, fittedValues,cubeFile,"_index")
                    findAngleOfOutflow(maxMoment, fittedValues,cubeFile,"_max")
                    # plt.show()
                except:
                    alanasys_failed_2 += 1
                    print("exception occured :(")
    print("done")

def main():
    cubeFile = "Serp/member.uid___A001_X1467_X291.Serpens_South_C7_sci.spw25.cube.I.pbcor.fits"
    contFile = "Serp/member.uid___A001_X1467_X291.Serpens_South_C7_sci.spw25_27_29_31_33_35_37_39_41_43.cont.I.pbcor.fits"
    #cubeFile = "TestData/12CO.pbcor.fits"
    #contFile = "TestData/cont.fits"
    # findMoment("Serp/member.uid___A001_X1467_X291.Serpens_South_C7_sci.spw25_27_29_31_33_35_37_39_41_43.cont.I.pbcor.fits","Serp/member.uid___A001_X1467_X291.Serpens_South_C7_sci.spw33.cube.I.pbcor.fits",True)
    fittedValues = fit2DGaussianToContData(contFile, True)
    coordinates = getPointsWithinGaussian(fittedValues, 1 / 15, 20, False)
    means = meanSpectralProfile(cubeFile, coordinates, True)
    indicies = findRangesByRMS(means)
    # print(indicies)
    plt.figure()
    moment1 = computeMomentByIndex(cubeFile, indicies)
    # plt.figure(10)
    # cont1 = plt.contour(moment1)
    #

    # refineContours(cont1,x,y)
    moment = computeMomentsByMax(cubeFile, indicies)
    findAngleOfOutflow(contFile, moment, fittedValues)
    findAngleOfOutflow(contFile, moment1, fittedValues)
    plt.show()

if __name__ == "__main__":
    main()
