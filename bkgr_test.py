import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.stats import mad_std,SigmaClip,sigma_clipped_stats
from astropy.convolution import convolve
from photutils.background import Background2D,MedianBackground
from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture,aperture_photometry
from photutils.segmentation import make_2dgaussian_kernel,detect_sources,detect_threshold
from photutils.utils import circular_footprint
import numpy as np

#### Setup some values to use later
norm = ImageNormalize(stretch=SqrtStretch())
sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
bkg_estimator = MedianBackground(sigma_clip=sigma_clip)

### Open a stacked image from a Full Moon test
hdu = fits.open("Stacked 100 full moon.fit")
image = np.array(hdu[0].data,dtype=np.double)
image_width,image_height = hdu[0].data.shape
print(image_width,image_height)

#### Create a mask to hide the Sun/Moon
threshold = detect_threshold(image, nsigma=10.0, sigma_clip=sigma_clip)
segment_img = detect_sources(image, threshold, npixels=1000)
# plt.imshow(segment_img, cmap=segment_img.cmap, interpolation='nearest', origin='lower')
# plt.show()
footprint = circular_footprint(radius=25)
mask = segment_img.make_source_mask(footprint=footprint)
#mask = segment_img.make_source_mask()

#### Variables for the 2D background estimation
boxes = [(5,5), (15,15), (25,25), (50,50), (100,100), (200,200), (250,250)]
filts = [(3,3), (3,3), (5,5), (5,5), (5,5), (11,11), (13,13)]
####
cleaned = image
for i in range(len(boxes)):
    print(boxes[i],filts[i])
    ## background estimation
    ## NOTE: Remove bkg_estimator for default instead of median background
    bkg = Background2D(cleaned, boxes[i], mask=mask, filter_size=filts[i], sigma_clip=sigma_clip, edge_method='crop', bkg_estimator=bkg_estimator)
    ## subtract background from image
    cleaned -= bkg.background
    ## add back negative subtraction
    #cleaned += np.abs(np.min(cleaned))
    ## clean up any NaN or inf values
    cleaned[~np.isfinite(cleaned)] = 0

#### Remove the median to set the background to ~0
_,median,_ = sigma_clipped_stats(cleaned,mask=mask,sigma=3.0)
cleaned -= median
# plt.imshow(cleaned, norm=norm, cmap='gray_r', origin='lower')
# plt.show()

#### Save the output
hdu[0].data = cleaned
hdu.writeto('bkgr_out.fits',overwrite=True)

#### Locate some stars
std = mad_std(cleaned)
daofind = DAOStarFinder(fwhm=3, threshold=10 * std, peakmax=30000)
sources = daofind(cleaned, mask=mask)
sources.sort('flux')
sources.reverse()
#sources = sources[0:25]
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
#print(positions)
apertures = CircularAperture(positions, r=8.0)
phot_table = aperture_photometry(cleaned, apertures)
print(phot_table)

#### Plot the result
plt.imshow(cleaned, norm=norm, cmap='gray_r', origin='lower')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
plt.show()
