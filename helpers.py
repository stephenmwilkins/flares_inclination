import numpy as np
from numba import jit, njit, float64, int32, prange
from photutils import CircularAperture, aperture_photometry
from scipy.interpolate import interp1d

import SynthObs
from SynthObs.SED import models
import FLARE
import FLARE.filters
from FLARE.photom import lum_to_M, M_to_lum

@njit(float64[:](float64[:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:], int32), parallel=True, nogil=True)
def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, lkernel, kbins):

    """

    Compute the los metal surface density (in Msun/Mpc^2) for star particles inside the galaxy taking
    the z-axis as the los.
    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """
    n = len(s_cood)
    Z_los_SD = np.zeros(n)
    #Fixing the observer direction as z-axis. Use make_faceon() for changing the
    #particle orientation to face-on
    xdir, ydir, zdir = 0, 1, 2
    for ii in prange(n):

        thisspos = s_cood[ii]
        ok = np.where(g_cood[:,zdir] > thisspos[zdir])[0]
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:,xdir] - thisspos[xdir]
        y = thisgpos[:,ydir] - thisspos[ydir]

        b = np.sqrt(x*x + y*y)
        boverh = b/thisgsml

        ok = np.where(boverh <= 1.)[0]
        kernel_vals = np.array([lkernel[int(kbins*ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok]*thisgZ[ok]/(thisgsml[ok]*thisgsml[ok]))*kernel_vals) #in units of Msun/Mpc^2


    return Z_los_SD


def get_spherical_from_cartesian(coords):

    x, y, z = coords

    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    theta = np.arctan2(np.sqrt(xy), z) # for elevation angle defined from Z-axis down
    phi = np.arctan2(y, x)

    return r, theta, phi


def get_cartesian_from_spherical(t_angles):

    x = np.sin(t_angles[0])*np.cos(t_angles[1])
    y = np.sin(t_angles[0])*np.sin(t_angles[1])
    z = np.cos(t_angles[0])

    return np.array([x, y, z])

def get_rotation_matrix(i_v, unit=None):
    # This solution is from ---
    # https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis

    # This uses the Rodrigues' rotation formula for the re-projection

    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [0.0, 0.0, 1.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )



def ang_mom_vector(this_mass, this_cood, this_vel):

    #Get the angular momentum unit vector
    L_tot = np.array([this_mass]).T * np.cross(this_cood, this_vel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis = 0)**2))
    L_unit = np.sum(L_tot, axis = 0)/L_tot_mag

    return L_unit


def kappa(this_smass, this_scoord, this_svel):


    L_tot = np.array([this_smass]).T*np.cross(this_scoord, this_svel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis = 0)**2))

    L_unit = np.sum(L_tot, axis = 0)/L_tot_mag

    R_z = np.cross(this_scoord,L_unit)
    absR_z = np.sqrt(np.sum(R_z**2, axis = 1))
    mR = this_smass*absR_z
    K = np.nansum(this_smass*np.sum(this_svel**2, axis = 1))

    L = np.sum(L_tot*L_unit, axis = 1)
    L_co = np.copy(L)
    co = np.where(L_co > 0.)
    L_co = L_co[co]

    L_mR = (L/mR)**2
    L_co_mR = (L_co/mR[co])**2
    Krot = np.nansum(this_smass*L_mR)/K


    Kco = np.nansum(this_smass[co]*L_co_mR)/K


    return Kco, Krot


def lum(MetSurfaceDensities, Masses, Ages, Metallicities, DTM, kappa=0.0795, BC_fac=1.0, IMF = 'Chabrier_300', filters = ['FAKE.TH.FUV'], Type = 'Total', log10t_BC = 7., extinction = 'default'):

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM = ('simple', {'slope': -1.})    #Define dust curve for ISM
        model.dust_BC = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM = ('Starburst_Calzetti2000', {''})
        model.dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM = ('SMC_Pei92', {''})
        model.dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM = ('MW_Pei92', {''})
        model.dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM = ('MW_N18', {''})
        model.dust_BC = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    # --- create rest-frame luminosities
    F = FLARE.filters.add_filters(filters, new_lam = model.lam)
    model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz

    DustSurfaceDensities = DTM * MetSurfaceDensities

    if Type == 'Total':
        tauVs_ISM = kappa * DustSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
        tauVs_BC = BC_fac * (Metallicities/0.01)
        fesc = 0.0

    elif Type == 'Pure-stellar':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = np.zeros(len(Masses))
        fesc = 1.0

    elif Type == 'Intrinsic':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = np.zeros(len(Masses))
        fesc = 0.0

    elif Type == 'Only-BC':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = BC_fac * (Metallicities/0.01)
        fesc = 0.0

    else:
        ValueError(F"Undefined Type {Type}")

    Lnu = models.generate_Lnu(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, fesc = fesc, log10t_BC = log10t_BC) # --- calculate rest-frame Luminosity. In units of erg/s/Hz
    Lums = list(Lnu.values())

    return Lums


def lum_from_stars(MetSurfaceDensities, Masses, Ages, Metallicities, DTM, kappa=0.0795, BC_fac=1.0, IMF = 'Chabrier_300', filters = ['FAKE.TH.FUV'], Type = 'Total', log10t_BC = 7., extinction = 'default'):

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM = ('simple', {'slope': -1.})    #Define dust curve for ISM
        model.dust_BC = ('simple', {'slope': -1.})     #Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM = ('Starburst_Calzetti2000', {''})
        model.dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM = ('SMC_Pei92', {''})
        model.dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM = ('MW_Pei92', {''})
        model.dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM = ('MW_N18', {''})
        model.dust_BC = ('MW_N18', {''})
    else: ValueError("Extinction type not recognised")

    # --- create rest-frame luminosities
    F = FLARE.filters.add_filters(filters, new_lam = model.lam)
    model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz

    DustSurfaceDensities = DTM * MetSurfaceDensities

    if Type == 'Total':
        tauVs_ISM = kappa * DustSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
        tauVs_BC = BC_fac * (Metallicities/0.01)
        fesc = 0.0

    elif Type == 'Pure-stellar':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = np.zeros(len(Masses))
        fesc = 1.0

    elif Type == 'Intrinsic':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = np.zeros(len(Masses))
        fesc = 0.0

    elif Type == 'Only-BC':
        tauVs_ISM = np.zeros(len(Masses))
        tauVs_BC = BC_fac * (Metallicities/0.01)
        fesc = 0.0

    else:
        ValueError(F"Undefined Type {Type}")

    Lnu = {f: models.generate_Lnu_array(model, Masses, Ages, Metallicities, tauVs_ISM, tauVs_BC, F, f, fesc = fesc) for f in filters} # --- calculate rest-frame Luminosity. In units of erg/s/Hz
    Lums = list(Lnu.values())

    return Lums


def calc_axes(coods):
    """
    Args:
        coods - normed coordinates

    Returns:
        [a, b, c]:
        e_vectors:
    """

    I = np.zeros((3, 3))

    I[0,0] = np.sum(coods[:,1]**2 + coods[:,2]**2)
    I[1,1] = np.sum(coods[:,0]**2 + coods[:,2]**2)
    I[2,2] = np.sum(coods[:,1]**2 + coods[:,0]**2)

    I[0,1] = I[1,0] = - np.sum(coods[:,0] * coods[:,1])
    I[1,2] = I[2,1] = - np.sum(coods[:,2] * coods[:,1])
    I[0,2] = I[2,0] = - np.sum(coods[:,2] * coods[:,0])

    e_values, e_vectors = np.linalg.eig(I)

    sort_idx = np.argsort(e_values)

    e_values = e_values[sort_idx]
    e_vectors = e_vectors[sort_idx,:]

    a = ((5. / (2 * len(coods))) * (e_values[1] + e_values[2] - e_values[0]))**0.5
    b = ((5. / (2 * len(coods))) * (e_values[0] + e_values[2] - e_values[1]))**0.5
    c = ((5. / (2 * len(coods))) * (e_values[0] + e_values[1] - e_values[2]))**0.5

#     print a, b, c

    return [a,b,c], e_vectors

def create_image(pos, Ndim, i, j, imgrange, ls, smooth):
    # Define x and y positions for the gaussians
    Gx, Gy = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], Ndim),
                         np.linspace(imgrange[1][0], imgrange[1][1], Ndim))
    # Initialise the image array
    gsmooth_img = np.zeros((Ndim, Ndim))
    # Loop over each star computing the smoothed gaussian distribution for this particle
    for x, y, l, sml in zip(pos[:, i], pos[:, j], ls, smooth):
        # Compute the image
        g = np.exp(-(((Gx - x) ** 2 + (Gy - y) ** 2) / (2.0 * sml ** 2)))
        # Get the sum of the gaussian
        gsum = np.sum(g)
        # If there are stars within the image in this gaussian add it to the image array
        if gsum > 0:
            gsmooth_img += g * l / gsum
    # img, xedges, yedges = np.histogram2d(pos[:, i], pos[:, j], bins=nbin, range=imgrange, weights=ls)
    return gsmooth_img



def calc_halflightradius(coords, L, sml, z):

    # Define comoving softening length in pMpc
    csoft = 0.001802390 / (0.6777 * (1 + z)) * 1e3
    # Define width (in Mpc)
    ini_width = 62
    # Compute the resolution
    ini_res = ini_width / csoft
    res = int(np.ceil(ini_res))
    # Compute the new width
    width = csoft * res

    # Define range and extent for the images
    imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))
    imgextent = [-width / 2, width / 2, -width / 2, width / 2]
    # Set up aperture objects
    positions = [(res / 2, res / 2)]
    app_radii = np.linspace(0.001, res / 4, 500)  # 500 apertures out to 1/4 the image width
    apertures = [CircularAperture(positions, r=r) for r in app_radii]
    app_radii *= csoft

    tot_l = np.sum(L)
    img = create_image(coords,res, 0, 1, imgrange, L, sml)

    hlr = get_img_hlr(img, apertures, tot_l, app_radii, res, csoft)

    return hlr


def get_img_hlr(img, apertures, tot_l, app_rs, res, csoft):
    # Apply the apertures
    phot_table = aperture_photometry(img, apertures, method='subpixel', subpixels=5)
    # Extract the aperture luminosities
    row = np.lib.recfunctions.structured_to_unstructured(np.array(phot_table[0]))
    lumins = row[3:]
    # Get half the total luminosity
    half_l = tot_l / 2
    # Interpolate to increase resolution
    func = interp1d(app_rs, lumins, kind="linear")
    interp_rs = np.linspace(0.001, res / 4, 10000) * csoft
    interp_lumins = func(interp_rs)
    # Get the half mass radius particle
    hlr_ind = np.argmin(np.abs(interp_lumins - half_l))
    hlr = interp_rs[hlr_ind]
    return hlr
