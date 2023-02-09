#Authored by : Pongporamat C. 
#Updated by Surasak C.


import numpy as np
import pandas as pd
import uuid
import shutil


import os
import subprocess
from numba import jit
from tqdm.notebook import tqdm

import utm
from osgeo import ogr, gdal, gdal_array, gdalconst
import geopandas as gpd
import shapely
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon, mapping
import rasterio
from rasterio import Affine, transform
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.io import MemoryFile
import pyproj
from pyproj import Proj, CRS

from skimage import filters, exposure
from skimage.io import imsave
import matplotlib.pyplot as plt

from zkyhaxpy import io_tools

np.seterr(divide='ignore', invalid='ignore')

PROJECT_47 = pyproj.Transformer.from_crs(
    'epsg:4326',   # source coordinate system
    'epsg:32647',  # destination coordinate system
    always_xy=True # Must have
).transform

PROJECT_48 = pyproj.Transformer.from_crs(
    'epsg:4326',   # source coordinate system
    'epsg:32648',  # destination coordinate system
    always_xy=True # Must have
).transform 

def pixel_row_col_to_xy(in_row, in_col, in_transform, in_crs):
    '''

    Return x, y of given row, col of a pixel from a specific transform.

    Parameters
    ----------
    in_row: int
        the row number of a pixel
    in_col: int
        the column number of a pixel
    in_transform: rasterio transform
        the transform of a original raster of the pixel
    in_crs: crs or str
        the CRS of the original raster
        
    Returns
    -------
    (x, y): tuple
        a tuple of x and y of the pixel
	'''
 
    t = in_transform * Affine.translation(0.5, 0.5) # reference the pixel centre
    rc2xy = lambda r, c: (c, r) * t 
    x,y = rc2xy(in_row, in_col)
    
    return (x, y)

    
    
def xy_to_latlon(in_x, in_y, in_crs, in_zone):
    pp = Proj( in_crs, proj="utm", zone=in_zone)
    out_lon, out_lat = pp(in_x, in_y, inverse=True)
    return (out_lat, out_lon)
        

def extract_bits(img, position):
    """
    Contributed by Pongporamat Charuchinda
    Extract specific bit(s)
    

    Parameters
    ----------
    img: numpy array (M, N)
        QA image.
    position: tuple(int, int) or int
        Bit(s)'s position read from Left to Right (if tuple)
    
    Examples
    --------
    >>> extract_bits(qa_img, position=(6, 5)) # For cloud confidence
    
    Returns
    -------
    None
    """    
    if type(position) is tuple:
        bit_length = position[0]-position[1]+1
        bit_mask = int(bit_length*"1", 2)
        return ((img>>position[1]) & bit_mask).astype(np.uint8)
    
    elif type(position) is int:
        return ((img>>position) & 1).astype(np.uint8)
    
    


def resample_raster(path_file_input, path_file_output, upscale_factor = 0.25, resampling='bilinear'):
    '''
    Resample a raster of one resolution into another resolution with the same projection.
    If upscale_factor > 1, upscaling will be performed.
    If upscale_factor < 1, downscaling will be performed.
    '''
    assert(upscale_factor > 0)

    if resampling == 'bilinear':
        resampling = rasterio.enums.Resampling.bilinear
    
    with rasterio.open(path_file_input) as dataset:
        
        profile_resample = dataset.profile

        # resample data to target shape
        arr_resample = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=resampling
        )

        # scale image transform
        n_bands, h_resample, w_resample = arr_resample.shape
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / arr_resample.shape[-1]),
            (dataset.height / arr_resample.shape[-2])
        )

        profile_resample.update({
            'transform':transform,
            'width':w_resample,
            'height':h_resample,
        })


    with rasterio.open(path_file_output, mode='w', **profile_resample) as dst:
        for i in range(n_bands):
            dst.write(arr_resample[i], i+1)

    print(f'{path_file_output} has been created.')
    
    
    



def reproject_raster_from_ref(src_path, dest_path, ref_path, dest_dtype='src', driver_nm='GTiff'):
    '''

    Reproject a source raster into a destination raster with reference raster's transform.
    
    Parameters
    ----------
    src_path: str
        A path of the source raster
        
    dest_path: str
        A path of the destination raster to be saved
        
    ref_path: str
        A path of the reference raster
        
    dest_dtype: str
        Data type of the destination raster.
        
    Returns
    -------
    None

	'''  
  

    inputfile = src_path
    input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
    inputProj = input.GetProjection()
    inputTrans = input.GetGeoTransform()

    
    reference = gdal.Open(ref_path, gdalconst.GA_ReadOnly)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    
    x = reference.RasterXSize 
    y = reference.RasterYSize

    if dest_dtype=='src':
        dest_dtype=input.GetRasterBand(1).DataType
    elif dest_dtype=='ref':
        dest_dtype=reference.GetRasterBand(1).DataType
    
    driver= gdal.GetDriverByName(driver_nm)
    output = driver.Create(dest_path, x, y, 1, dest_dtype)
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)

    gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Bilinear)

    del output





def df_to_gdf(df, geometry, drop_old_geom_col=False, drop_z=True):
    '''
    INPUT 
    df : dataframe with a column containing geometry in wkt format
    geometry : str specifies column name of geometry in df
    
    OUTPUT
    return :GeoPandas's DataFrame (gdf)
    '''
    df = df.copy()
    df['geometry'] = df[geometry].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs={'init' : 'epsg:4326'})
    del(df)
    if drop_old_geom_col==True:
        gdf = gdf.drop(columns=[geometry]).copy()
        
    if drop_z==True:
        if gdf['geometry'].has_z.sum() > 0:
            gdf_has_z = gdf[gdf['geometry'].has_z]
            for s_idx, s_row in tqdm(gdf_has_z.iterrows(), 'Removing Z axis...'):
                gdf.loc[s_idx,'geometry'] = shapely.wkb.loads(shapely.wkb.dumps(s_row['geometry'], output_dimension=2))
        
        
    return gdf


def shape_to_raster(in_shp, out_tif, ref_tif, no_data=0, all_touched=False, attribute=None, burn_val=1, datatype=gdal.GDT_Byte):
    #Code by PongporC 23/Jan/2020
    #Modified by Surasak C.
   
   
    '''
    Rasterize a shape file into GeoTiff file.
   
    inputs
    ----------------------------------
    in_shp :
        path of input shapefile
       
    out_tif :
        path of output tif
       
    ref_tif :
        path of reference tif
       
    no_data :
        value for no data
       
    all_touched :
        True or False
       
    attribute :
        a column name in the shape file to be rasterized
       
    burn_val :
        value to be filled in raster if no attribute is provided
       
    datatype :
        gdal's datatype
      
    '''

    if attribute:
        burn_val = None

    InputVector = in_shp
    OutputImage = out_tif

    RefImage = ref_tif

    gdalformat = 'GTiff'
   
           
       
    ##########################################################
    # Get projection info from reference image
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

    # Open Shapefile
    Shapefile = ogr.Open(InputVector)
    Shapefile_layer = Shapefile.GetLayer()

    # Rasterize
    print("Rasterizing shapefile...")
    Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype, options=['COMPRESS=DEFLATE'])
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform())

    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(no_data)
   
   
   
    if attribute:
        options=[f"ATTRIBUTE={attribute}", f"ALL_TOUCHED={str(all_touched).upper()}"]
        gdal.RasterizeLayer(Output, [1], Shapefile_layer, options=options)   
    else:
        options=[f"ALL_TOUCHED={str(all_touched).upper()}"]
        gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burn_val], options=options)   
   

    # Close datasets
    Band = None
    Output = None
    Image = None
    Shapefile = None

    # Build image overviews
    subprocess.call("gdaladdo --config COMPRESS_OVERVIEW DEFLATE "+OutputImage+" 2 4 8 16 32 64", shell=True)
    print("Done.")



def calculate_polygon_area_from_lat_long(multi_polygon):
    #Code by PongporC 23/Jan/2020
    ''' Return the area of polygon in square metre unit
        "multi_polygon" is the string of latlon 
    '''
    # Create polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    coords = split_polygon(multi_polygon)
    for item in coords:
        coord = utm.from_latlon(float(item.split(" ")[1]), float(item.split(" ")[0]))[0:2]
        ring.AddPoint(coord[0], coord[1])
    # Find polygon area
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    polygon = ogr.CreateGeometryFromWkt(poly.ExportToWkt())
    area = polygon.GetArea()
    return area


def raster_to_jpg(output_path, raster=None, raster_path=None, save_im=None, band=1, min_value=-1.0, max_value=1.0, na_value=0.0):  
    #Code by PongporC 23/Jan/2020
    #   
    if raster == None and raster_path != None and save_im==None:
        raster = gdal.Open(raster_path)
        save_im = raster.GetRasterBand(band).ReadAsArray()
    elif raster != None and raster_path == None and save_im==None:
        save_im = raster.GetRasterBand(band).ReadAsArray()
    elif raster == None and raster_path == None and save_im!=None:    
        save_im = save_im
    else :
        print('Invalid input')
        return
    
    max_value = max_value
    min_value = min_value
    
    #fill nan with 0.0
    save_im = np.where(save_im == save_im, save_im, na_value)
    
    #override out-of-bound values with min/max values
    save_im = np.where(save_im > max_value, max_value, save_im)
    save_im = np.where(save_im < min_value, min_value, save_im)
    

    mask = save_im!=na_value
    save_im = (255*((save_im-save_im.min())/(save_im.max()-save_im.min()))).astype('uint8')
    save_im = exposure.equalize_hist(save_im)
    save_im = exposure.equalize_hist(save_im, mask=mask)
    save_im = (255*((save_im-save_im.min())/(save_im.max()-save_im.min()))).astype('uint8')
    imsave(output_path, save_im)    


def show_raster(raster, band=1, min_pctl=2, max_pctl=98, figsize=(10, 10), cmap=None, show_cbar=False):
    '''

    Display raster as a matplotlib's plot.

    Parameters
    ----------
    raster: str or array-like
        A path of raster or an array of raster  
    band: int
        Band ID to be shown in case a path of raster is given.
    min_pctl (max_pctl): int
        Min (Max) percentile to be filterred out.
    figsize: tuple (h, w)
        A tuple for plot figure size (Height, Width)
    show_cbar: boolean
        If True, show cbar

    Returns
    -------
    None

	'''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    
    if type(raster) == np.ndarray:
        arr_raster = raster
    else:
        with rasterio.open(raster) as raster:
            arr_raster = raster.read(band)    
        
    vmin, vmax = np.nanpercentile(arr_raster, (min_pctl, max_pctl))    
    im = ax.imshow(arr_raster, vmin=vmin, vmax=vmax, cmap=cmap)
    
    if show_cbar==True:
        cbar = ax.figure.colorbar(im)
    



def adjust_brightness_contrast(arr_img, alpha, beta):

    '''
    Adjust brightness and constrast of given image array according to basic formular 
    "g(x)=αf(x)+β"

    '''

    arr_out = np.clip(alpha*arr_img + beta, 0, 255)
    return arr_out



def split_polygon(multi_polygon):
    #Code by PongporC 23/Jan/2020
    ''' split the polygon string in excel cell into a list of lat and long
        "multi_polygon" is the string of latlon 
    '''
    polygon_list = multi_polygon.split("(")[3].split(")")[0].split(",")
    return polygon_list



    
def create_polygon_from_wkt(wkt_polygon, crs="epsg:4326", to_crs=None):
    """
    Create shapely polygon from string (wkt format) "MULTIPOLYGON(((...)))"
    https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects/127432#127432

    Parameters
    ----------
    wkt_polygon: str
        String of polygon (wkt format).
    crs: str
        wkt_polygon's crs (should be "epsg:4326").
    to_crs: str (optional), default None
        Re-project crs to "to_crs".

    Examples
    --------
    >>> create_polygon_from_wkt(wkt_polygon)
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32647")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32648")
    >>> create_polygon_from_wkt(wkt_polygon, to_crs="epsg:32660")
    >>> create_polygon_from_wkt(wkt_polygon, crs="epsg:32647", to_crs="epsg:4326")

    Returns
    -------
    Polygon
    """
    polygon = wkt.loads(wkt_polygon)
    if to_crs is not None:
        if crs == "epsg:4326":
            if to_crs == "epsg:32647":
                polygon = transform(PROJECT_47, polygon)
            elif to_crs == "epsg:32648":
                polygon = transform(PROJECT_48, polygon)
        else:
            project = pyproj.Transformer.from_crs(
                crs,     # source coordinate system
                to_crs,  # destination coordinate system
                always_xy=True # Must have
            ).transform
            polygon = transform(project, polygon)
    return polygon
    
    
def check_gdf_in_geom(gdf, geom, simplify_geom_tole=0.0005, check_using_centroid=True):
    '''

    To check whether geometries in a GeoDataFrame are in a Shapely geometry or not.
    Both gdf and geom must have a same crs.

    Parameters
    ----------
    gdf: A GeoDataFrame 
        A GeoDataFrame to be checked

    geom: A shapely geometry
        A shapely geometry to be checked against input gdf
        
    simplify_geom_tole: float or None
        If a float between 0-1 given, the geom will be simplified using this tolerance.

    check_using_centroid: boolean
        If True, geometries in gdf will be converted into centroid before checking.
        
    Returns
    -------
    A pandas series of boolean.
	'''
    gdf = gdf.copy()
    if check_using_centroid:
        gdf['geometry'] = gdf.centroid    
    
    if simplify_geom_tole:
        geom = geom.simplify(simplify_geom_tole)
        
    s_geom = gpd.GeoSeries([geom])
            
    s_chk_result = s_geom.apply(lambda val: gdf['geometry'].within(val)).squeeze()
    s_chk_result.name = 'in_geom_f'
    
    return s_chk_result


def wkt_to_geometry(in_str_wkt, in_ori_crs="epsg:4326", in_target_crs=None):
    '''
    Transform a wkt string into shapely geometry 
    
    Parameters
    ----------
    in_str_wkt: str
        String of polygon (wkt format).
    in_ori_crs: str
        wkt_polygon's crs (should be "epsg:4326").
    in_target_crs: str or CRS (optional), default None
        Re-project crs to "to_crs".

    Examples
    --------
    >>> wkt_to_geometry(in_str_wkt)
    >>> wkt_to_geometry(in_str_wkt, in_target_crs="epsg:32647")
    >>> wkt_to_geometry(in_str_wkt, in_target_crs="epsg:32648")
    >>> wkt_to_geometry(wkt_polygon, in_target_crs="epsg:32660")
    >>> wkt_to_geometry(wkt_polygon, in_target_crs="epsg:32647", in_target_crs="epsg:4326")

    Returns
    -------
    Shapely geometry
    '''

    proj_47 = pyproj.Transformer.from_crs(
        'epsg:4326',   # source coordinate system
        'epsg:32647',  # destination coordinate system
        always_xy=True # Must have
    ).transform

    proj_48 = pyproj.Transformer.from_crs(
        'epsg:4326',   # source coordinate system
        'epsg:32648',  # destination coordinate system
        always_xy=True # Must have
    ).transform 


    out_geom = wkt.loads(in_str_wkt)
    if in_target_crs is not None:
        if in_ori_crs == "epsg:4326":
            if in_target_crs == "epsg:32647":
                out_geom = shapely.ops.transform(proj_47, out_geom)
            elif in_target_crs == "epsg:32648":
                out_geom = shapely.ops.transform(proj_48, out_geom)
        else:
            project = pyproj.Transformer.from_crs(
                in_ori_crs,     # source coordinate system
                in_target_crs,  # destination coordinate system
                always_xy=True # Must have
            ).transform
            out_geom = shapely.ops.transform(project, out_geom)
    return out_geom  





def create_row_col_arr(in_shape):
    '''
    Generate a numpy array of row and column mapping corresponding to the input array

    Parameters
    ----------
    in_arr: numpy array
        a numpy array

    '''

    nrow, ncol = in_shape
    arr_row = np.array([np.arange(0, nrow, 1)] * ncol).T
    arr_col = np.array([np.arange(0, ncol, 1)] * nrow)
    out_arr_row_col = np.array([arr_row, arr_col]).astype(np.int16)

    return out_arr_row_col



def create_row_col_mapping_raster(in_raster, out_raster_path=None, out_mem=None, out_nodata_value = -1000, **kwargs):
    '''
    Create row & col mapping of given raster. Output can be either an actual file or an on-memory file (rasterio's dataset).
    
    Parameters
    ----------
    in_raster: str or rasterio.io.DatasetReader
        a raster to be extracted

    out_raster_path: str of path
        a path for output raster

    out_mem: True or False
        if True, return as a memory file's dataset
    
    '''

    if 'tmp_folder' in kwargs.keys():
        tmp_folder=kwargs['tmp_folder']
    else:
        tmp_folder = None

    if type(in_raster) != rasterio.io.DatasetReader:
        tmp_ds = rasterio.open(in_raster, 'r')
    else:
        tmp_ds = in_raster

    tmp_arr_row_col = create_row_col_arr(tmp_ds.shape)
   
    with rasterio.Env():
        profile = tmp_ds.profile
        profile.update(
            dtype=rasterio.int16,
            count=2,
            compress='lzw',
            nodata=out_nodata_value)

        if out_raster_path:
            io_tools.create_folders(out_raster_path)
            with rasterio.open(out_raster_path, 'w', **profile) as dst:
                #row
                dst.write(tmp_arr_row_col[0], 1)
                dst.set_band_description(1, 'row')

                #col
                dst.write(tmp_arr_row_col[1], 2)
                dst.set_band_description(2, 'col')
            print(f'{out_raster_path} has been created')
            return out_raster_path

            
        elif (out_mem==False) & (out_raster_path==None):
            if tmp_folder==None:                                                
                if os.path.exists('/tmp'):
                    out_raster_folder = os.path.join('/tmp', uuid.uuid4().hex)
                else:
                    out_raster_folder = os.path.join('c:', 'tmp', uuid.uuid4().hex)
            else:                
                out_raster_folder = os.path.join(tmp_folder, uuid.uuid4().hex)
            out_raster_path = os.path.join(out_raster_folder, f'rowcol_map_{os.path.basename(in_raster)}')
            io_tools.create_folders(out_raster_path)
            with rasterio.open(out_raster_path, 'w', **profile) as dst:
                #row
                dst.write(tmp_arr_row_col[0], 1)
                dst.set_band_description(1, 'row')

                #col
                dst.write(tmp_arr_row_col[1], 2)
                dst.set_band_description(2, 'col')
            print(f'{out_raster_path} has been created')
            return out_raster_path
            
        elif out_mem==True:
            memfile = MemoryFile()
            ds_mem = memfile.open( **profile)
            
            #row
            ds_mem.write(tmp_arr_row_col[0], 1)
            ds_mem.set_band_description(1, 'row')

            #col
            ds_mem.write(tmp_arr_row_col[1], 2)
            ds_mem.set_band_description(2, 'col')
            return ds_mem
                







def get_pix_row_col(in_polygon, in_row_col_mapping_raster, in_crs_polygon='epsg:4326', all_touched=False):
    '''
    To get all row & col of pixels for given (single) polygon that located in given raster.
    Using not all touched as masking option. If no pixel is found, will get pixel that contains use polygon's centroid instead.
    If input polygon is a point, will use all touched instead.

    Parameters
    ----------
    in_polygon: str or shapely polygon like
        a wkt string or polygon to extract

    in_raster: str or rasterio.io.DatasetReader or  rasterio.io.DatasetWriter
        a raster to be extracted

    Returns
    -------
    A 2-D array of pixel row & col [[row_1, col_1], [row_2, col_2], [row_3, col_3], ...]
    '''


    #Get rasterio dataset if a path is given
    if type(in_row_col_mapping_raster) in ([rasterio.io.DatasetReader, rasterio.io.DatasetWriter]):
        tmp_ds = in_row_col_mapping_raster
        
    else:    
        tmp_ds = rasterio.open(in_row_col_mapping_raster, 'r')
        

    profile = tmp_ds.profile
    nodata_val = profile['nodata']
    is_polygon_overlap = False

    #Get shapely geometry the input polygon is WKT
    if type(in_polygon) == str:
        tmp_polygon = wkt_to_geometry(in_polygon, in_crs_polygon, tmp_ds.crs)
    else:
        tmp_polygon = in_polygon

    #Get pixels' row & col by masking with row col mapping raster
    
    try:
        if tmp_polygon.geom_type=='Point':
            arr_row_col, _ = mask(tmp_ds, [tmp_polygon], crop=True, all_touched=True, nodata=nodata_val)    
        else:
            arr_row_col, _ = mask(tmp_ds, [tmp_polygon], crop=True, all_touched=all_touched, nodata=nodata_val)    

        arr_row_col = np.where(arr_row_col == nodata_val, np.nan, arr_row_col)
        arr_row_col = arr_row_col.reshape(2, -1).T
        arr_row_col = arr_row_col[(~np.isnan(arr_row_col)).all(axis=1)]

        #if polygon is overlapping the raster but the polygon is smaller than the pixel, using pixel that contains centroid instead
        if len(arr_row_col) == 0:
            polygon_centroid = tmp_polygon.centroid
            centroid_x = polygon_centroid.x
            centroid_y = polygon_centroid.y         
            arr_row_col, _ = mask(tmp_ds, [polygon_centroid], crop=True, all_touched=True, nodata=nodata_val)   
            arr_row_col = arr_row_col.reshape(2, -1).T

        assert(len(arr_row_col) > 0)
        is_polygon_overlap = True
        nbr_pixels = len(arr_row_col)
        tmp_ds.close()
        del(tmp_ds)
    except Exception as e:
        #If input polygon does not overlap the raster, return nan for row & col
        if str(e) == 'Input shapes do not overlap raster.':
            is_polygon_overlap = False    
            arr_row_col = np.array([[np.nan, np.nan]])    
            nbr_pixels = 0
            tmp_ds.close()
            del(tmp_ds)
        else:
            raise(e)
    
    return is_polygon_overlap, nbr_pixels, arr_row_col.astype(np.float64)





@jit(nopython=True)
def extract_values_from_2d_array_with_row_col_numba(arr_2d_value, arr_row_col):
    '''
    Get pixel values for given 2d-array (1-band raster)

    inputs
    ---------------------------------
    arr_2d_value: 2d numpy array (y, x)
        an array of raster values

    arr_row_col: 2d numpy array
        an array of pixels' row-id & col-id 

    '''
    list_values = []
    for (row, col) in arr_row_col:
        list_values.append(arr_2d_value[row, col])

    return list_values




@jit(nopython=True)
def __extract_values_from_3d_array_with_row_col_numba(arr_3d_value, arr_row_col):
    '''
    Get pixel values for given 3d-array (multi-band raster)

    inputs
    ---------------------------------
    arr_3d_value: 3d numpy array (m, y, x)
        an array of multibands raster values. (same format as output of rasterio.read([band_ids]) from rasterio's dataset )

    arr_row_col: 2d numpy array
        an array of pixels' row-id & col-id 

    '''

    list_values = []
    for arr_2d_value in arr_3d_value:
        list_values.append(extract_values_from_2d_array_with_row_col_numba(arr_2d_value, arr_row_col))

    return np.array(list_values).T




@jit(nopython=True)
def __reformat_list_row_col_for_df(in_list_polygon_id, in_list_arr_row_col):
    
    
    for i in range(0, len(in_list_arr_row_col)):        
        id_tmp = in_list_polygon_id[i]
        arr_row_col_tmp = in_list_arr_row_col[i]


    #     # arr_row_col_tmp = arr_row_col_tmp.astype(int)
        arr_polygon_id_tmp = np.full((arr_row_col_tmp.shape[0]), id_tmp)
        if i == 0:
            arr_polygon_id = arr_polygon_id_tmp.copy()
            arr_row_col = arr_row_col_tmp.copy()
            i = 1
        else:
            arr_polygon_id = np.concatenate((arr_polygon_id, arr_polygon_id_tmp)).copy()
            arr_row_col = np.concatenate((arr_row_col, arr_row_col_tmp)).copy()
            i = i + 1

            
            
    return arr_polygon_id, arr_row_col[:,0], arr_row_col[:,1]





def get_df_row_col(in_s_polygon, in_raster_path, **kwargs):
    '''
    To extract pixel values of a given polygon (wkt or shapely geometry).

    Parameters
    ----------
    in_s_polygon: a pandas series 
        a series of polygons to be extracted

    in_raster_path: str of raster path
        a path of the raster to get row&col        
        
    Returns
    -------
    A dataframe of pixel with row&col (index of series will be shown as another column).

    '''

    #Create row col mappint raster    
    path_ds_mapping = create_row_col_mapping_raster(in_raster_path, out_mem=False)
    

    #getting row-col from on-memory row-col raster
    list_arr_row_col = []    
    list_polygon_id = []    
    
    for polygon_id, tmp_polygon in tqdm(in_s_polygon.iteritems(), 'Getting row&col of pixels...', total=len(in_s_polygon)):
        is_polygon_overlap, nbr_pixels, arr_row_col = get_pix_row_col(tmp_polygon, path_ds_mapping)
        if is_polygon_overlap:
            list_polygon_id.append(polygon_id)
            list_arr_row_col.append(arr_row_col)
        
    shutil.rmtree(os.path.dirname(path_ds_mapping))
    arr_polygon_id, arr_row, arr_col = __reformat_list_row_col_for_df(list_polygon_id, list_arr_row_col)
    list_data = zip(arr_polygon_id, arr_row, arr_col)
    out_df_polygon_row_col = pd.DataFrame(list_data, columns=[in_s_polygon.index.name, 'row', 'col'])
    out_df_polygon_row_col[['row', 'col']] = out_df_polygon_row_col[['row', 'col']].astype(int)

    return out_df_polygon_row_col





def extract_pixval_single_file(in_s_polygon, in_raster_path, in_list_out_col_nm, in_list_target_raster_band_id, nodata_val=-999, **kwargs):
    '''
    To extract pixel values of a given polygon (wkt or shapely geometry) from single raster file with one or more bands.

    Parameters
    ----------
    in_s_polygon: a pandas series 
        a series of polygons to be extracted

    in_raster_path: str
        a path of the raster to be extracted
        
    in_list_out_col_nm: list 
        a list of column names of output dataframe representing each raster
    
    in_list_target_raster_band_id: int
        indicate band id of raster to extract

    nodata_val: number
        a value of nodata value of input raster which will be replaced with np.nan
        
    Returns
    -------
    A dataframe of pixel values with row-col.

    '''
    #check no. of output columns equals to no. of raster files
    assert(len(in_list_out_col_nm) == len(in_list_target_raster_band_id))

    #Convert input lists into list to prevent unexpected behavior if input lists are not really list.
    in_list_out_col_nm = list(in_list_out_col_nm)
    in_list_target_raster_band_id = list(in_list_target_raster_band_id)    

    df_polygon_row_col_pixval = get_df_row_col(in_s_polygon, in_raster_path)
    
    for i in tqdm(range(len(in_list_out_col_nm)), 'Getting pixel values...'):
        col_nm = in_list_out_col_nm[i]        
        band_id = in_list_target_raster_band_id[i]
        with rasterio.open(in_raster_path) as ds:
            arr_raster = ds.read(band_id)

        arr_pixval_1d = extract_values_from_2d_array_with_row_col_numba(arr_raster, df_polygon_row_col_pixval[['row', 'col']].values)
        df_polygon_row_col_pixval[col_nm] = np.where(arr_pixval_1d==nodata_val, np.nan, arr_pixval_1d)

    return df_polygon_row_col_pixval


def extract_pixval_multi_files(in_s_polygon, in_list_raster_path, in_list_out_col_nm, in_target_raster_band_id=1, nodata_val=-999, check_raster_consistent=True, **kwargs) :
    '''
    To extract pixel values of a given polygon (wkt or shapely geometry) from multiple raster files with the same geo reference.

    Parameters
    ----------
    in_s_polygon: a pandas series 
        a series of polygons to be extracted

    in_list_raster_path: list
        a list of paths of the raster to be extracted
        
    in_list_out_col_nm: list 
        a list of column names of output dataframe representing each raster
    
    in_target_raster_band_id: int
        indicate band id of raster to extract

    nodata_val: number
        a value of nodata value of input raster which will be replaced with np.nan
        
    Returns
    -------
    A dataframe of pixel values with row-col.

    '''
    #check no. of output columns equals to no. of raster files
    assert(len(in_list_out_col_nm) == len(in_list_raster_path))

    #Convert input lists into list to prevent unexpected behavior if input lists are not really list.
    in_list_raster_path = list(in_list_raster_path)
    in_list_out_col_nm = list(in_list_out_col_nm)    


    #check all of given raster paths are having the same geo reference & transform
    tmp_raster_path = in_list_raster_path[0]   
    if check_raster_consistent:
        with rasterio.open(tmp_raster_path) as ds_tmp:
            tmp_transform = ds_tmp.transform
            tmp_crs = ds_tmp.crs    
        for tmp_raster_path in in_list_raster_path:
            with rasterio.open(tmp_raster_path) as ds_tmp:
                assert(tmp_transform == ds_tmp.transform)
                assert(tmp_crs == ds_tmp.crs)

    

    df_polygon_row_col_pixval = get_df_row_col(in_s_polygon, tmp_raster_path)
    
    for i in tqdm(range(len(in_list_out_col_nm)), 'Getting pixel values...'):
        col_nm = in_list_out_col_nm[i]
        raster_path = in_list_raster_path[i]
        with rasterio.open(raster_path) as ds:
            arr_raster = ds.read(in_target_raster_band_id)

        arr_pixval_1d = extract_values_from_2d_array_with_row_col_numba(arr_raster, df_polygon_row_col_pixval[['row', 'col']].values)
        df_polygon_row_col_pixval[col_nm] = np.where(arr_pixval_1d==nodata_val, np.nan, arr_pixval_1d)

    return df_polygon_row_col_pixval


def get_arr_rowcol_mapping_from_raster(raster_path):
    '''
    Get array of pixel row & col mapping from a given raster.
    Return an array of 6 columns for row, col, lat_upper, lat_lower, lon_lower, lon_upper accordingly.
    '''
    assert(os.path.exists(raster_path))        
    path_ds_mapping = create_row_col_mapping_raster(raster_path, out_mem=False)
    with rasterio.env():
        with rasterio.open(path_ds_mapping) as ds_mapping:

            (lon_col_0, lon_pix_size, _, lat_row_0, _, lat_pix_size) = ds_mapping.get_transform()
            arr_mapping = ds_mapping.read()
            ds_mapping = None

            arr_mapping = arr_mapping.T.reshape(-1, 2)
            arr_lat_upper = lat_row_0 + (arr_mapping[:, 0] * lat_pix_size)
            arr_lat_lower = lat_row_0 + ((arr_mapping[:, 0] + 1) * lat_pix_size)    
            arr_lon_upper = lon_col_0 + ((arr_mapping[:, 1] + 1) * lon_pix_size)
            arr_lon_lower = lon_col_0 + (arr_mapping[:, 1] * lon_pix_size)
            arr_mapping = np.concatenate(
                (
                    arr_mapping, 
                    arr_lat_upper.reshape(-1, 1),
                    arr_lat_lower.reshape(-1, 1),             
                    arr_lon_upper.reshape(-1, 1),
                    arr_lon_lower.reshape(-1, 1), 
                ),
                axis=1)
    shutil.rmtree(os.path.dirname(path_ds_mapping))
    return arr_mapping

 

@jit(nopython=True)
def __get_pixel_rowcol_from_latlon_numba(lat, lon, arr_mapping):
    arr_pixel_row_col = arr_mapping[((arr_mapping[:, 2] >= lat ) & (arr_mapping[:, 3] < lat) & (arr_mapping[:, 4] >= lon ) & (arr_mapping[:, 5] < lon))]
    if(len(arr_pixel_row_col)==1):
        row, col = arr_pixel_row_col[0, 0:2]
        row = int(row)
        col = int(col)
    return row, col

 


def get_pixel_rowcol_from_latlon(lat, lon, df_mapping=None, raster_path=None, arr_mapping=None, check_format=False):
    '''
    Get pixel row & col idx of a lat lon from a mapping dataframe or a raster or an arr
    '''
    assert((type(df_mapping)==pd.DataFrame) or (type(raster_path)==str) or (type(arr_mapping) == np.ndarray))
    if type(df_mapping)==pd.DataFrame:
        assert(type(raster_path)==type(None))
        assert(type(arr_mapping)==type(None))      
        
        if check_format==True:
            assert(set(df_mapping.columns.to_list()) == set(['row', 'col', 'lat_upper', 'lat_lower', 'lon_lower', 'lon_upper']))        
            
        df_pixel = df_mapping[(df_mapping['lat_upper'] >= lat) & (df_mapping['lat_lower'] < lat) & (df_mapping['lon_upper'] >= lon) & (df_mapping['lon_lower'] < lon)]
        if len(df_pixel)==1:
            row = int(df_pixel.iloc[0]['row'])
            col = int(df_pixel.iloc[0]['col'])
        else:
            row = -999
            col = -999
        return row, col
    elif type(raster_path)==str:
        
        arr_mapping = get_arr_rowcol_mapping_from_raster(raster_path)
        return get_pixel_rowcol_from_latlon(lat, lon, arr_mapping=arr_mapping)
    
    elif type(arr_mapping) == np.ndarray:
        if check_format==True:
            assert(arr_mapping[:, 2:4].min() >= -90) #range of lat
            assert(arr_mapping[:, 2:4].max() <= 90) #range of lat
            assert(arr_mapping[:, 4:6].min() >= -180) #range of lon
            assert(arr_mapping[:, 4:6].max() <= 180) #range of lon

 

#         arr_pixel_row_col = arr_mapping[((arr_mapping[:, 2] >= lat ) & (arr_mapping[:, 3] < lat) & (arr_mapping[:, 4] >= lon ) & (arr_mapping[:, 5] < lon))]
#         if(len(arr_pixel_row_col)==1):
#             row, col = arr_pixel_row_col[0, 0:2]
#             row = int(row)
#             col = int(col)
#         return row, col
        return __get_pixel_rowcol_from_latlon_numba(lat, lon, arr_mapping)