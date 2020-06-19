#Authored by : Pongporamat C. 
#Updated by Surasak C.

from pandas import DataFrame
import numpy as np
import pandas as pd
import datetime
import os
import subprocess
from tqdm import tqdm
import tarfile
import utm
from osgeo import ogr, gdal, gdal_array, gdalconst
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon, mapping
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask

import skimage
from skimage import filters, exposure
from skimage.io import imsave







np.seterr(divide='ignore', invalid='ignore')

def raster_reproject(src_path, dest_path, reference_path):

    inputfile = src_path
    input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
    inputProj = input.GetProjection()
    inputTrans = input.GetGeoTransform()

    referencefile = reference_path
    reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
    referenceProj = reference.GetProjection()
    referenceTrans = reference.GetGeoTransform()
    bandreference = reference.GetRasterBand(1)    
    x = reference.RasterXSize 
    y = reference.RasterYSize


    outputfile = dest_path
    driver= gdal.GetDriverByName('GTiff')
    output = driver.Create(outputfile,x,y,1,bandreference.DataType)
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)

    gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Bilinear)

    del output





def df_to_gdf(df, geom_col_nm):
    '''
    INPUT 
    df : dataframe with a column containing geometry in wkt format
    geom_col_nm : str specifies column name of geometry in df
    
    OUTPUT
    return :GeoPandas's DataFrame (gdf)
    '''
    
    df['geometry'] = df[geom_col_nm].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs={'init' : 'epsg:4326'})
    return gdf



def shape_to_raster(in_shp, out_tif, ref_tif):
    #Code by PongporC 23/Jan/2020
    '''
    params
    ----------------------------------
    in_shp : path of input shapefile
    out_tif : path of output tif
    ref_tif : path of reference tif 
    '''


    InputVector = in_shp
    OutputImage = out_tif

    RefImage = ref_tif

    gdalformat = 'GTiff'
    datatype = gdal.GDT_Byte
    burnVal = 1 #value for the output image pixels
    ##########################################################
    # Get projection info from reference image
    Image = gdal.Open(RefImage, gdal.GA_ReadOnly)

    # Open Shapefile
    Shapefile = ogr.Open(InputVector)
    Shapefile_layer = Shapefile.GetLayer()

    # Rasterise
    print("Rasterising shapefile...")
    Output = gdal.GetDriverByName(gdalformat).Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 1, datatype, options=['COMPRESS=DEFLATE'])
    Output.SetProjection(Image.GetProjectionRef())
    Output.SetGeoTransform(Image.GetGeoTransform()) 

    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(0)
    gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])

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


    
def shapefile_intersection_shpshp(gdf, gdf_ref, txt_print = None):
    #Code by PongporC 23/Jan/2020
    ''' Intersection between two shapefiles 

    Parameters
    ----------------------------------
    shapefile1 : shape file 1 (GeoDataFrame)
    shapefile2 : shape file 2 (GeoDataFrame)

    Return
    ----------------------------------
    GeoPandas.GeoDataFrame

    '''
    if txt_print != None:
        print('Start : {}'.format(txt_print))

    data = []
    for index1, poly1 in tqdm(gdf.iterrows(), total=len(shapefile1)):
        for index2, poly2 in gdf_ref.iterrows():
           if poly1['geometry'].intersects(poly2['geometry']):
              data.append(poly1)
    gdf = gpd.GeoDataFrame(data)
    if txt_print != None:
        print('Finish : {}'.format(txt_print))
        print()

    return gdf



def shapefile_intersection_tifshp(in_shp, in_rst, all_touched=True):
    #Code by PongporC 23/Jan/2020
    ''' Intersection between raster and shapefile
        "raster" is the directory of rice gistda raster
        "polygon" is the shapefile of polygon
    '''
    # raster is rice gistda, polygon is plot polygon
    # Intersection
    geoms = in_shp.geometry.values
    geoms = [mapping(geoms[0])]
    
    with rasterio.open(in_rst) as rst:
        try:
            intersect_image, intersect_transform = mask(rst, geoms, crop=True, all_touched=True)
        except:
            return None, None, True
    # rasterize polygon
    raster_shape = intersect_image.shape[1], intersect_image.shape[2] 
    rasterize_polygon = rasterize(
            [(shape, 1) for shape in in_shp['geometry']],
            out_shape = raster_shape,
            transform = intersect_transform,
            fill = 0,
            all_touched = all_touched,
            dtype = rasterio.uint8
            )
    return intersect_image, rasterize_polygon, False
    



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


def stack_raster_to_tif(raster_filepath_list, output_tif):
    '''
    raster_filepath_list : list of raster file paths to be stacked
    output_tif :
    '''
    raster_list = []
    ref_raster = None
    
    print('Loading raster files.')
    for filepath in tqdm(raster_filepath_list):
        raster_list.append(gdal.Open(filepath).ReadAsArray())

        #Define georeference raster as first file
        if ref_raster == None:
            ref_raster = gdal.Open(filepath)

    print('Stacking files.')
    stacked = np.array(raster_list)
    gdal_array.SaveArray(stacked.astype("float"), output_tif, "GTiff", ref_raster)



def ls8_ndvi(raster_nir_path, raster_red_path):
    #Code by PongporC 23/Jan/2020

    #%%
    raster_nir = rasterio.open(raster_nir_path)
    raster_r = rasterio.open(raster_red_path)
    #%%
    raster_nir_im = raster_nir.read().astype(np.float64)
    raster_r_im = raster_r.read().astype(np.float64)
    #%%
    raster_ndvi = np.where((raster_nir_im+raster_r_im)==0, 0, (raster_nir_im-raster_r_im)/(raster_nir_im+raster_r_im))
    #%%

    return raster_ndvi


def show_raster(raster, min_pctl=2, max_pctl=98):
    #Code by PongporC 23/Jan/2020

    vmin, vmax = np.nanpercentile(raster, (min_pctl, max_pctl))    
    plt.imshow(raster_ndvi[0], vmin=vmin, vmax=vmax)
    






def split_polygon(multi_polygon):
    #Code by PongporC 23/Jan/2020
    ''' split the polygon string in excel cell into a list of lat and long
        "multi_polygon" is the string of latlon 
    '''
    polygon_list = multi_polygon.split("(")[3].split(")")[0].split(",")
    return polygon_list

def multi_lat_long_QGIS(multi_polygon):
    #Code by PongporC 23/Jan/2020
    ''' convert the polygon string to csv file where the frist column is "latitude" and the second is 'longitude'
        "multi_polygon" is the string of latlon 
    '''
    polygon_list = split_polygon(multi_polygon) 
    lat_long = dict()
    lat_long['lat'] = list()
    lat_long['long'] = list() 
    for i in range(len(polygon_list)):
        lat_long['lat'].append(polygon_list[i].split(' ')[1])
        lat_long['long'].append(polygon_list[i].split(' ')[0])  
    df = DataFrame(lat_long, columns=['lat', 'long'])
    df.to_csv(r"C:\Users\PongporC\Desktop\Multi-lat_long_QGIS.csv", index = None, header = False)    
    print(r"C:\Users\PongporC\Desktop\Multi-lat_long_QGIS.csv -- Write file done!!")

def extract_tar_file(landsat_tar_dir):
    ## Extract landsat .tar.gz file (choose which date from files in folder "area_date_name")
    list_dir = os.listdir(landsat_tar_dir)
    landsat_tar_name = list()
    for name in list_dir:
        if name.endswith('.gz'):
            landsat_tar_name.append(name)
    [print("%2d:"%(idx+1), name) for idx, name in enumerate(landsat_tar_name)]
    index = input("Enter file idx: ")    
    fname = landsat_tar_name[int(index)-1]
    if os.path.isdir(os.path.join(landsat_tar_dir, fname).split(".")[0]):
        print("Already extract")
    else:
        tar = tarfile.open(os.path.join(landsat_tar_dir, fname))
        tar.extractall(os.path.join(landsat_tar_dir, fname.split('.')[0]))
        tar.close()
    return os.path.join(landsat_tar_dir, fname.split(".")[0])
    
def extract_rice_csv(csv_file_dir, header_names):
    #Code by PongporC 23/Jan/2020
    ''' Return require header from csv file and convert year, month, day into datetime.date
        "csv_file_dir" is the directory of csv_file
        "header_names" is the required columns 
    '''
    csv_file = pd.read_parquet(csv_file_dir)[header_names]
    csv_file['plant_date'] = pd.to_datetime(dict(year = csv_file['final_plant_year'], month = csv_file['final_plant_month'], day = csv_file['final_plant_day'])).dt.date
    csv_file['harvest_date'] = pd.to_datetime(dict(year = csv_file['final_harvest_year'], month = csv_file['final_harvest_month'], day = csv_file['final_harvest_day'])).dt.date

    activity_list = csv_file['ACTIVITY_ID']
    multi_polygon_list = csv_file['final_polygon']
    province_code_list = csv_file['PLANT_PROVINCE_CODE']
    plant_date_list = csv_file['plant_date']
    harvest_date_list  = csv_file['harvest_date']

    return [activity_list, multi_polygon_list, province_code_list, plant_date_list , harvest_date_list]
    
        
def find_shape_in_range(start_date, stop_date, rice_shape_list):
    #Code by PongporC 23/Jan/2020
    ''' Return the list of shape files that in the range of start and stop date
        "start_date" is the start datetime
        "stop_date" is the stop datetime
        "rice_shape_list" is the list name of all shapefiles
    '''
    shape_in_range = list()
    for shape_name in rice_shape_list:
        shape_year  = int(shape_name.split('_')[1][0:4])
        shape_month = int(shape_name.split('_')[1][4:6])
        shape_day   = int(shape_name.split('_')[1][6:8])
        shape_date  = datetime.date(shape_year, shape_month, shape_day)
        if start_date <= shape_date <= stop_date:
            shape_in_range.append(shape_name)
    return shape_in_range

def polygon_to_shapefile(multi_polygon, to_crs = {'init' : 'epsg:4326'}, save = False, output = ""):
    #Code by PongporC 23/Jan/2020
    ''' Change lat, lon polygon into geopandas shapefile
        "multi_polygon" is the string of latlon 
        "to_crs" is the projection type of the output shapefile
        "save" : save or not 
        "output" : output name of saved file 
    '''
    multi_polygon_split = split_polygon(multi_polygon)
    lat = []
    lon = []
    for coord in multi_polygon_split:
        lat.append(float(coord.split(" ")[1]))
        lon.append(float(coord.split(" ")[0]))
    polygon_geom = Polygon(zip(lon, lat))
    crs = {'init' : 'epsg:4326'}
    polygon = gpd.GeoDataFrame(index = [0], crs=crs, geometry=[polygon_geom])
    if crs != to_crs:
        polygon = polygon.to_crs(to_crs)
    if save:
        polygon.to_file(filename = output, drive ='ESRI Shapefile')  
    return polygon


    
def polygon_size_vs_crop_regist(rice_details_dir, output_dir):
    #Code by PongporC 23/Jan/2020
    ''' Collect polygon size and plot size 
        "rice_details_dir" is the directory of csv file
        "output_dir" is the directory of output file 
    '''
    columns_req = ['final_polygon', 'ACTIVITY_ID', 'ACT_RAI_ORI', 'ACT_NGAN_ORI', 'ACT_WA_ORI']
    rice_details = extract_rice_csv(rice_details_dir = rice_details_dir, shape_file_dir=None, columns_req = columns_req)
    rai_list  = rice_details['ACT_RAI_ORI']
    ngan_list = rice_details['ACT_NGAN_ORI']
    wa_list   = rice_details['ACT_WA_ORI']
    activity_list = rice_details['ACTIVITY_ID']
    multi_polygon_list    = rice_details['final_polygon']
    
    validation_report = dict()
    validation_report['ACTIVITY_ID'] = list()
    validation_report['Polygon_area'] = list()
    validation_report['Crop_regist_area'] = list()
    for idx in range(len(rai_list)):
        multi_polygon = multi_polygon_list[idx]
        rai  = rai_list[idx]
        ngan = ngan_list[idx]
        wa   = wa_list[idx]
        activity_id = activity_list[idx]
        # all in square meters
        polygon_area = calculate_polygon_area_from_lat_long(multi_polygon)
        crop_regist_area = (400*rai + 100*ngan + wa) * 4
        
        validation_report['ACTIVITY_ID'].append(activity_id)
        validation_report['Polygon_area'].append(polygon_area)
        validation_report['Crop_regist_area'].append(crop_regist_area)
        
    df = DataFrame(validation_report, columns=['ACTIVITY_ID', 'Polygon_area', 'Crop_regist_area'])
    df.to_csv(output_dir, index = None, header = True)    

def append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage, detail = ''):
    #Code by PongporC 23/Jan/2020
    ''' Append the details into the report
        Need to revise everytime you want to add or remove dictionary
    '''
    report['ACTIVITY_ID'].append(activity_id)
    report['Polygon'].append(multi_polygon)
    report['P_code'].append(P_code)
    report['Plant date'].append(start_date)
    report['Shapefile date'].append(shape_date)
    report['Harvest date'].append(stop_date)
    report['Intersection percentage'].append(intersection_percentage)
    report['Detail'].append(detail)
    return report

def intersection_stat_vector_approach(split_shapefile_dir, rice_shape_list, activity_id, P_code, start_date, stop_date, multi_polygon, report):
    #Code by PongporC 23/Jan/2020
    ''' In this function polygon's and rice shapefile's crs are already reprojection to {epsg:4326}
        "split_shapefile_dir" is the directory of splited shapefile based on P_code
        "rice_shape_list" is the list of directories of rice gistda shapefile
        "others" are the required details for append to the report
    '''
    shape_in_range = find_shape_in_range(start_date, stop_date, rice_shape_list)
    polygon = polygon_to_shapefile(multi_polygon)#, to_crs = {'init' : 'epsg:32647'})
    if not polygon.is_valid[0]:
        shape_date = np.nan
        report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "Polygon is not valid")
        return report
    
    poly_area = polygon.area
    for rice_idx, rice_shape_name in enumerate(shape_in_range):
        shape_date = rice_shape_name.split('_')[1]
        shape_date = shape_date[4:6] + "/" + shape_date[6:8] + "/" + shape_date[0:4]  
        if not os.path.isfile(os.path.join(split_shapefile_dir, rice_shape_name, str(P_code), str(P_code) + '.shp')):
            report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "Cannot find shapefile in province_code: %02d"%(P_code))
            continue
        
        rice_shape = gpd.GeoDataFrame.from_file(os.path.join(split_shapefile_dir, rice_shape_name, str(P_code), str(P_code) + '.shp'))
        try:
            df = shapefile_intersection_shpshp(rice_shape, polygon)  
        except:
            report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "TopologyException: Input geom 1 is invalid: Ring Self-intersection")
            continue
        
        if (len(df) == 0):
            report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "Cannot intersect")

        elif (len(df) != 0):
            intersection_area = df.area.sum()
            intersection_percentage = 100 * (intersection_area/poly_area)[0]  
            report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage)

    return report

def intersection_stat_raster_approach(split_shapefile_dir, rice_shape_list, activity_id, P_code, start_date, stop_date, multi_polygon, report):
    #Code by PongporC 23/Jan/2020
    ''' In this function polygon's crs are created in {epsg:32647} and rice raster's crs are already {epsg:32647}
        The rice shapefile are rasterized to 30 meters resolution with all_touch
        "split_shapefile_dir" is the directory of splited shapefile based on P_code
        "rice_shape_list" is the list of directories of rice gistda shapefile
        "others" are the required details for append to the report        
    '''
    # Find the name list of shapefile that in the range of plant and harvest date
    shape_in_range = find_shape_in_range(start_date, stop_date, rice_shape_list)
    try:
        ## In case of something goes wroung while creating polygon
        polygon = polygon_to_shapefile(multi_polygon, to_crs = {'init' : 'epsg:32647'})
    except:
        ## Append the data into report where shape_date is Nan and Detail is "Polygon is not valid"
        shape_date = np.nan
        report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "Polygon is not valid")
        return report
    
    if not polygon.is_valid[0]:
        ## If the polygon is not valid, append the data into report where shape_date is Nan and the detail is "Polygon is not valid"
        shape_date = np.nan
        report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "Polygon is not valid")
        return report
    
    # Loop for each rice shapefile that in the range of plant and harvest date
    for rice_idx, rice_shape_name in enumerate(shape_in_range):
        shape_date = rice_shape_name.split('_')[1]
        shape_date = datetime.date(int(shape_date[0:4]), int(shape_date[4:6]), int(shape_date[6:8]))
        if not os.path.isfile(os.path.join(split_shapefile_dir, rice_shape_name, str(P_code), str(P_code) + '.tif')):
            # If there is no raster of shapefile in that province(P_code)
            report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "Cannot find shapefile in province_code: %02d"%(P_code))
            continue
        # Directory of rice raster
        rice_shape = os.path.join(split_shapefile_dir, rice_shape_name, str(P_code), str(P_code) + '.tif')
        
        # Intersection between rice raster and plot polygon
        intersect_image, polygon_image, out_of_bound = shapefile_intersection_tifshp(rice_shape, polygon)   
        if out_of_bound:
            # If polygon is not in the raster, append the data into report where the detail is "Cannot intersect (out of bound)"
            report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage = 0, detail = "Cannot intersect (out of bound)")
            
        else:
            # Successfully intersection
            intersection_percentage = 100*intersect_image.sum() /polygon_image.sum()          
            if intersect_image.sum() > 0:
                # If can intesect
                report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage)

            else:
                # If cannot intersect
                report = append_data(report, activity_id, multi_polygon, P_code, start_date, shape_date, stop_date, intersection_percentage, detail = "Cannot intersect")

    return report
##################################################################################

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    