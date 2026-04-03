import numpy as np
import pandas as pd
import rasterio

# The function "inspect_tif" inspects the raster data files to get their characteristics

def inspect_tif(filename):

    with rasterio.open(filename) as src:
        print("Number of bands :", src.count)
        print("Descriptions     :", src.descriptions)  # names of the bands if they exist
        print("Data types :", src.dtypes)
        print("NoData           :", src.nodata)
        print("CRS              :", src.crs)
        print("Resolution       :", src.res)
        print("Dimensions       :", src.width, "x", src.height)


# The function "remove_nodata" takes a band and the value that corresponds to a missing value in the raster data files and returns a binary filter

def remove_nodata(data, nodata_value=None):
    
    if nodata_value is not None:
        mask = (data != nodata_value) & np.isfinite(data)
    else:
        mask = np.isfinite(data)  # eliminates NaN et inf

    print(f"Total pixels  : {data.size}")
    print(f"Valid pixels  : {mask.sum()}")
    print(f"Pixels deleted: {data.size - mask.sum()}")

    return mask



# The following function "raster_to_dataframe" takes a band and transforms it into a dataframe. 
# It applies the mask defined in the previous function to get rid of missing values

def raster_to_dataframe(data, transform, mask=None):
   
    height, width = data.shape
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())

    if mask is not None:
        m = mask.ravel()
        xs = np.array(xs)[m]
        ys = np.array(ys)[m]
        data_vals = data.ravel()[m]
    else:
        xs = np.array(xs)
        ys = np.array(ys)
        data_vals = data.ravel()

    df = pd.DataFrame({
        'x': xs,
        'y': ys,
        'band_1': data_vals
    })

    return df



# The function "get_reference_file" goes through all the raster data files to check their resolution and returns the file with the lowest resolution/biggest pixel size

def get_reference_file(filenames):
    
    max_res = 0
    reference = None
    
    for filename in filenames:
        with rasterio.open(filename) as src:
            res = src.res[0]
            print(f"{filename} : resolution = {res}")
            if res > max_res:
                max_res = res
                reference = filename
    
    print(f"\nReference file : {reference}")
    print(f"Lowest resolution : {max_res}")
    
    return reference, max_res



# The function "get_reference_file" goes through all the raster data files to check their resolution and returns only the lowest resolution/biggest pixel size


def get_lowest_resolution(filenames):
    
    max_res = 0
    
    for filename in filenames:
        with rasterio.open(filename) as src:
            res = src.res[0]  
            print(f"{filename} : resolution = {res}")
            if res > max_res:
                max_res = res
    
    print(f"\nLowest resolution : {max_res}")
    return max_res




# The function "resample_tif" scales all files down to the lowest resolution obtained by the previous function

def resample_tif(filename, target_res):
    
    with rasterio.open(filename) as src:
        
        # Scaling factor
        scale = src.res[0] / target_res
        
        # New dimensions
        new_height = int(src.height * scale)
        new_width = int(src.width * scale)
        
        # Read and resample
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Upload the transform file
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height
        )
        
    return data, new_transform



# The function "align_rasters" aligns all files on the grid of the reference file

from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

def align_rasters(filenames, reference_file):
    
    aligned_data = {}
    
    # Read the properties of the reference file
    with rasterio.open(reference_file) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_nodata = ref.nodata
    
    # Align each file on the reference
    for filename in filenames:
        with rasterio.open(filename) as src:
            
            # Destination array with the dimension of the reference file
            destination = np.zeros((src.count, ref_height, ref_width), dtype=src.dtypes[0])
            
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
        
        name = filename.split("/")[-1].replace(".tif", "")
        aligned_data[name] = destination
        print(f"✅ {name} aligned — shape: {destination.shape}")
    
    return aligned_data




# "Merge_dataframes" combines all dataframes into one on the coordinates (x,y)

def merge_dataframes(dataframes):
    
    df_merged = None
    
    for nom, df in dataframes.items():
        
        # Rename the band column with the name of the file 
        df = df.rename(columns={'band_1': nom})
        
        if df_merged is None:
            df_merged = df  
        else:
            # Join on the (x,y) coordinates
            df_merged = df_merged.merge(df, on=['x', 'y'], how='inner')
        
        print(f"✅ {nom} merged — shape: {df_merged.shape}")
    
    print(f"\nFinal DataFrame : {df_merged.shape}")
    print(df_merged.head())
    
    return df_merged