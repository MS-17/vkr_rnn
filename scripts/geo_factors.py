from pprint import pprint

import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from calendar import monthrange
import os, json
from rasterstats import zonal_stats
import rasterio


def main(year_):
	data_year = year_

	base_p = "/home/sergei/Downloads/vkr/trying_rnn/code"

	project_files = base_p + "/project_data/"

	fires_path = project_files + "dataset_plus_grid/"
	fires_nonfires = f"dataset_{str(data_year)}.geojson"
	meteo_data = f"irkutsk_{str(data_year)}_hourly.json"
	meteo_data_path = project_files + "meteo_data/data/"
	# aggregation period 1 day
	agg_period = 1
	project_crs = "epsg:3857"
	print(
		"Paths:", project_files, fires_path, meteo_data_path, fires_nonfires, meteo_data,
		"Aggregation period:", agg_period, "Project crs:", project_crs, sep="\n"
	)

	#########################################################################################################
	# %% [markdown]
	# #### Connect meteo factors

	# export fires_nonfires dataset
	fires_nonfires_ds = gpd.read_file(fires_path + fires_nonfires)
	if not fires_nonfires_ds.crs == project_crs:
		print(f"Converting \"{fires_nonfires_ds.name}\" to {project_crs}")
		fires_nonfires_ds = fires_nonfires_ds.to_crs(project_crs)
	print(fires_nonfires_ds, fires_nonfires_ds.columns, fires_nonfires_ds.dtypes, fires_nonfires_ds.event_date, sep="\n")

	# load meteo data json for the given
	meteo_data_p = meteo_data_path + meteo_data
	print("Path:", meteo_data_p)
	with open(meteo_data_p, mode='r', encoding="Windows-1251") as f:
		meteo_data = json.load(f)
	print(type(meteo_data))

	# json data exploration
	for k, v in meteo_data.items():
		print("Keys:", k, type(k), "Values:", type(meteo_data[k]))

	print(
		"Meta length:", len(meteo_data["METADATA"]),
		"Meta:", json.dumps(meteo_data["METADATA"], indent=4, ensure_ascii=False),
		"Data length:", len(meteo_data["DATA"]),
		"Data[0]:", json.dumps(meteo_data["DATA"][0], indent=4), sep="\n",
	)

	# print the data fields russian description
	metadata_desc_rus = {}
	for k, v in meteo_data["METADATA"].items():
		metadata_desc_rus[k] = v["DESC_RUS"]
	print("Data fields russian description:", json.dumps(metadata_desc_rus, indent=4, ensure_ascii=False))

	# data structure
	# METADATA: {"factor1": {}, "factor2": {}, ...}, DATA: [{factor1: val1, factor2: val2, ...}, {factor1: val, ...}, ...]

	# ensure that all metadata keys used in data keys
	# only id is an extra key not stated in metadata
	d = {}
	for k in meteo_data["DATA"][0].keys():
		d[k] = k in meteo_data["METADATA"].keys()
	print(d)

	# convert the meteo json to the meteo dataset
	meteo_ds = pd.DataFrame(meteo_data["DATA"])
	# pd.options.display.max_columns = None
	print(meteo_ds, meteo_ds.dtypes, sep="\n")

	# convert meteo data date column to datetime
	meteo_ds["DATE"] = pd.to_datetime(meteo_ds["DATE"])
	print(meteo_ds.dtypes, meteo_ds.shape, meteo_ds.DATE)

	# small corrections
	meteo_ds["LAT"] = meteo_ds["LAT"].astype(float)
	meteo_ds["LON"] = meteo_ds["LON"].astype(float)
	print(meteo_ds["LAT"], meteo_ds["LON"])

	# convert meteo data to the gmt+8
	meteo_date = meteo_ds.DATE
	meteo_time = meteo_ds.TIME
	meteo_date_time = pd.to_datetime(meteo_date.astype(str) + " " + meteo_time)
	meteo_ds["date_time"] = meteo_date_time + pd.DateOffset(hours=8)
	# print(meteo_ds.head(5), meteo_ds.date_time.dt.date)

	def mode_num(x: pd.Series) -> float:
		"""
		Description: counts mode for the series and gets its first value
		Params:
		x - the series
		Returns: one of the series modes or numpy nan if the mode was not found
		"""
		m = x.mode()
		if len(m) == 0:
			return np.nan
		return float(m[0])

	ex_cols = ["ID", "DATE", "TIME", "LAT", "LON", "date_time"]
	target_columns = meteo_ds.columns[~meteo_ds.columns.isin(ex_cols)]

	# Extract meteo factors
	# T, RH, WIND_DIR, WIND_SPEED, APCP, soilw, tmpgr (agg functions: mean, max, min
	# std, mode, median)
	# !!! Attention: nan and None values are skipped
	# aggregate for the aggregation period = 7
	factors = {}
	for i in fires_nonfires_ds.index:
		lat = fires_nonfires_ds.iloc[i].grid_lat
		lon = fires_nonfires_ds.iloc[i].grid_lon
		date = fires_nonfires_ds.iloc[i].event_date
		start = date - pd.DateOffset(days=agg_period)
		# include the target day as well, totally there're actually 8 days for the aggregation (the last one included)
		end = date + pd.DateOffset(days=1)		
		# print("Start date:", start, "End date:", end, "Grid lat:", lat, "Grid lon:", lon)
		date_mask = (meteo_ds["date_time"] >= start) & (meteo_ds["date_time"] <= end)
		meteo_date_mask = meteo_ds[date_mask]
		# print(meteo_date_mask)
		# consider the floating point error
		coord_mask = ((meteo_date_mask["LAT"] - lat).abs() <= 1.0e-5) & ((meteo_date_mask["LON"] - lon).abs() <= 1.0e-5)
		meteo_factors = meteo_date_mask[coord_mask]
		
		factors[i] = {}
		# should be competable with the pd.df.aggregate method
		agg_functions = ["mean", "std"] # "max", "min", "std", mode_num, "median",]
		for col in target_columns:
			agg_d = meteo_factors[col].aggregate(agg_functions).to_dict()
			# factors[i][col] = {f"{col.lower()}_{k}" : v for k, v in agg_d.items()}
			agg_d_ = {f"{col.lower()}_{k}" : v for k, v in agg_d.items()}
			factors[i].update(agg_d_)

	assert len(factors) == fires_nonfires_ds.shape[0]
	k_ = json.dumps({k : factors[k] for k in list(factors.keys())[:1]}, indent=4)
	print(k_)

	# form a meteo factors dataset
	meteo_factors_ds = pd.DataFrame.from_dict(factors, orient="index")
	print(meteo_factors_ds.dtypes, meteo_factors_ds.shape, meteo_factors_ds.head(5), sep="\n")

	# round to 6 decimal places
	prec = 6
	meteo_factors_ds = meteo_factors_ds.apply(lambda x: np.round(x, decimals=6))
	factors_ds_nan = fires_nonfires_ds.join(meteo_factors_ds)

	# replace nans by the parameter mode calculated for the event month
	nan_idx = factors_ds_nan[factors_ds_nan.isnull().any(axis=1)].index
	# for each row that has at least one nan value, extract the column of that value
	# get a month of the row, get all values for the month and get mode, write to nan
	factors_ds = factors_ds_nan.copy()
	for idx in nan_idx:
		row = factors_ds.loc[idx]
		cols = row[row.isnull()].index.to_list()
		for col in cols:
			row_month = row.event_date.month
			m_ = factors_ds[factors_ds.event_date.dt.month == row_month][col].mode()
			m = 0 if len(m_) == 0 else m_[0]
			factors_ds.loc[idx, col] = m

	# assert that no nans are left
	assert factors_ds.isnull().sum().sum() == 0


	#########################################################################################################
	# %% [markdown]
	# #### Connect raster factors

	# %%
	# topography
	topography_dir = project_files + "topography/"
	elevation_path = topography_dir + "elevation.tif"
	slope_path = topography_dir + "slope.tif"
	aspect_path = topography_dir + "aspect.tif"
	# topography columns names
	elevation = "elevation"
	slope = "slope"
	aspect = "aspect"

	# vegetation
	vegetation_dir = project_files + "vegetation/"
	vegetation_types_path = vegetation_dir + "CompositeMerged.tif"
	# vegetation column name
	vegetation_t = "vegetation_type"

	print(elevation_path, slope_path, aspect_path, vegetation_types_path, elevation, slope, aspect, vegetation_t, sep="\n")
	print("We'll use a dataset with the connected meteo factors onwards")
	print(factors_ds.head(5))


	# %% [markdown]
	# ##### Connect raster factors functions

	def get_geometry_by_geom_type(
		geometry_column: gpd.GeoSeries, geom_type: str, raster_crs: rasterio.crs.CRS
	) -> tuple[gpd.GeoSeries, gpd.GeoSeries]:
		"""
		Description: extract the geometry data with a given type and in a given crs
		Params:
		geometry_column - a column that contains the GeoDataFrame geometry
		geom_type - a geometry type to be extracted
		raster_crs - a crs of a raster layer
		Returns: the GeoSeries that contains the geometry column with the given type and in the given crs and a mask according
		to which this series was extracted from the original column
		"""
		# print(type(geometry_column))
		geom_mask = geometry_column.geometry.geom_type == geom_type
		masked = geometry_column[geom_mask].to_crs(raster_crs).geometry
		return masked, geom_mask


	# %%
	def add_raster_factor(factor: str, factor_path: str, 
						  dataset: gpd.GeoDataFrame, copy: bool = True, 
						  statistics: list=["mean"], add_stats: dict | None = None,
						 ) -> gpd.GeoDataFrame:
		"""
		Description: extracts values from the raster file describing the factor in a given vector geometry dataset
		Params:
		factor - a factor to extract
		factor_path - a path to the factor raster file 
		dataset - a GeoDataFrame with the geometry column
		copy - copy the dataset
		statistics - the statistics to apply to polygons, default: mean, should be a list
		add_stats - is an optional argument that will be passed to the zonal statistics to calculate the custom statistics
		(for ex, get the array of pixel values that intersect the given geometry). The argument should be passed in the
		following form: {'statistics_name':function_name}. Only for print use, if not none not printed
		Returns: the dataset with the factor which values are extracted from the corresponding raster file
		Limitations: works only with the first raster band and computes zonal statistics only for the first 
		value in the statistics list
		"""
		# Algorithm
		# add a factor column to the dataset
		# open the raster file
		# get the raster nodata value 
		# get all available geometry types in the dataset
		# for each geom type: 
		# separate points and polygons using get_geometry_by_geom_type function
		# if type==point, perform sampling using rasterio, if type==(multi)polygon apply zonal statistics from rasterstats
		# write values to the factor column
		# endfor
		# close file
		if copy:
			dataset = dataset.copy()
		
		dataset[factor] = 0.0
		raster_band = 1
		
		factor_raster = rasterio.open(factor_path)
		factor_raster_ds = factor_raster.read(raster_band)
		# nodatavals returns the nodata value for each band, here we get only for the first one
		nodata_value = factor_raster.nodatavals[raster_band - 1]

		factor_raster_ds = factor_raster_ds.astype("float64")
		# print(factor_raster_ds[:5], factor_raster_ds.dtype)
		
		# print zonal statistic valid statistics: the majority stands for the mode
		# print(rasterstats.utils.VALID_STATS)

		# factor_raster mask if needed
		# factor_raster_m = factor_raster.read_masks(raster_band)
		# print(factor_raster_m)
		# print("No data", nodata_value, factor_raster.dtypes)

		geometry_column = dataset.geometry
		geom_types = geometry_column.geometry.geom_type.unique()
		# print(geom_types)

		supported_geom_types = ["Point", "Polygon", "MultiPolygon"]
		for geom_type in geom_types:        
			geometry_by_type, geometry_by_type_mask = get_geometry_by_geom_type(geometry_column, geom_type, factor_raster.crs)
			# print(geometry_by_type[:2], geometry_by_type_mask[:2])
			if geom_type == supported_geom_types[0]:
				coord_list = [(x, y) for x, y in zip(geometry_by_type.x, geometry_by_type.y)]
				# print(coord_list[:5])
				factor_l = [x[0] for x in factor_raster.sample(coord_list, masked=True)]
				# to get rid of the masked values by converting them to nan
				factor_arr = np.array(factor_l).astype(float)  
				# print(len(factor_l))
				# print(factor_arr, type(factor_l[1460-1215+4:1469-1215][0]), factor_arr[1460-1215+4:1469-1215][0], type(factor_arr[1460-1215+4:1469-1215][0]))
				dataset.loc[geometry_by_type_mask, factor] = factor_arr     # assign values
				# print(dataset.iloc[1465])
				# print(dataset[geometry_by_type_mask], dataset[geometry_by_type_mask].shape, dataset[geometry_by_type_mask].geometry.geom_type.unique())
				assert dataset[geometry_by_type_mask].geometry.geom_type.unique() == [geom_type]
			elif geom_type in supported_geom_types[1:]:
				transform = factor_raster.transform
				# source: https://pythonhosted.org/rasterstats/rasterstats.html
				# params: polygons geopandas dataframe (or path to the shp file), an array from a raster image, a coordinates
				# transformation matrix and the stats to count. all_touched is used to include pixels that're touched by
				# the geometry (by default if the geometry center doesn't intersect with the pixel that doesn't count) and
				# nodata_value is used to exclude nodata value from computation
				# pass only the geometry column as the function works a little bit faster
				# print("Statistics:", statistics)
				stats = zonal_stats(
					geometry_by_type, factor_raster_ds, 
					affine=transform, stats=statistics,
					all_touched=True, nodata=nodata_value, add_stats=add_stats,  #raster_out=False,
				)
				if not add_stats is None:
					print(stats)
				poly_stats_l = [x[statistics[0]] for x in stats]
				# print(poly_stats_l[:5])
				# to get rid of the "masked" values
				poly_stats_arr = np.array(poly_stats_l).astype(float)
				dataset.loc[geometry_by_type_mask, factor] = poly_stats_arr
				# print(dataset[geometry_by_type_mask], dataset[geometry_by_type_mask].shape, dataset[geometry_by_type_mask].geometry.geom_type.unique())
				assert dataset[geometry_by_type_mask].geometry.geom_type.unique() == [geom_type]
		
		if not factor_raster.closed:
			# print("Closing raster dataset")
			factor_raster.close()
		
		return dataset


	# %%
	def my_stat(x):
		"""
		Pass as an argument "add_stats" to the add_raster_factor function to get the underlying array of pixel values 
		that intersect the polygons
		"""
		return np.ma.getdata(x)


	# connect raster factors
	statistics_ = ["mean", "majority"]
	print(statistics_[1])
	raster_ds = add_raster_factor(elevation, elevation_path, factors_ds)
	add_raster_factor(slope, slope_path, raster_ds, copy=False)
	# add_raster_factor(aspect, aspect_path, raster_ds, copy=False)
	add_raster_factor(aspect, aspect_path, raster_ds, copy=False, statistics=[statistics_[1]])
	add_raster_factor(vegetation_t, vegetation_types_path, raster_ds, copy=False, statistics=[statistics_[1]])
	print(raster_ds)

	# describe dataset
	e = elevation
	s = slope
	asp = aspect
	vt = vegetation_t
	print(raster_ds[[elevation, slope, aspect, vegetation_t]].describe())

	# %% [markdown]
	# #####  None type values fill

	# check for nans
	# nan masks
	el_nan_m = raster_ds.elevation.isnull()
	slope_nan_m = raster_ds.slope.isnull()
	asp_nan_m = raster_ds.aspect.isnull()
	vt_nan_m = raster_ds.vegetation_type.isnull()
	print(
		"The nans number for each raster factor:",
		raster_ds[el_nan_m].shape, raster_ds[slope_nan_m].shape, raster_ds[asp_nan_m].shape, raster_ds[vt_nan_m].shape
	)
	print(
		"Nans in the raster factors",
		raster_ds[el_nan_m],
		raster_ds[slope_nan_m],
		raster_ds[asp_nan_m],
		raster_ds[vt_nan_m],
		sep="\n"
	)

	# first drop the vegetation type where nan
	vt_nan_idx = raster_ds[vt_nan_m].index
	print(vt_nan_idx)
	raster_ds_clean = raster_ds.drop(index=vt_nan_idx).reset_index(drop=True)
	assert abs(raster_ds_clean.shape[0] - raster_ds.shape[0]) == vt_nan_idx.shape[0]
	raster_ds_clean[raster_ds_clean.vegetation_type.isnull()].shape
	# raster_ds_clean.head(2), raster_ds_clean.index
	# raster_ds_clean.is_fire.value_counts()

	# second: fill elevation and slope nans with the mean and aspect
	# with the mode value for each column
	# !!! Attention: changes the original dataset
	prec = 6
	el_nan_idx = raster_ds_clean[raster_ds_clean.elevation.isnull()].index
	raster_ds_clean.loc[el_nan_idx, elevation] = raster_ds_clean.elevation.mean().round(prec)

	slope_nan_idx = raster_ds_clean[raster_ds_clean.slope.isnull()].index
	raster_ds_clean.loc[slope_nan_idx, slope] = raster_ds_clean.slope.mean().round(prec)

	asp_nan_idx = raster_ds_clean[raster_ds_clean.aspect.isnull()].index
	raster_ds_clean.loc[asp_nan_idx, aspect] = raster_ds_clean.aspect.mode()[0].round(prec)

	# print(el_nan_idx, slope_nan_idx, asp_nan_idx, sep="\n")
	# print(raster_ds_clean.loc[el_nan_idx, elevation], raster_ds_clean.loc[slope_nan_idx, slope],
	#       raster_ds_clean.loc[asp_nan_idx, aspect], sep="\n")

	# finally check all columns for nones/nans
	print("None values left in the dataset:", raster_ds_clean.isnull().sum(), sep="\n")

	#########################################################################################################
	# %% [markdown]
	# #### Connect social factors

	# social
	social_dir = project_files + "social/"
	roads_path = social_dir + "auto_roads.geojson"
	rivers_path = social_dir + "rivers.geojson"
	localities_path = social_dir + "localities_Irk_obl.geojson"
	techno_objects_path = social_dir + "techno_obj.csv"
	print(roads_path, rivers_path, localities_path, techno_objects_path, sep="\n")
	print("We'll use a dataset with the connected meteo and raster factors onwards")
	print(raster_ds_clean.head(5))

	# write the dataset with the connected social factors to the social_fac_ds
	social_fac_ds = None

	# connect roads
	roads = gpd.read_file(roads_path)
	if not roads.crs == project_crs:
		print(f"Converting \"{type(roads)}\" to {project_crs}")
		roads = roads.to_crs(project_crs)
	print(roads.head(5), roads.dtypes, roads.shape, roads.crs, sep="\n")

	# select columns we need
	keep_cols = ["type", "id", "geometry"]
	roads_ = roads[keep_cols]
	roads_.rename(columns={"type" : "road_type", "id" : "road_id"}, inplace=True)

	# join datasets by distance
	roads_ds = gpd.sjoin_nearest(raster_ds_clean, roads_, how="left", distance_col="road_dist")
	roads_ds.drop(columns=["index_right"], inplace=True)
	print(roads_ds.columns, roads_ds.shape, raster_ds_clean.shape, sep="\n")

	# drop duplicates in the index
	print("Duplicates indeces: ", roads_ds[roads_ds.index.duplicated()].index)
	roads_ds = roads_ds[~roads_ds.index.duplicated(keep="first")]    # keep only the first found duplicated row
	print("Duplicates left:", roads_ds.index.duplicated().sum())
	roads_ds.reset_index(drop=True, inplace=True)
	roads_ds.index, roads_ds


	# connect rivers
	rivers = gpd.read_file(rivers_path)
	if not rivers.crs == project_crs:
		print(f"Converting \"{type(rivers)}\" to {project_crs}")
		rivers = rivers.to_crs(project_crs)
	print(rivers.head(5), rivers.dtypes, rivers.shape, rivers.crs, sep="\n")

	# select columns we need
	keep_cols = ["id", "geometry"]
	rivers_ = rivers[keep_cols]
	rivers_.rename(columns={"id" : "river_id"}, inplace=True)

	# join datasets by distance (use dataset with the connected roads)
	rivers_ds = gpd.sjoin_nearest(roads_ds, rivers_, how="left", distance_col="river_dist")
	rivers_ds.drop(columns=["index_right"], inplace=True)
	print(rivers_ds.columns, rivers_ds.shape, raster_ds_clean.shape, sep="\n")

	# drop duplicates in the index
	print("Duplicates indeces: ", rivers_ds[rivers_ds.index.duplicated()].index)
	rivers_ds = rivers_ds[~rivers_ds.index.duplicated(keep="first")]    # keep only the first found duplicated row
	print("Duplicates left:", rivers_ds.index.duplicated().sum())
	rivers_ds.reset_index(drop=True, inplace=True)
	rivers_ds.index, rivers_ds


	# connect localities
	locs = gpd.read_file(localities_path)
	if not locs.crs == project_crs:
		print(f"Converting \"{type(locs)}\" to {project_crs}")
		locs = locs.to_crs(project_crs)
	print(locs.head(5), locs.columns, locs.dtypes, locs.shape, locs.crs, sep="\n")

	# select columns we need
	keep_cols = ["type", "id", "geometry"]
	locs_ = locs[keep_cols]
	locs_.rename(columns={"type" : "locality_type", "id" : "locality_id"}, inplace=True)

	# join datasets by distance (use rivers_ds)
	locs_ds = gpd.sjoin_nearest(rivers_ds, locs_, how="left", distance_col="locality_dist")
	locs_ds.drop(columns=["index_right"], inplace=True)
	print(locs_ds.columns, locs_ds.shape, raster_ds_clean.shape, sep="\n")

	# drop duplicates in the index
	print("Duplicates indeces: ", locs_ds[locs_ds.index.duplicated()].index)
	locs_ds = locs_ds[~locs_ds.index.duplicated(keep="first")]    # keep only the first found duplicated row
	print("Duplicates left:", locs_ds.index.duplicated().sum())
	locs_ds.reset_index(drop=True, inplace=True)
	locs_ds.index, locs_ds.head(2)


	# connect techno objects
	techno_obj_raw = gpd.pd.read_csv(techno_objects_path, sep=";", usecols=["id", "WKT"])
	techno_obj_geom = gpd.GeoSeries.from_wkt(techno_obj_raw["WKT"])
	techno_obj = gpd.GeoDataFrame(data=techno_obj_raw, geometry=techno_obj_geom, crs="epsg:4326")
	if not techno_obj.crs == project_crs:
		print(f"Converting \"{type(techno_obj)}\" to {project_crs}")
		techno_obj = techno_obj.to_crs(project_crs)
	print(techno_obj.head(5), techno_obj.columns, techno_obj.dtypes, techno_obj.shape, techno_obj.crs, sep="\n")

	# select columns we need
	keep_cols = ["geometry"]
	techno_obj_ = techno_obj[keep_cols]

	# join datasets by distance (use locs_ds)
	techno_obj_ds = gpd.sjoin_nearest(locs_ds, techno_obj_, how="left", distance_col="techno_obj_dist")
	techno_obj_ds.drop(columns=["index_right"], inplace=True)
	print(techno_obj_ds.columns, techno_obj_ds.shape, raster_ds_clean.shape, sep="\n")

	# drop duplicates in the index
	print("Duplicates indeces: ", techno_obj_ds[techno_obj_ds.index.duplicated()].index)
	social_fac_ds = techno_obj_ds[~techno_obj_ds.index.duplicated(keep="first")]    # keep only the first found duplicated row
	print("Duplicates left:", social_fac_ds.index.duplicated().sum())
	social_fac_ds.reset_index(drop=True, inplace=True)


	# some cleaning
	# drop rows where the distance to the locality < 50 m or to the techno object < 100 m
	# then drop techno_obj_dist colum
	loc_col = "locality_dist"
	tech_col = "techno_obj_dist"
	idx_ = social_fac_ds[(social_fac_ds[loc_col] < 50) | (social_fac_ds[tech_col] < 100)].index
	print(social_fac_ds[(social_fac_ds[loc_col] < 50)].shape, social_fac_ds[(social_fac_ds[tech_col] < 100)].shape)
	s_ds = social_fac_ds.drop(index=idx_)

	soc_fac_ds = s_ds.reset_index(drop=True)
	print(soc_fac_ds.shape, social_fac_ds.shape)
	print(
	    "Distance to locality < 50:",soc_fac_ds[soc_fac_ds[loc_col] < 50].shape, 
	    "Distance to techno obj < 100:", soc_fac_ds[soc_fac_ds[tech_col] < 100].shape,
	    sep="\n"
	)
	print("Duplicates in idx:", soc_fac_ds.index.duplicated().sum())
	vc = soc_fac_ds.is_fire.value_counts()
	print("Fires number:", vc[1], "Non-fires number:", vc[0])

	print("Drop techno_obj_dist column")
	soc_fac_ds.drop(columns=["techno_obj_dist"], inplace=True)
	print("Dropped?", not soc_fac_ds.columns.isin(["techno_obj_dist"]).any())
	print("Shape:", soc_fac_ds.shape)
	print("Columns:", soc_fac_ds.columns.to_list())

	# save to file
	write_path = project_files + "fires_with_factors/"
	if not os.path.exists(write_path):
	  os.mkdir(write_path)
	soc_fac_ds.to_file(write_path + "factor_dataset_" + str(data_year) + ".geojson")

	return 0


years = list(range(2015, 2025))
print("Years:", years)
for year_ in years:
	print(f"Processing year {year_}")
	main(year_)
