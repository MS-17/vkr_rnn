import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from calendar import monthrange
import os


def main(year_):
	data_year = year_

	base_p = "/home/sergei/Downloads/vkr/trying_rnn/code"

	project_files = base_p + "/project_data/"
	irkutsk_region_path = project_files + "irkutsk_boundaries/"
	regular_grid_path = project_files + "grid_meteo/"
	fires_path = project_files + "fires_nonfires_data/fires/"
	nonfires_path = project_files + "fires_nonfires_data/non_fires/"
	months_range = list(range(3, 11))
	project_crs = "epsg:3857"
	print(
		"Paths:", project_files, irkutsk_region_path, regular_grid_path, fires_path, nonfires_path, 
		"Data year:", data_year, "Data months range:", months_range, "Project crs:", project_crs, sep="\n"
	)

	# initializing datasets
	# irkutsk region polygon
	irkutsk_region = gpd.read_file(irkutsk_region_path + "Irkutsk_region.geojson")
	if not irkutsk_region.crs == project_crs:
		print(f"Converting \"{irkutsk_region.name}\" to {project_crs}")
		irkutsk_region = irkutsk_region.to_crs(project_crs)
	print(irkutsk_region.shape, irkutsk_region.crs, irkutsk_region.geometry, sep="\n")
	# irkutsk_region.head()

	# regular grid
	regular_grid = gpd.read_file(regular_grid_path + "grid_meteo.csv")
	regular_grid.LAT = regular_grid.LAT.astype(float)
	regular_grid.LON = regular_grid.LON.astype(float)
	regular_grid, regular_grid.dtypes
	
	# fires
	fires = gpd.read_file(fires_path + f"fires_{str(data_year)}.shp")
	if not fires.crs == project_crs:
		print(f"Converting \"{fires.name}\" to {project_crs}")
		fires = fires.to_crs(project_crs)
	print(fires.shape, fires.crs, fires.geometry, fires.dtypes, sep="\n")
	fires.head(5)
	# fires.explore()
	
	# non-fires
	non_fires = gpd.read_file(nonfires_path + f"nonfires_{str(data_year)}.shp")
	if not non_fires.crs == project_crs:
		print(f"Converting \"{non_fires.name}\" to {project_crs}")
		non_fires = non_fires.to_crs(project_crs)
	print(non_fires.shape, non_fires.crs, non_fires.geometry, non_fires.dtypes, sep="\n")
	non_fires.head(5)
	
	# map
	# irk_fires_map = irkutsk_region.explore(
	#     color="blue",
	#     tooltip=False,
	#     name="Irkutsk region",
	#     style_kwds=dict(color="black", fillOpacity=0.1),
	#     highlight=False,
	# )
	# fires.explore(m=irk_fires_map, color="red", zorder=1)
	# non_fires.explore(m=irk_fires_map, zorder=2, marker_kwds=dict(radius=2))
	# irk_fires_map
	
	# check if there's any nan value
	print("Fires nulls:", fires.isnull().sum(), "\tNon-fires nulls:", non_fires.isnull().sum())

	#########################################################################################################
	# %% [markdown]
	# ##### Prepare datasets
	
	# make copies of the original datasets (the following code works with them)
	fires_cp = fires.copy()
	nonfires_cp = non_fires.copy()
	print(fires_cp.head(1), fires_cp.dtypes, fires.shape, sep="\n")
	print(nonfires_cp.head(1), nonfires_cp.dtypes, nonfires_cp.shape, sep="\n")
	
	# convert time columns to GMT+8, Irkutsk
	fires_cp["dt_first"] = gpd.pd.to_datetime(fires_cp.dt_first) + gpd.pd.DateOffset(hours=8)
	fires_cp["dt_last"] = gpd.pd.to_datetime(fires_cp.dt_last) + gpd.pd.DateOffset(hours=8)
	
	# check march and november
	# print("March:", fires_cp[(fires_cp.dt_first.dt.month < 4)].shape[0], "Nov:", fires_cp[(fires_cp.dt_first.dt.month > 10)].shape[0])
	# print("March-Apr:", fires_cp[(fires_cp.dt_first.dt.month == 3) & (fires_cp.dt_last.dt.month == 4)].shape[0])
	# print("Oct-Nov:", fires_cp[(fires_cp.dt_first.dt.month == 10) & (fires_cp.dt_last.dt.month == 11)].shape[0])
	
	# # drop dt_last <= feb and dt_first >= nov
	# # !!! Attention: changes the original dataset (perform only once)
	feb_fires = fires_cp[fires_cp.dt_last.dt.month <= 2].index
	nov_fires = fires_cp[fires_cp.dt_first.dt.month >= 11].index
	print(feb_fires, nov_fires, sep="\n")
	print(feb_fires.shape[0], nov_fires.shape[0], fires_cp.shape[0])
	fires_cp.drop(feb_fires, axis=0, inplace=True)
	fires_cp.drop(nov_fires, axis=0, inplace=True)
	print(fires_cp.shape[0], nonfires_cp.shape[0])

	#########################################################################################################
	# %% [markdown]
	# ##### Generate non-fires dates
	
	# keys should be sorted in the ascending order
	# count fire events for each month
	# there're going to be march fires (dt_first=march and dt_last=april)
	fires_stat = {}
	for i in months_range:
		fires_stat[i] = fires_cp[
			(fires_cp.dt_first.dt.month == i) |
			((fires_cp.dt_first.dt.month == i - 1) & (fires_cp.dt_last.dt.month == i))
		].shape[0]
		print(fires_cp[fires_cp.dt_first.dt.month == i].shape[0], fires_cp[(fires_cp.dt_first.dt.month == i - 1) & (fires_cp.dt_last.dt.month == i)].shape[0])
	print(fires_stat, fires_cp.shape[0])
	
	print("fires_stat values sum:", sum(fires_stat.values()))
	try:
		assert sum(fires_stat.values()) == fires_cp.shape[0]
	except AssertionError:
		print("Assert failed")
	
	
	def gen_nonfire_stat(monthly_fire_occurence: dict) -> dict:
		"""
		Description:
		Generate the non-fire statistics for the multiple months
		Creates a dictionary that contains the monthly non-fire occurences number
		Parameters:
		monthly_fire_occurence is a dictionary representing the number of fire occurences in a month
		key = a month number
		value = a fires number in the according month
		Returns:
		The dictionary that contains the monthly non-fire occurences number
		"""
		monthly_fire_occurence = dict(sorted(monthly_fire_occurence.items()))
		fires_l = list(monthly_fire_occurence.values())
		# print(fires_l)
	
		min_max_list = sorted(fires_l)
		while len(min_max_list) > 1:
			# print(min_max_list)
	
			min_val = min_max_list[0]
			max_val = min_max_list[-1]
			# print(min_val, max_val)
	
			if min_val == max_val:
				return monthly_fire_occurence
	
			idx_min = [i for i, e in enumerate(fires_l) if e == min_val]
			idx_max = [i for i, e in enumerate(fires_l) if e == max_val]
			# print(idx_min, idx_max)
	
			for i in idx_min:
				fires_l[i] = max_val
			for i in idx_max:
				fires_l[i] = min_val
	
			min_max_list = [i for i in min_max_list if i != min_val]
			min_max_list = [i for i in min_max_list if i != max_val]
			# print(min_max_list)
			# print(fires_l)
	
		monthly_non_fire_occurence = {}
		i = 0
		for k in monthly_fire_occurence.keys():
			monthly_non_fire_occurence[k] = fires_l[i]
			i+=1
	
		return monthly_non_fire_occurence

	nonfires_stat = gen_nonfire_stat(fires_stat)
	print("Fires statistics:", fires_stat)
	print("Non-fires statistics:", nonfires_stat)

	# what if I don't generate the revert distribution?
	# nonfires_stat = fires_stat.copy()
	# print("Fires statistics:", fires_stat)
	# print("Non-fires statistics:", nonfires_stat)


	# make so that the non-fires distribution count sum becomes equal to the non-fires number 
	stat_diff = nonfires_cp.shape[0] - sum(nonfires_stat.values())
	if stat_diff < 0:
		raise ValueError("Non-fires number is less than the fires number, not supported")
	keys_num = len(nonfires_stat.keys())
	p_ = np.round(stat_diff / keys_num)
	if 0 < p_ < 1.0:
		k_ = list(nonfires_stat.keys())[-1]
		nonfires_stat[k_] += stat_diff
	if p_ >= 1.0:
		p_ = int(p_)
		add_l = [p_] * keys_num
		# p_ = 20 stat_diff = 143, 143 - 20 * 7 = 143 - 140 = +3
		# p_ = 39 stat_diff = 271 271 - 39 * 7 = 271 - 273 = -2
		add_l[-1] = add_l[-1] + (stat_diff - p_ * keys_num)
		# print(p_, add_l)
		for idx, k in enumerate(list(fires_stat.keys())):
			nonfires_stat[k] += add_l[idx]
	print(nonfires_stat)
	print(stat_diff, nonfires_cp.shape[0], sum(nonfires_stat.values()))
	print("Assert that nonfires_stat values sum is equal to the number of nonfires")
	assert sum(nonfires_stat.values()) == nonfires_cp.shape[0]
	
	
	def get_nonfire_dist(nonfires_stat: dict, year: int) -> list:
		"""
		Description:
		Generates a non-fires normal distribution for a month
		Parameters:
		nonfires_stat - a dict, such that the key is a month number and the value is a number of the non-fire occurences
		year - a year
		Returns:
		A dict that contains a monthly non-fire occurences distribution
		"""
		nonfires_dist = {}
		for month, nonfires_num in nonfires_stat.items():
			_, days = monthrange(year, month)
			mean = days / 2 + 2   # mean is a half of a month
			std_dev = days / 3    # standard deviation is a third of a month
			n_dist = np.random.normal(mean, std_dev, nonfires_num).round().astype(int)
			n_dist = np.clip(n_dist, 1, days)   # clip the dist values to fit in the range of (1, days)
			nonfires_dist[month] = n_dist
		return nonfires_dist
	
	nonfires_dist = get_nonfire_dist(nonfires_stat, data_year)
	print("Non-fires distribution length for each month:", [len(x) for x in nonfires_dist.values()])
	print(
		"Non-fires distribution length for each month is equal to the number of non-fires:",
		sum([len(x) for x in nonfires_dist.values()]) == nonfires_cp.shape[0]
	)
	

	# generate non-fires dates list for all months
	nonfires_dates = []
	for month, nonfires_num in nonfires_dist.items():
		# print(month, nonfires_num)
		year_month = str(data_year) + "-" + str(month)
		j = list(map(lambda x: "-".join([year_month, str(x)]), nonfires_num))
		# print(j)
		nonfires_dates += j
	print("Non-fires dates list length:", len(nonfires_dates))
	print("Non-fires dates list (the first 10 el)", nonfires_dates[:10])
	
	# shuffle the non-fires dates list
	rng = np.random.default_rng()        # init random generator
	nonfires_dates_l = rng.permuted(nonfires_dates)
	print("Shuffled non-fires dates list length:", len(nonfires_dates_l))
	print("Shuffled non-fires dates list (the first 10 el)", nonfires_dates_l[:10])
	
	# convert non-fires dates list to series
	nonfires_dates_s = gpd.pd.Series(nonfires_dates_l, dtype="datetime64[ns]")
	print(nonfires_dates_s.head(5))
	
	# plot the fires non-fires dates histogram 
	# nonfires_m = nonfires_dates_s.dt.month
	# fires_m = []
	# for k, v in fires_stat.items():
	# 	fires_m += v * [k]
	# fig, ax = plt.subplots(1, 2,)
	# months_ = [int(m) for m in months_range]
	# labels_ = map(lambda x: str(x), months_)
	# plt.xticks(months_, labels_)
	# plt.ylim(0, 400)
	# # plt.xlim(3, 11)
	# ax[0].hist(fires_m,  bins=np.arange(12.5) - 0.5, rwidth=0.4)
	# # plt.xlim(3, 11)
	# ax[1].hist(nonfires_m, bins=np.arange(15) - 0.5, rwidth=0.4)
	# plt.show()

	# fires in september
	# a = nonfires_dates_s[nonfires_dates_s.dt.month == 9].dt.day
	# # print(a[a==1].size)
	# months_ = [int(m) for m in range(31)]
	# labels_ = map(lambda x: str(x), months_)
	# fig, ax = plt.subplots(1, 1)
	# plt.xticks(months_, labels_)
	# plt.xlim(0, 31)
	# # plt.gca().set_aspect("equal", adjustable="box")
	# # plt.axis("auto")
	# plt.hist(a, bins=np.arange(31) - 0.5, rwidth=0.6)
	# plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	# plt.show()
	# return 0


	#########################################################################################################
	# %% [markdown]
	# ##### Unite fires and non-fires datasets, drop the unnecessary columns
	
	# create a date column in the non-fires dataset
	nonfires_cp["dt_first"] = nonfires_dates_s
	nonfires_cp.shape, nonfires_cp.dtypes
	
	# concatenate fires and non-fires datasets
	fires_nonfires_ds = gpd.pd.concat([fires_cp, nonfires_cp]).reset_index(drop=True)
	print(fires_nonfires_ds.shape, fires_cp.shape, fires_cp.shape[0] + nonfires_cp.shape[0])
	# display 10 elements in the middle
	a = fires_nonfires_ds.iloc[fires_cp.shape[0] - 10:fires_cp.shape[0] + 10, :]
	print("Display 10 el in the middle:", a.shape, a.dtypes, a.reset_index(drop=True), sep="\n")
	
	# fires_nonfires_ds.explore()
	
	# mark is_fire (1 if fire, 0 if non-fire)
	fires_nonfires_ds["is_fire"] = 1
	# print(fires_nonfires_ds["is_fire"])
	fires_nonfires_ds.loc[fires_nonfires_ds.fire_id.isnull(), "is_fire"] = 0
	print(
		fires_nonfires_ds.head(1), fires_nonfires_ds.tail(1),
		fires_nonfires_ds.is_fire.value_counts(),
		sep="\n"
	)
	
	final_ds = fires_nonfires_ds.drop(["fire_id", "dt_last", "dt_liq", "Area", "Area_les", "id"], axis=1)
	assert final_ds.isnull().any().sum() == 0
	
	# small corrections: change year type to int, cut the time part of the value in dt_first
	# and rename dt_first column to event_date
	final_ds.year = final_ds.year.astype("int32")
	final_ds.dt_first = final_ds.dt_first.dt.normalize()
	result_ds = final_ds.rename(columns={"dt_first" : "event_date"})
	print(result_ds.dtypes)
	result_ds.head(5)

	#########################################################################################################
	# %% [markdown]
	# ##### Connect the regular grid
	
	# use result_ds and regular_grid
	print(result_ds.head(5), regular_grid.head(5))
	
	# convert regular grid pandas dataframe to geopandas dataframe
	points = gpd.points_from_xy(x=regular_grid.LON, y=regular_grid.LAT, crs="EPSG:4326")
	regular_grid["geometry"] = points.to_crs(project_crs)
	regular_grid = gpd.GeoDataFrame(regular_grid)
	regular_grid
	
	# join datasets by distance
	grid_ds = result_ds.sjoin_nearest(regular_grid, how="left")
	grid_ds.drop("index_right", axis=1, inplace=True)
	print("After sjoin:", grid_ds.dtypes, sep="\n")
	
	# m = grid_ds.explore()
	# regular_grid.explore(m=m, color="red")

	# rename columns
	# !!! Attention: changes the original dataset (run only once)
	grid_ds.rename(columns={"LAT" : "grid_lat"}, inplace=True)
	grid_ds.rename(columns={"LON" : "grid_lon"}, inplace=True)
	grid_ds.columns
	
	# check for duplicates and drop them
	dup_idx = grid_ds[grid_ds.index.duplicated()].index
	print("Duplicates in index:", grid_ds[grid_ds.index.duplicated()].index)
	grid_ds = grid_ds[~grid_ds.index.duplicated(keep="first")]
	print("Duplicates in index:", grid_ds.index.duplicated().sum())
	grid_ds.reset_index(drop=True, inplace=True)
	grid_ds.index
	
	# check that the number of (non)fires correspond to the origanl (non)fires datasets
	vc = grid_ds.is_fire.value_counts() 
	print("Fires number is equal:", vc[1] == fires_cp.shape[0])
	print("Non-fires number is equal:", vc[0] == nonfires_cp.shape[0])
	
	# write file
	write_path = project_files + "dataset_plus_grid/"
	if not os.path.exists(write_path):
		os.mkdir(write_path)
	grid_ds.to_file(write_path + "dataset_" + str(data_year) + ".geojson")
	
	return 0


years = list(range(2015, 2025))
print("Years:", years)
for year_ in years:
	print(f"Processing year {year_}")
	main(year_)
