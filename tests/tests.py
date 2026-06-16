from jimg_int.intensity import test_data

test_data()

###############################################################################

from jimg_int.intensity import FeatureIntensity

# Select intenity are data for 1st Image - healthy

# initiate class
fi = FeatureIntensity()

# check current metadata
fi.current_metadata

# if required, change parameters
fi.set_projection(projection="avg")

fi.set_correction_factorn(factor=0.2)

# fi.set_scale(scale = 0.5)
# fi.set_selection_list(rm_list = [2,5,6,7])
# OR
# load JIMG project where scale and rm_lis is set in project metadata
# fi.load_JIMG_project_(path = '')
# for more information go to: https://github.com/jkubis96/JIMG
# rm_list and scale can be omitted

# load image
fi.load_image_3D(path="test_data/intensity/ctrl/image.tiff")

# or 1D image after projection, be sure that image was not adjusted, for analysis should be use !RAW! image
# fi.load_image_(path)

###############################################################################

fi.load_mask_(path="test_data/intensity/ctrl/mask_1.png")

###############################################################################

fi.load_normalization_mask_(path="test_data/intensity/ctrl/background_1.png")

###############################################################################

# strat calculations
fi.run_calculations()


# get results
results = fi.get_results()


# save results for further analysis, ensuring each feature
# is stored in a separate directory (single directory
# should contain data with the same 'feature_name'),
# this setup allows running fi.concatenate_intensity_data()
# in the specific directory of each feature
# while preventing errors from incorrect feature concatenation

fi.save_results(
    path="",
    mask_region="brain",
    feature_name="Feature1",
    individual_number=1,
    individual_name="CTRL",
)


###############################################################################


# Select intenity are data for 2st Image - disease

# initiate class
fi = FeatureIntensity()

fi.set_projection(projection="avg")

fi.set_correction_factorn(factor=0.2)

fi.load_image_3D(path="test_data/intensity/dise/image.tiff")

###############################################################################


fi.load_mask_(path="test_data/intensity/dise/mask_1.png")

###############################################################################

fi.load_normalization_mask_(path="test_data/intensity/dise/background_1.png")


###############################################################################


fi.run_calculations()

results = fi.get_results()

fi.save_results(
    path="",
    mask_region="brain",
    feature_name="Feature1",
    individual_number=1,
    individual_name="DISEASE",
)


###############################################################################

# concatenate data of experiment 1 & 2
fi.concatenate_intensity_data(directory="", name="example_data")

###############################################################################


import pandas as pd

from jimg_int.intensity import IntensityAnalysis

# initiate class
ia = IntensityAnalysis()


input_data = pd.read_csv("example_data_Feature1_brain.csv")

# check columns
input_data.head()

data = ia.df_to_percentiles(
    data=input_data,
    group_col="individual_name",
    values_col="norm_intensity",
    replication_col="individual_number",
    sep_perc=1,
)

stats = ia.get_stats(data, tested_value="avg")

results = ia.hist_compare_plot(
    data=stats, queue=["CTRL", "DISEASE"], p_adj=True, txt_size=20
)


###############################################################################


results.savefig("example_results.svg", format="svg", dpi=300, bbox_inches="tight")

###############################################################################
