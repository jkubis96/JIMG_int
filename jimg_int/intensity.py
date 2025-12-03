import json
import os
import random
import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from scipy.stats import chi2_contingency
from tqdm import tqdm

import jimg_int.config as cfg

from .utils import *

random.seed(42)


class FeatureIntensity(ImageTools):

    def __init__(
        self,
        input_image=None,
        image=None,
        normalized_image_values=None,
        mask=None,
        background_mask=None,
        typ=None,
        size_info=None,
        correction_factor=None,
        img_type=None,
        scale=None,
        stack_selection=None,
    ):

        self.input_image = input_image or None
        self.image = image or None
        self.normalized_image_values = normalized_image_values or None
        self.mask = mask or None
        self.background_mask = background_mask or None
        self.typ = typ or "avg"
        self.size_info = size_info or None
        self.correction_factor = correction_factor or 0.1
        self.scale = scale or None
        self.stack_selection = stack_selection or []

    @property
    def current_metadata(self):
        """
        This property returns current metadata parameters.


        Returns:
            projection_type (str) - type of data projection, if 3D-image
            correction_factor (str) - correction factor for backgroud separation during pixels intensity normalization
                The formula applied for each target_mask_pixel is:

                    Result_{i,j} = T_{i,j} - (mean(B) * (1 + c))

                Where:
                * Result_{i,j} – the result value for each pixel (i,j) in the target mask
                * T_{i,j} – intensity value of pixel (i,j) in the target mask
                * mean(B) – the mean intensity of the pixels in the normalization_mask
                * c – correction factor

            scale (str) - scale loaded using the load_JIMG_project_() method or set by the set_scale() method
            stack_selection (list) - list of partial z-axis images from the full 3D-image to exclude from the projection

        """

        print(f"Projection type: {self.typ}")
        print(f"Correction factor: {self.correction_factor}")
        print(f"Scale (unit/px): {self.scale}")
        print(f"Selected stac to remove: {self.stack_selection}")

        return self.typ, self.correction_factor, self.scale, self.stack_selection

    def set_projection(self, projection: str):
        """
        This method sets 'projection' parameter. The projection is a parameter used for 3D-image projection to 1D-image.

        Args:
           projection (str) - the projection value ['avg', 'median', 'std', 'var', 'max', 'min']. Default: 'avg'

        """

        t = ["avg", "median", "std", "var", "max", "min"]
        if projection in t:
            self.typ = projection
        else:
            print(f"\nProvided parameter is incorrect. Avaiable projection types: {t}")

    def set_correction_factorn(self, factor: float):
        """
        This method sets 'correction_factor' parameter.
        The correction_factor is a parameter used for backgroud separation during pixels intensity normalization

        The formula applied for each target_mask_pixel is:

            Result_{i,j} = T_{i,j} - (mean(B) * (1 + c))

        Where:
        * Result_{i,j} – the result value for each pixel (i,j) in the target mask
        * T_{i,j} – intensity value of pixel (i,j) in the target mask
        * mean(B) – the mean intensity of the pixels in the normalization_mask
        * c – correction factor


        Args:
           factor (float) - the correction_factor value [factor < 1 and factor > 0]. Default: 0.1

        """

        if factor < 1 and factor > 0:
            self.correction_factor = factor
        else:
            print(
                "\nProvided parameter is incorrect. The factor should be a floating-point value within the range of 0 to 1."
            )

    def set_scale(self, scale):
        """
        This method sets the 'scale' parameter. The scale is used to calculate the actual size of the tissue or organ.

        The scale is also loaded using the load_JIMG_project_() method.

        Args:
           scale (float) - the scale value [um/px]

        """

        self.scale = scale

    def set_selection_list(self, rm_list: list):
        """
        This method sets the 'rm_list' parameter. The 'rm_list' is used to exclude partial z-axis images from the full 3D image in the projection.

        Args:
           rm_list (list) - list of images to remove.

        """

        self.stack_selection = rm_list

    def load_JIMG_project_(self, path):
        """
        This method loads a JIMG project. The project file must have the *.pjm extension.

        Args:
            file_path (str) - path to the project file.

        Returns:
            project (class) - loaded project object

        Raises:
            ValueError: If the file does not have a *.pjm extension.

        """

        path = os.path.abspath(path)

        if ".pjm" in path:
            metadata = self.load_JIMG_project(path)

            try:
                self.scale = metadata.metadata["X_resolution[um/px]"]
            except:

                try:
                    self.scale = metadata.images_dict["metadata"][0][
                        "X_resolution[um/px]"
                    ]

                except:
                    print(
                        '\nUnable to set scale on this project! Set scale using "set_scale()"'
                    )

            self.stack_selection = metadata.removal_list

        else:
            print(
                "\nWrong path. The provided path does not point to a JIMG project (*.pjm)."
            )

    def stack_selection_(self):
        if len(self.input_image.shape) == 3:
            if len(self.stack_selection) > 0:
                self.input_image = self.input_image[
                    [
                        x
                        for x in range(self.input_image.shape[0])
                        if x not in self.stack_selection
                    ]
                ]
            else:
                print("\nImages to remove from the stack were not selected!")

    def projection(self):

        if self.typ == "avg":
            img = np.mean(self.input_image, axis=0)

        elif self.typ == "std":
            img = np.std(self.input_image, axis=0)

        elif self.typ == "median":
            img = np.median(self.input_image, axis=0)

        elif self.typ == "var":
            img = np.var(self.input_image, axis=0)

        elif self.typ == "max":
            img = np.max(self.input_image, axis=0)

        elif self.typ == "min":
            img = np.min(self.input_image, axis=0)

        self.image = img

    def detect_img(self):
        check = len(self.input_image.shape)

        if check == 3:
            print("\n3D image detected! Starting processing for 3D image...")
            print(f"Projection - {self.typ}...")

            self.stack_selection_()
            self.projection()

        elif check == 2:
            print("\n2D image detected! Starting processing for 2D image...")

        else:
            print("\nData does not match any image type!")

    def load_image_3D(self, path):
        """
        This method loads an 3D-image (*.tiff) into the class.

        Args:
            path (str) - path to the *.tiff image.

        """
        path = os.path.abspath(path)

        self.input_image = self.load_3D_tiff(path)

    def load_image_(self, path):
        """
        This method loads an image into the class.

        Args:
            path (str) - path to the image.

        """
        path = os.path.abspath(path)

        self.input_image = self.load_image(path)

    def load_mask_(self, path):
        """
        This method loads an image mask into the class. The mask can be in different formats, such as 16-bit or 8-bit, with extensions like *.png and *.jpeg, but it must be in binary format (e.g., 0/2**16-1, 0/255, 0/1, etc.).
        If the `load_normalization_mask_()` method is not used, the mask from the `load_mask_()` method is set as the normalization mask.
        The mean pixel intensity from the area of the reversed normalization mask (where reversed binary == 0 becomes 1 and values greater than 0 become 0) is used for normalization.

        The formula applied for each target_mask_pixel is:

            Result_{i,j} = T_{i,j} - (mean(B) * (1 + c))

            Where:
            * Result_{i,j} – the result value for each pixel (i,j) in the target mask
            * T_{i,j} – intensity value of pixel (i,j) in the target mask
            * mean(B) – the mean intensity of the pixels in the normalization_mask
            * c – correction factor

        Args:
            path (str) - path to the mask image

        """
        path = os.path.abspath(path)

        self.mask = self.load_mask(path)

        print(
            "\nThis mask was also set as the reverse background mask. If you want a different background mask for normalization, use 'load_normalization_mask()'."
        )
        self.background_mask = self.load_mask(path)

    def load_normalization_mask_(self, path):
        """
        This method loads an image mask for normalization into the class. The mask can be in different formats, such as 16-bit or 8-bit, with extensions like *.png and *.jpeg, but it must be in binary format (e.g., 0/2**16-1, 0/255, 0/1, etc.).
        The mean pixel intensity from the area of the reversed normalization mask (where reversed binary == 0 becomes 1 and values greater than 0 become 0) is used for normalization.
        The user defines the mask by drawing the area of interest (tisse, part of tissue, organ, ...), and normalization will be applied to the area that is the inverse of the defined area.

        The formula applied for each target_mask_pixel is:

            Result_{i,j} = T_{i,j} - (mean(B) * (1 + c))

            Where:
            * Result_{i,j} – the result value for each pixel (i,j) in the target mask
            * T_{i,j} – intensity value of pixel (i,j) in the target mask
            * mean(B) – the mean intensity of the pixels in the normalization_mask
            * c – correction factor

        Args:
            path (str) - path to the mask image

        """
        path = os.path.abspath(path)

        self.background_mask = self.load_mask(path)

    def intensity_calculations(self):
        tmp_mask = self.ajd_mask_size(image=self.image, mask=self.mask)
        tmp_bmask = self.ajd_mask_size(image=self.image, mask=self.background_mask)

        selected_values = self.image[tmp_mask == np.max(tmp_mask)]

        threshold = np.mean(self.image[tmp_bmask == np.min(tmp_bmask)])

        # normalization
        final_val = selected_values - (threshold + (threshold * self.correction_factor))

        final_val[final_val < 0] = 0

        tmp_dict = {
            "norm_min": np.min(final_val),
            "norm_max": np.max(final_val),
            "norm_mean": np.mean(final_val),
            "norm_median": np.median(final_val),
            "norm_std": np.std(final_val),
            "norm_var": np.var(final_val),
            "norm_values": final_val.tolist(),
            "min": np.min(selected_values),
            "max": np.max(selected_values),
            "mean": np.mean(selected_values),
            "median": np.median(selected_values),
            "std": np.std(selected_values),
            "var": np.var(selected_values),
        }

        self.normalized_image_values = tmp_dict

    def size_calculations(self):

        tmp_mask = self.ajd_mask_size(image=self.image, mask=self.mask)

        size_px = int(len(tmp_mask[tmp_mask > np.min(tmp_mask)]))

        if self.scale is not None:
            size = float(size_px * self.scale)
        else:
            size = None
            print(
                '\nUnable to calculate real size, scale (unit/px) not provided, use "set_scale()" or load JIMG project .pjm metadata "load_pjm()" to set scale for calculations!'
            )

        non_zero_indices = np.where(tmp_mask == np.max(tmp_mask))

        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])

        max_length_x_axis = int(max_x - min_x + 1)
        max_length_y_axis = int(max_y - min_y + 1)

        tmp_val = {
            "size": size,
            "px_size": size_px,
            "max_length_x_axis": max_length_x_axis,
            "max_length_y_axis": max_length_y_axis,
        }

        self.size_info = tmp_val

    def run_calculations(self):
        """
        This method performs analysis on the image provided by the `load_image_()` method, using either default parameters or parameters set by the user, along with masks loaded by the `load_mask_()` and/or `load_normalization_mask_()` methods.

        To display the current parameters, run:
        - current_metadata

        To set new parameters, run:
        - set_projection()
        - set_correction_factor()
        - set_scale() - cannot be defined
        - set_selection_list() - cannot be defined
        - load_JIMG_project_() - cannot be defined

        Returns:

            For results, use the `get_results()` method.


        """

        if self.input_image is not None:

            if self.mask is not None:

                print("\nStart...")
                self.detect_img()
                self.intensity_calculations()
                self.size_calculations()
                print("\nCompleted!")

    def get_results(self):
        """
        This method returns the results from the `run_calculations()` method in dictionary format.


        Returns:

            results_dict (dict) - dictionary containing results from run_calculations()

        """

        if self.normalized_image_values is not None and self.size_info is not None:

            results = {
                "intensity": self.normalized_image_values,
                "size": self.size_info,
            }

            return results

        else:
            print('\nAnalysis were not conducted. Run analysis "run_calculations()"')

    def save_results(
        self,
        path="",
        mask_region: str = "",
        feature_name: str = "",
        individual_number: int = 0,
        individual_name: str = "",
    ):
        """
        This method saves the results from the `run_calculations()` method in dictionary format to a *.json file.

        Args:

            path (str) - path to the directory for saving the file. If not provided, the current working directory is used
            mask_region (str) - name or identifier of the mask region (e.g., tissue, part of tissue, etc.)
            feature_name (str) - name of the feature being analyzed. It is also processed to replace any underscores or spaces with periods
            individual_number (int) - unique number or identifier for the individual in the analysis (e.g., 1, 2, 3)
            individual_name (str) - name of the individual (e.g., species name, tissue, organoid, etc.)


        The method checks if valid values for `mask_region`, `feature_name`, `individual_number`, and `individual_name` are provided.
        If so, and the results (`normalized_image_values` and `size_info`) from `run_calculations()` exist, it saves them as a dictionary
        in a `.int` file (JSON format) in the specified directory. If the directory does not exist, it is created.

        If the analysis has not been conducted or the provided parameters are incorrect, an error message is printed.

        File name format:
            '<individual_name>_<individual_number>_<mask_region>_<feature_name>.int'

        Raises:
            FileNotFoundError: If the path cannot be created or accessed.
            ValueError: If any of 'mask_region', 'feature_name', 'individual_number', or 'individual_name' is missing or invalid.


        """
        path = os.path.abspath(path)

        if (
            len(mask_region) > 1
            and len(feature_name) > 1
            and individual_number != 0
            and len(individual_name) > 1
        ):

            if self.normalized_image_values is not None and self.size_info is not None:

                results = {
                    "intensity": self.normalized_image_values,
                    "size": self.size_info,
                }

                mask_region = re.sub(r"[_\s]+", ".", mask_region)
                feature_name = re.sub(r"[_\s]+", ".", feature_name)
                individual_number = re.sub(r"[_\s]+", ".", str(individual_number))
                individual_name = re.sub(r"[_\s]+", ".", individual_name)

                full_name = f"{individual_name}_{individual_number}_{mask_region}_{feature_name}"

                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path, exist_ok=True)

                full_path = os.path.join(
                    path, re.sub("\\.json", "", full_name) + ".int"
                )

                with open(full_path, "w") as file:
                    json.dump(results, file, indent=4)

            else:
                print(
                    '\nAnalysis were not conducted. Run analysis "run_calculations()"'
                )

        else:
            print(
                "\nAny of 'mask_region', 'feature_name', 'individual_number', 'individual_name' parameters were provided wrong!"
            )

    def concatenate_intensity_data(self, directory: str = "", name: str = ""):
        """
        This method processes and concatenates intensity data from multiple `.int` files in a specified directory.
        It groups the data by gene (feature) and mask region, and then saves the concatenated results as CSV files.

        Args:

            directory (str) - path to the directory containing the `.int` files. If not provided, the current working directory is used
            name (str): Prefix for the output CSV file names. The final CSV files will be saved in the format '<name>_<gene>_<region>.csv'



        Raises:
            FileNotFoundError: If the directory cannot be accessed or no `.int` files are found.
            ValueError: If the `.int` file format is incorrect or missing expected data.

        Output:
            One CSV file per unique gene-region combination, saved in the specified directory.

        """

        directory = os.path.abspath(directory)

        files_list = [f for f in os.listdir(directory) if f.endswith(".int")]

        genes_set = set([re.sub("\\.int", "", x.split("_")[3]) for x in files_list])
        regions_set = set([re.sub("\\.int", "", x.split("_")[2]) for x in files_list])

        for g in genes_set:
            for r in regions_set:
                json_to_save = {
                    "individual_name": [],
                    "individual_number": [],
                    "norm_intensity": [],
                    "size": [],
                }

                for f in tqdm(files_list):
                    if g in f and r in f:
                        with open(os.path.join(directory, f), "r") as file:
                            data = json.load(file)

                            json_to_save["norm_intensity"] = (
                                json_to_save["norm_intensity"]
                                + data["intensity"]["norm_values"]
                            )
                            json_to_save["individual_name"] = json_to_save[
                                "individual_name"
                            ] + [f.split("_")[0]] * len(
                                data["intensity"]["norm_values"]
                            )
                            json_to_save["individual_number"] = json_to_save[
                                "individual_number"
                            ] + [f.split("_")[1]] * len(
                                data["intensity"]["norm_values"]
                            )
                            json_to_save["size"] = json_to_save["size"] + [
                                data["size"]["px_size"]
                            ] * len(data["intensity"]["norm_values"])

        pd.DataFrame(json_to_save).to_csv(f"{name}_{g}_{r}.csv", index=False)


class IntensityAnalysis:

    def drop_up_df(self, data: pd.DataFrame, group_col: str, values_col: str):
        """
        Removes upper outliers from the DataFrame based on the specified value column and grouping column.
        Outliers are calculated and removed separately for each group defined by the grouping column.

        Args:
            data (pd.DataFrame) - the input DataFrame
            group_col (str) - the name of the column used for grouping
            values_col (str) - the column containing the values from which upper outliers will be removed

        Returns:
            filtered_data (pd.DataFrame) - a filtered DataFrame with the upper outliers removed

        """

        def iqr_filter(group):
            q75 = np.quantile(group[values_col], 0.75)
            q25 = np.quantile(group[values_col], 0.25)
            itq = q75 - q25
            return group[group[values_col] <= (q75 + 1.5 * itq)]

        filtered_data = data.groupby(group_col).apply(iqr_filter).reset_index(drop=True)

        return filtered_data

    def percentiles_calculation(self, values, sep_perc: int = 1):
        """
        Calculates percentiles for a given set of values with a specified separation interval.

        This function computes percentiles from 0 to 100, at intervals defined by the `sep_perc` parameter.
        Additionally, it creates a loopable list of percentile ranges, useful for further data analysis or binning.

        Args:
            values (array-like) - the input data values for which the percentiles are calculated
            sep_perc (int) - the separation between percentiles (default is 1, meaning percentiles are calculated at every 1%)

        Returns:
            percentiles (np.ndarray) - nn array of calculated percentile values
            percentiles_loop (list of tuples) - a list of tuples representing consecutive percentile ranges (e.g., [(0, 1), (1, 2), ...])

        """

        per_vector = values.copy()

        percentiles = np.percentile(per_vector, np.arange(0, 101, sep_perc))
        percentiles[0] = 0

        percentiles_loop = [(i, i + 1) for i in range(int(100 / sep_perc))]

        return percentiles, percentiles_loop

    def to_percentil(self, values, percentiles, percentiles_loop):
        """
        Aggregates statistics for a given set of values based on calculated percentile ranges.

        This function calculates summary statistics (e.g., count, average, median, standard deviation, and variance) for each percentile range
        in `percentiles_loop`. The results are based on the percentiles calculated in the `percentiles_calculation()` method.

        Parameters:
            values [array-like] - the input data values for which the statistics are calculated
            percentiles [np.ndarray] - the array of percentile values used to define the ranges
            percentiles_loop [list of tuples] - a list of tuples representing consecutive percentile ranges (e.g., [(0, 1), (1, 2), ...])

        Returns:
            data (dict) - a dictionary containing the following keys:
                - 'n' (list): The number of elements in each percentile range
                - 'n_standarized' (list): The proportion of elements in each percentile range relative to the total number of elements
                - 'avg' (list): The average value of elements within each percentile range
                - 'median' (list): The median value of elements within each percentile range
                - 'std' (list): The standard deviation of elements within each percentile range
                - 'var' (list): The variance of elements within each percentile range

        """

        per_vector = values.copy()

        data = {
            "n": [],
            "n_standarized": [],
            "avg": [],
            "median": [],
            "std": [],
            "var": [],
        }

        amount = len(per_vector)

        for x in percentiles_loop:
            if (
                len(
                    per_vector[
                        (per_vector > percentiles[x[0]])
                        & (per_vector <= percentiles[x[1]])
                    ]
                )
                > 0
            ):
                data["n"].append(
                    len(
                        per_vector[
                            (per_vector > percentiles[x[0]])
                            & (per_vector <= percentiles[x[1]])
                        ]
                    )
                )
                data["n_standarized"].append(
                    len(
                        per_vector[
                            (per_vector > percentiles[x[0]])
                            & (per_vector <= percentiles[x[1]])
                        ]
                    )
                    / amount
                )
                data["avg"].append(
                    np.mean(
                        per_vector[
                            (per_vector > percentiles[x[0]])
                            & (per_vector <= percentiles[x[1]])
                        ]
                    )
                )
                data["std"].append(
                    np.std(
                        per_vector[
                            (per_vector > percentiles[x[0]])
                            & (per_vector <= percentiles[x[1]])
                        ]
                    )
                )
                data["median"].append(
                    np.median(
                        per_vector[
                            (per_vector > percentiles[x[0]])
                            & (per_vector <= percentiles[x[1]])
                        ]
                    )
                )
                data["var"].append(
                    np.var(
                        per_vector[
                            (per_vector > percentiles[x[0]])
                            & (per_vector <= percentiles[x[1]])
                        ]
                    )
                )
            else:
                data["n"].append(1)
                data["n_standarized"].append(0)
                data["avg"].append(0)
                data["std"].append(0)
                data["median"].append(0)
                data["var"].append(0)

        return data

    def df_to_percentiles(
        self,
        data: pd.DataFrame,
        group_col: str,
        values_col: str,
        sep_perc: int = 1,
        drop_outlires: bool = True,
    ):
        """
        Calculates summary statistics based on percentile ranges for each group in a DataFrame.

        This method groups the data by the specified `group_col`, calculates percentile ranges for each group's values in the `values_col`, and
        computes summary statistics (e.g., count, average, median, standard deviation, and variance) for each percentile range.
        Optionally, it can drop upper outliers from the data before performing the calculations.

        Args:
            data (pd.DataFrame) - the input DataFrame containing the data
            group_col (str) - the name of the column used for grouping the data
            values_col (str) - the name of the column containing the values for which percentiles are calculated.
            sep_perc (int) - the separation interval for percentiles (default is 1, meaning percentiles are calculated at every 1%)
            drop_outlires (bool) - whether to remove upper outliers from the data before performing calculations (default is True)

        Returns:
            full_data (dict) - a dictionary where each key is a group name (from `group_col`), and the value is another dictionary containing:
                - 'n' (list): The number of elements in each percentile range
                - 'n_standarized' (list): The proportion of elements in each percentile range relative to the total number of elements
                - 'avg' (list): The average value of elements within each percentile range
                - 'median' (list): The median value of elements within each percentile range
                - 'std' (list): The standard deviation of elements within each percentile range
                - 'var' (list): The variance of elements within each percentile range

        """

        full_data = {}

        if drop_outlires == True:
            data = self.drop_up_df(
                data=data, group_col=group_col, values_col=values_col
            )

        groups = set(data[group_col])

        percentiles, percentiles_loop = self.percentiles_calculation(
            data[values_col], sep_perc=sep_perc
        )

        for g in groups:

            print(f"Group: {g} ...")

            tmp_values = data[values_col][data[group_col] == g]

            per_dat = self.to_percentil(tmp_values, percentiles, percentiles_loop)

            full_data[g] = per_dat

        return full_data

    def round_to_scientific_notation(self, num):
        if num == 0:
            return "0.0"

        if abs(num) < 0.0001:
            rounded_num = np.format_float_scientific(num, precision=1, exp_digits=1)
            return rounded_num
        else:
            return f"{num:.1f}"

    def aov_percentiles(self, data, testes_col, comb: str = "*"):
        """
        Performs a Welch's ANOVA on percentile-based group data.

        This method calculates group values by combining the columns specified in `testes_col` according to the operation defined in `comb`.
        It then performs a Welch's ANOVA to test for differences in means between the groups. Welch's ANOVA is suitable when the groups have
        unequal variances.

        Parameters:
            data (dict of pd.DataFrame) - a dictionary where keys are group names and values are DataFrames containing the data.
            testes_col (str or list of str) - column name(s) from which the group values are derived. If a list is provided, columns will be
            combined based on the `comb` operation.
            comb (str) - the operation used to combine multiple columns if `testes_col` is a list. Options include:
                '*' (multiplication),
                '+' (addition),
                '**' (exponentiation),
                '-' (subtraction),
                '/' (division),
                Default is '*'

        Returns:
            F (float) - the F-statistic from Welch's ANOVA.
            p-val (float) - the uncorrected p-value from Welch's ANOVA, testing for significant differences between groups.

        Notes:
            - If `testes_col` is a single string, no combination is performed, and the group values are taken directly from that column.
            - Welch's ANOVA is used as it accounts for unequal variances between groups.
            - The `df.melt()` method is used to reshape the data, allowing the ANOVA to be applied to all groups.

        Example Usage:
            welch_F, welch_p = self.aov_percentiles(data, testes_col=['col1', 'col2'], comb='+')
            print(f"Welch's ANOVA F-statistic: {welch_F}, p-value: {welch_p}")


        """

        groups = []

        for d in data.keys():

            if isinstance(testes_col, str):
                g = data[d][testes_col]
            elif isinstance(testes_col, list):
                g = [1] * len(data[d][testes_col[0]])
                for t in testes_col:
                    if comb == "*":
                        g = [a * b for a, b in zip(g, data[d][t])]
                    elif comb == "+":
                        g = [a + b for a, b in zip(g, data[d][t])]
                    elif comb == "**":
                        g = [a**b for a, b in zip(g, data[d][t])]
                    elif comb == "-":
                        g = [a - b for a, b in zip(g, data[d][t])]
                    elif comb == "/":
                        g = [a / b for a, b in zip(g, data[d][t])]

            groups.append(g)

        df = pd.DataFrame({f"group_{i}": group for i, group in enumerate(groups)})

        df_melted = df.melt(var_name="group", value_name="value")

        welch_results = pg.welch_anova(data=df_melted, dv="value", between="group")

        return welch_results["F"].values[0], welch_results["p-unc"].values[0]

    def post_aov_percentiles(self, data, testes_col, comb: str = "*"):
        """
        Performs a Welch's ANOVA on percentile-based group data and pairwise comparisons Welch's T-test.

        Args:
            data (dict of pd.DataFrame) - dictionary where keys are group names and values are DataFrames containing the data
            testes_col (str or list of str) - column name(s) from which the group values are derived
                If a list is provided, columns will be combined based on the `comb` operation
            comb (str) - operation used to combine multiple columns if `testes_col` is a list. Options include:
                '*' (multiplication),
                '+' (addition),
                '**' (exponentiation),
                '-' (subtraction),
                '/' (division).
                Default is '*'

        Returns:
            p_val (float) - the uncorrected p-value from Welch's ANOVA
            final_results (dict) - results of pairwise t-tests with keys:
                'group1', 'group2', 'stat', 'p_val', 'adj_p_val'

        """

        p_val = self.aov_percentiles(data=data, testes_col=testes_col, comb=comb)[1]

        pairs = list(combinations(data, 2))
        final_results = {
            "group1": [],
            "group2": [],
            "stat": [],
            "p_val": [],
            "adj_p_val": [],
        }

        for group1, group2 in pairs:
            if isinstance(testes_col, str):
                g1 = data[group1][testes_col]
            elif isinstance(testes_col, list):
                g1 = [1] * len(data[group1][testes_col[0]])
                for t in testes_col:
                    if comb == "*":
                        g1 = [a * b for a, b in zip(g1, data[group1][t])]
                    elif comb == "+":
                        g1 = [a + b for a, b in zip(g1, data[group1][t])]
                    elif comb == "**":
                        g1 = [a**b for a, b in zip(g1, data[group1][t])]
                    elif comb == "-":
                        g1 = [a - b for a, b in zip(g1, data[group1][t])]
                    elif comb == "/":
                        g1 = [a / b for a, b in zip(g1, data[group1][t])]

            if isinstance(testes_col, str):
                g2 = data[group2][testes_col]
            elif isinstance(testes_col, list):
                g2 = [1] * len(data[group2][testes_col[0]])
                for t in testes_col:
                    if comb == "*":
                        g2 = [a * b for a, b in zip(g2, data[group2][t])]
                    elif comb == "+":
                        g2 = [a + b for a, b in zip(g2, data[group2][t])]
                    elif comb == "**":
                        g2 = [a**b for a, b in zip(g2, data[group2][t])]
                    elif comb == "-":
                        g2 = [a - b for a, b in zip(g2, data[group2][t])]
                    elif comb == "/":
                        g2 = [a / b for a, b in zip(g2, data[group2][t])]

            stat, p_val = stats.ttest_ind(
                g1, g2, alternative="two-sided", equal_var=False
            )
            g = sorted([group1, group2])
            final_results["group1"].append(g[0])
            final_results["group2"].append(g[1])
            final_results["stat"].append(stat)
            final_results["p_val"].append(p_val)
            adj = p_val * len(pairs)
            if adj > 1:
                final_results["adj_p_val"].append(1)
            else:
                final_results["adj_p_val"].append(adj)

        return p_val, final_results

    def chi2_percentiles(self, input_hist):
        """
        Performs a Chi-squared test on percentile-based group data.

        This method takes input histogram data, reformats it into a contingency table,
        and then performs a Chi-squared test to evaluate whether there is a significant
        association between the groups.

        Args:
            input_hist (dict of pd.DataFrame) - a dictionary where keys are group names and
                values are DataFrames containing histogram data.
                The histogram data should include a column 'n' that contains counts
                for each percentile/bin.

        Returns:
            chi2_statistic (float) - the test statistic from the Chi-squared test
            p_value (float) - the p-value from the Chi-squared test
            dof (int) - degrees of freedom for the test
            expected (np.ndarray) - the expected frequencies for each group/bin under the null hypothesis
            chi_data (dict) - the formatted data used in the Chi-squared test

        Example Usage:
            chi2_stat, p_val, dof, expected, chi_data = self.chi2_percentiles(input_hist)
            print(f"Chi-squared statistic: {chi2_stat}, p-value: {p_val}")

        """

        chi_data = {}

        for d in input_hist.keys():
            tmp_dic = {}

            for n, c in enumerate(input_hist[d]["n"]):
                tmp_dic[f"p{n+1}"] = c

            chi_data[d] = tmp_dic

        chi2_statistic, p_value, dof, expected = chi2_contingency(
            pd.DataFrame(chi_data).T, correction=True
        )

        return chi2_statistic, p_value, dof, expected, chi_data

    def post_ch2_percentiles(self, input_hist):
        """
        Performs a Chi-squared test on percentile-based group data, including pairwise comparisons.

        This method first performs a Chi-squared test on the input histogram data across all groups to
        check for a significant association. Then, it performs pairwise Chi-squared tests between
        groups to identify specific group differences. Multiple comparisons are corrected using
        the Bonferroni method.

        Args:
            input_hist (dict of pd.DataFrame) - a dictionary where keys are group names and
                values are DataFrames containing histogram data. The histogram data should include
                a column 'n' that contains counts for each percentile/bin

        Returns:
            p_val (float) - the overall p-value from the initial Chi-squared test across all groups
            final_results (dict) - a dictionary containing the results of pairwise Chi-squared tests with keys:
                - 'group1' (list): The name of the first group in each comparison
                - 'group2' (list): The name of the second group in each comparison
                - 'chi2' (list): The Chi-squared statistic for each pairwise comparison
                - 'p_val' (list): The p-value for each pairwise comparison
                - 'adj_p_val' (list): The adjusted p-value (Bonferroni correction) for multiple comparisons

        Example Usage:
            p_val, final_results = self.post_ch2_percentiles(input_hist)
            print(f"Overall Chi-squared p-value: {p_val}")
            for i in range(len(final_results['group1'])):
                print(f"Comparison: {final_results['group1'][i]} vs {final_results['group2'][i]}")
                print(f"Chi2 stat: {final_results['chi2'][i]}, p-value: {final_results['p_val'][i]}, adj. p-value: {final_results['adj_p_val'][i]}")

        """

        res = self.chi2_percentiles(input_hist)

        pairs = list(combinations(res[4], 2))
        results = []

        for group1, group2 in pairs:
            table_pair = pd.DataFrame(res[4])[[group1, group2]]
            chi2_stat, p_val, _, _ = chi2_contingency(table_pair, correction=True)
            results.append((group1, group2, chi2_stat, p_val))

        final_results = {
            "group1": [],
            "group2": [],
            "chi2": [],
            "p_val": [],
            "adj_p_val": [],
        }

        for group1, group2, chi2_stat, p_val in results:
            g = sorted([group1, group2])
            final_results["group1"].append(g[0])
            final_results["group2"].append(g[1])
            final_results["chi2"].append(chi2_stat)
            final_results["p_val"].append(p_val)
            adj = p_val * len(results)
            if adj > 1:
                final_results["adj_p_val"].append(1)
            else:
                final_results["adj_p_val"].append(adj)

        return res[1], final_results

    def hist_compare_plot(
        self, data, queue, tested_value, p_adj: bool = True, txt_size: int = 20
    ):
        """
        Generates comparative histograms and displays results of statistical tests (ANOVA and Chi-squared).

        This method performs transformations on the input data, generates comparative histograms for
        each group, and displays statistical test results, including Welch's ANOVA and Chi-squared tests.
        It includes options for multiple comparison corrections using the Bonferroni method.

        Args:
            data (dict of pd.DataFrame) - a dictionary where keys are group names and values are DataFrames
                containing histogram data. The data should include the column for the tested variable
            queue (list) - a list defining the order of groups to be plotted
            tested_value (str) - the column name in `data` representing the variable to test and visualize
            p_adj (bool) - if True, applies Bonferroni correction for multiple comparisons. Default: True
            txt_size (int) - font size for text annotations in the plot. Default: 20

        Returns:
            fig (matplotlib.figure.Figure) - a Matplotlib figure object containing the generated histograms
                and statistical test results

        Example Usage:
            fig = self.hist_compare_plot(data, queue=['group1', 'group2', 'group3'], tested_value='n', p_adj=True, txt_size=18)
            plt.show()

        """

        from scipy import stats

        for i in data.keys():
            values = np.array(data[i][tested_value])
            values += 1
            transformed_values, fitted_lambda = stats.boxcox(values)
            data[i][tested_value] = transformed_values.tolist()

        if sorted(queue) != sorted(data.keys()):
            print(
                "\n Wrong queue provided! The queue will be sorted with default settings!"
            )
            queue = sorted(data.keys())

        # parametric selected value
        pk, dfk = self.post_aov_percentiles(data, testes_col=tested_value)

        dfk = pd.DataFrame(dfk)

        dfk = dfk.sort_values(
            by=["group1", "group2"],
            key=lambda col: [queue.index(val) if val in queue else -1 for val in col],
        ).reset_index(drop=True)

        # parametric standarized selected value
        pkc, dfkc = self.post_aov_percentiles(
            data, testes_col=[tested_value, "n_standarized"], comb="*"
        )

        dfkc = pd.DataFrame(dfkc)

        dfkc = dfkc.sort_values(
            by=["group1", "group2"],
            key=lambda col: [queue.index(val) if val in queue else -1 for val in col],
        ).reset_index(drop=True)

        # chi2
        pchi, dfchi = self.post_ch2_percentiles(data)

        dfchi = pd.DataFrame(dfchi)

        dfchi = dfchi.sort_values(
            by=["group1", "group2"],
            key=lambda col: [queue.index(val) if val in queue else -1 for val in col],
        ).reset_index(drop=True)

        ##############################################################################

        standarized_max, standarized_min, value_max, value_min = [], [], [], []
        for d in queue:
            standarized_max.append(max(data[d]["n_standarized"]))
            standarized_min.append(min(data[d]["n_standarized"]))
            value_max.append(max(data[d][tested_value]))
            value_min.append(min(data[d][tested_value]))

        num_columns = len(queue) + 1

        fig, axs = plt.subplots(
            3,
            num_columns,
            figsize=(8 * num_columns, 10),
            gridspec_kw={"width_ratios": [1] * len(queue) + [0.5], "wspace": 0.05},
        )

        for i, d in enumerate(queue):
            tmp_data = data[d]

            axs[0, i].bar(
                [str(n) for n in range(len(tmp_data["n_standarized"]))],
                tmp_data["n_standarized"],
                width=0.95,
                color="gold",
            )
            axs[0, i].set_ylim(
                min(standarized_min) * 0.9995, max(standarized_max) * 1.0005
            )

            if i == 0:
                axs[0, i].set_ylabel("Standarized\nfrequency", fontsize=txt_size)
            else:
                axs[0, i].set_yticks([])

            axs[0, i].set_xticks([])
            axs[0, i].tick_params(axis="y", labelsize=txt_size * 0.7)

            axs[1, i].bar(
                [str(n) for n in range(len(tmp_data[tested_value]))],
                tmp_data[tested_value],
                width=0.95,
                color="orange",
            )

            mean_value = np.mean(tmp_data[tested_value])
            axs[1, i].axhline(y=mean_value, color="red", linestyle="--")

            # axs[1, i].set_ylim(min(value_min) - 1, max(value_max) + 1)
            axs[1, i].set_ylim(min(value_min) * 0.9995, max(value_max) * 1.0005)

            if i == 0:
                axs[1, i].set_ylabel(f"Normalized\n{tested_value}", fontsize=txt_size)
            else:
                axs[1, i].set_yticks([])

            axs[1, i].set_xticks([])
            axs[1, i].tick_params(axis="y", labelsize=txt_size * 0.7)

            axs[2, i].bar(
                [str(n) for n in range(len(tmp_data["n_standarized"]))],
                [
                    a * b
                    for a, b in zip(tmp_data[tested_value], tmp_data["n_standarized"])
                ],
                width=0.95,
                color="goldenrod",
            )

            mean_value = np.mean(
                [
                    a * b
                    for a, b in zip(tmp_data[tested_value], tmp_data["n_standarized"])
                ]
            )
            axs[2, i].axhline(y=mean_value, color="red", linestyle="--")

            axs[2, i].set_ylim(
                (min(standarized_min) * min(value_min)) * 0.9995,
                (max(standarized_max) * max(value_max) * 1.0005),
            )
            axs[2, i].set_xlabel(d, fontsize=txt_size)

            if i == 0:
                axs[2, i].set_ylabel(
                    f"Standarized\nnorm_{tested_value}", fontsize=txt_size
                )
            else:
                axs[2, i].set_yticks([])

            axs[2, i].set_xticks([])
            axs[2, i].tick_params(axis="y", labelsize=txt_size * 0.7)

        sign = "ns"
        if float(self.round_to_scientific_notation(pk)) < 0.001:
            sign = "***"
        elif float(self.round_to_scientific_notation(pk)) < 0.01:
            sign = "**"
        elif float(self.round_to_scientific_notation(pk)) < 0.05:
            sign = "*"

        text = f"Test Welch's ANOVA\np-value: {self.round_to_scientific_notation(pk)} - {sign}\n"

        if p_adj == True:
            for i in range(len(dfk["group1"])):
                sign = "ns"
                if dfk["adj_p_val"][i] < 0.001:
                    sign = "***"
                elif dfk["adj_p_val"][i] < 0.01:
                    sign = "**"
                elif dfk["adj_p_val"][i] < 0.05:
                    sign = "*"

                text += f"{dfk['group1'][i]} vs. {dfk['group2'][i]}\np-value: {self.round_to_scientific_notation(dfk['adj_p_val'][i])} - {sign}\n"
        else:
            for i in range(len(dfk["group1"])):
                sign = "ns"
                if dfk["p_val"][i] < 0.001:
                    sign = "***"
                elif dfk["p_val"][i] < 0.01:
                    sign = "**"
                elif dfk["p_val"][i] < 0.05:
                    sign = "*"

                text += f"{dfk['group1'][i]} vs. {dfk['group2'][i]}\np-value: {self.round_to_scientific_notation(dfk['p_val'][i])} - {sign}\n"

        axs[1, -1].text(
            0.5, 0.5, text, ha="center", va="center", fontsize=txt_size * 0.7, wrap=True
        )
        axs[1, -1].set_axis_off()

        sign = "ns"
        if float(self.round_to_scientific_notation(pkc)) < 0.001:
            sign = "***"
        elif float(self.round_to_scientific_notation(pkc)) < 0.01:
            sign = "**"
        elif float(self.round_to_scientific_notation(pkc)) < 0.05:
            sign = "*"

        text = f"Test Welch's ANOVA\np-value: {self.round_to_scientific_notation(pkc)} - {sign}\n"

        if p_adj == True:
            for i in range(len(dfkc["group1"])):
                sign = "ns"
                if dfkc["adj_p_val"][i] < 0.001:
                    sign = "***"
                elif dfkc["adj_p_val"][i] < 0.01:
                    sign = "**"
                elif dfkc["adj_p_val"][i] < 0.05:
                    sign = "*"

                text += f"{dfkc['group1'][i]} vs. {dfkc['group2'][i]}\np-value: {self.round_to_scientific_notation(dfkc['adj_p_val'][i])} - {sign}\n"
        else:
            for i in range(len(dfkc["group1"])):
                sign = "ns"
                if dfkc["p_val"][i] < 0.001:
                    sign = "***"
                elif dfkc["p_val"][i] < 0.01:
                    sign = "**"
                elif dfkc["p_val"][i] < 0.05:
                    sign = "*"

                text += f"{dfkc['group1'][i]} vs. {dfkc['group2'][i]}\np-value: {self.round_to_scientific_notation(dfkc['p_val'][i])} - {sign}\n"

        axs[2, -1].text(
            0.5, 0.5, text, ha="center", va="center", fontsize=txt_size * 0.7, wrap=True
        )
        axs[2, -1].set_axis_off()

        sign = "ns"
        if float(self.round_to_scientific_notation(pchi)) < 0.001:
            sign = "***"
        elif float(self.round_to_scientific_notation(pchi)) < 0.01:
            sign = "**"
        elif float(self.round_to_scientific_notation(pchi)) < 0.05:
            sign = "*"

        text = f"Test Chi-squared\np-value: {self.round_to_scientific_notation(pchi)} - {sign}\n"

        if p_adj == True:
            for i in range(len(dfchi["group1"])):
                sign = "ns"
                if dfchi["adj_p_val"][i] < 0.001:
                    sign = "***"
                elif dfchi["adj_p_val"][i] < 0.01:
                    sign = "**"
                elif dfchi["adj_p_val"][i] < 0.05:
                    sign = "*"

                text += f"{dfchi['group1'][i]} vs. {dfchi['group2'][i]}\np-value: {self.round_to_scientific_notation(dfchi['adj_p_val'][i])} - {sign}\n"
        else:
            for i in range(len(dfchi["group1"])):
                sign = "ns"
                if dfchi["p_val"][i] < 0.001:
                    sign = "***"
                elif dfchi["p_val"][i] < 0.01:
                    sign = "**"
                elif dfchi["p_val"][i] < 0.05:
                    sign = "*"

                text += f"{dfchi['group1'][i]} vs. {dfchi['group2'][i]}\np-value: {self.round_to_scientific_notation(dfchi['p_val'][i])} - {sign}\n"

        axs[0, -1].text(
            0.5, 0.5, text, ha="center", va="center", fontsize=txt_size * 0.7, wrap=True
        )
        axs[0, -1].set_axis_off()

        plt.tight_layout()

        if cfg._DISPLAY_MODE:
            plt.show()

        return fig
