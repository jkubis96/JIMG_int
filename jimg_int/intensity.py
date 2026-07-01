import json
import os
import random
import re
from itertools import combinations, permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from scipy.stats import wasserstein_distance
from tqdm import tqdm

import jimg_int.config as cfg

from .utils import *

random.seed(42)


class FeatureIntensity(ImageTools):
    r"""
    Class for quantitative analysis of pixel intensity and size measurements
    in 2D/3D biological images. Supports projection of 3D stacks, mask-based
    intensity normalization, region size estimation and metadata extraction.

    Parameters
    ----------
    input_image : ndarray, optional
        Input image or 3D stack for analysis. If 3D, projection must be applied.

    image : ndarray, optional
        2D projected image (internal use).

    normalized_image_values : dict, optional
        Dictionary storing normalized intensity statistics.

    mask : ndarray, optional
        Binary mask of region of interest (ROI).

    background_mask : ndarray, optional
        Binary mask used for background estimation. If not provided, `mask` is used.

    typ : {"avg", "median", "std", "var", "max", "min"}, optional
        Projection type for 3D stacks. Default is `"avg"`.

    size_info : dict, optional
        Dictionary storing ROI size measurements.

    correction_factor : float, optional
        Normalization correction factor applied to background intensity.
        Must satisfy 0 < factor < 1. Default is 0.1.

    img_type : str, optional
        Image type metadata.

    scale : float, optional
        Pixel resolution in physical units (e.g. µm/px). Used in size calculations.

    stack_selection : list of int, optional
        List of Z-indices to remove when projecting a 3D image.

    Attributes
    ----------
    input_image : ndarray or None
        Loaded input image.

    image : ndarray or None
        Projected 2D image.

    mask : ndarray or None
        Region of interest mask.

    background_mask : ndarray or None
        Background normalization mask.

    scale : float or None
        Scale value for size estimation.

    normalized_image_values : dict or None
        Dictionary containing intensity metrics.

    size_info : dict or None
        Dictionary with ROI size information.

    typ : str
        Selected projection type for 3D images.

    stack_selection : list of int
        Z-levels excluded from projection.

    Notes
    -----
    The intensity normalization formula applied per pixel is:

    .. math::

        R_{i,j} = T_{i,j} - \\left( \\mu_B (1 + c) \\right)

    where
    * ``T_{i,j}`` – pixel intensity in ROI
    * ``μ_B`` – mean intensity in background region
    * ``c`` – correction factor
    * ``R_{i,j}`` – normalized pixel intensity

    Examples
    --------
    Load a 3D image, mask and compute statistics:

    >>> fi = FeatureIntensity()
    >>> fi.load_image_3D("stack.tiff")
    >>> fi.load_mask_("mask.png")
    >>> fi.set_projection("median")
    >>> fi.run_calculations()
    >>> results = fi.get_results()
    >>> results["intensity"]["norm_mean"]
    """

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
        """
        Initialize a FeatureIntensity analysis instance.

        Parameters
        ----------
        input_image : ndarray, optional
            Input image or 3D stack used for analysis. If the image is 3D, a
            projection will be computed depending on the `typ` parameter.

        image : ndarray, optional
            2D image buffer used internally after projection of the input image.
            Should not be set manually.

        normalized_image_values : dict, optional
            Dictionary containing normalized intensity statistics. Usually filled
            automatically after running `run_calculations()`.

        mask : ndarray, optional
            Binary mask of the target region of interest (ROI). Required for
            intensity and size calculations.

        background_mask : ndarray, optional
            Binary mask specifying the background region used to compute the
            normalization threshold. If not provided, the ROI mask is also used
            as the background reference.

        typ : {"avg", "median", "std", "var", "max", "min"}, optional
            Projection method for 3D images. Determines how the z-stack is
            collapsed into a 2D image. Default is `"avg"`.

        size_info : dict, optional
            Dictionary storing computed size metrics of the ROI. Populated after
            invoking `size_calculations()`.

        correction_factor : float, optional
            Correction term used during intensity normalization. Must satisfy
            0 < correction_factor < 1. Default is 0.1.

        img_type : str, optional
            Optional metadata about the image type (e.g., "tiff", "png").

        scale : float, optional
            Pixel resolution in physical units (e.g., µm/px). Required for
            real-size estimation in `size_calculations()`.

        stack_selection : list of int, optional
            Indices of z-planes to exclude during projection of a 3D stack.

        Notes
        -----
        Values not provided are initialized to `None`, except for `typ`, which
        defaults to `"avg"`, and `correction_factor`, which defaults to 0.1.

        The class is designed to be populated by loading functions:
        `load_image_()`, `load_image_3D()`, `load_mask_()`,
        and optionally `load_normalization_mask_()` and `load_JIMG_project_()`.
        """

        self.input_image = input_image or None
        """ Input image or 3D stack used for analysis. If the image is 3D, a
         projection will be computed depending on the `typ` parameter."""

        self.image = image or None
        """  2D image buffer used internally after projection of the input image.
          Should not be set manually."""

        self.normalized_image_values = normalized_image_values or None
        """Dictionary containing normalized intensity statistics. Usually filled
        automatically after running `run_calculations()`."""

        self.mask = mask or None
        """Binary mask of the target region of interest (ROI). Required for
        intensity and size calculations."""

        self.background_mask = background_mask or None
        """ Binary mask specifying the background region used to compute the
         normalization threshold. If not provided, the ROI mask is also used
         as the background reference."""

        self.typ = typ or "avg"
        """Projection method for 3D images. Determines how the z-stack is
        collapsed into a 2D image. Default is `"avg"`."""

        self.size_info = size_info or None
        """Dictionary storing computed size metrics of the ROI. Populated after
        invoking `size_calculations()`."""

        self.correction_factor = correction_factor or 0.1
        """ Correction term used during intensity normalization. Must satisfy
         0 < correction_factor < 1. Default is 0.1."""

        self.scale = scale or None
        """ Pixel resolution in physical units (e.g., µm/px). Required for
         real-size estimation in `size_calculations()`."""

        self.stack_selection = stack_selection or []
        """Indices of z-planes to exclude during projection of a 3D stack."""

    @property
    def current_metadata(self):
        r"""
        Return current metadata parameters used in image processing and normalization.

        Returns
        -------
        tuple
            A tuple containing:

            projection_type : str
                Projection method used for 3D image reduction (e.g., "avg", "median").

            correction_factor : float
                Correction factor used for background subtraction during intensity
                normalization. The applied formula is:

                .. math::

                    R_{i,j} = T_{i,j} - ( \mu_B (1 + c) )

                where
                * ``R_{i,j}`` — normalized pixel intensity
                * ``T_{i,j}`` — original pixel intensity
                * ``μ_B`` — mean background intensity
                * ``c`` — correction factor
            scale : float or None
                Pixel resolution (unit/px), loaded via `load_JIMG_project_()` or set manually
                using `set_scale()`.

            stack_selection : list of int
                Indices of z-slices excluded from projection of a 3D image.

        Notes
        -----
        This property also prints the metadata values to the console for quick inspection.
        """

        print(f"Projection type: {self.typ}")
        print(f"Correction factor: {self.correction_factor}")
        print(f"Scale (unit/px): {self.scale}")
        print(f"Selected stac to remove: {self.stack_selection}")

        return self.typ, self.correction_factor, self.scale, self.stack_selection

    def set_projection(self, projection: str):
        """
        Set the projection method for 3D image stack reduction.

        Parameters
        ----------
        projection : {"avg", "median", "std", "var", "max", "min"}
            Projection method to reduce a 3D image stack to a 2D image. Default is `"avg"`.

        Notes
        -----
        This method updates the `typ` attribute of the class. The selected projection
        determines how the z-stack is collapsed:
        - `"avg"` : average intensity across slices
        - `"median"` : median intensity across slices
        - `"std"` : standard deviation across slices
        - `"var"` : variance across slices
        - `"max"` : maximum intensity across slices
        - `"min"` : minimum intensity across slices
        """

        t = ["avg", "median", "std", "var", "max", "min"]
        if projection in t:
            self.typ = projection
        else:
            print(f"\nProvided parameter is incorrect. Avaiable projection types: {t}")

    def set_correction_factorn(self, factor: float):
        r"""
        Set the correction factor for background subtraction during intensity normalization.

        Parameters
        ----------
        factor : float
            Correction factor to adjust background subtraction. Must satisfy 0 < factor < 1.
            Default is 0.1.

        Notes
        -----
        The correction is applied per pixel in the target mask using the formula:

        .. math::

            R_{i,j} = T_{i,j} - ( \mu_B (1 + c) )

        where
        * ``R_{i,j}`` — normalized pixel intensity
        * ``T_{i,j}`` — original pixel intensity
        * ``μ_B`` — mean intensity in the background mask
        * ``c`` — correction factor
        """

        if factor < 1 and factor > 0:
            self.correction_factor = factor
        else:
            print(
                "\nProvided parameter is incorrect. The factor should be a floating-point value within the range of 0 to 1."
            )

    def set_scale(self, scale):
        """
        Set the scale for converting pixel measurements to physical units.

        Parameters
        ----------
        scale : float
            Pixel resolution in physical units (e.g., µm/px). Used to calculate the
            actual size of the tissue or organ.

        Notes
        -----
        The scale can also be automatically loaded from a JIMG project using
        `load_JIMG_project_()`. This value is required for size calculations in
        `size_calculations()`.
        """

        self.scale = scale

    def set_selection_list(self, rm_list: list):
        """
        Set the list of z-slices to exclude when projecting a 3D image stack.

        Parameters
        ----------
        rm_list : list of int
            List of indices corresponding to z-slices that should be removed from
            the full 3D image stack before projection.

        Notes
        -----
        This updates the `stack_selection` attribute, which is used by the
        `stack_selection_()` method during projection.
        """

        self.stack_selection = rm_list

    def load_JIMG_project_(self, path):
        """
        Load a JIMG project from a `.pjm` file.

        Parameters
        ----------
        file_path : str
            Path to the JIMG project file. The file must have a `.pjm` extension.

        Returns
        -------
        project : object
            Loaded project object containing images and metadata.

        Raises
        ------
        ValueError
            If the provided file path does not point to a `.pjm` file.

        Notes
        -----
        The method attempts to automatically set the `scale` and `stack_selection`
        attributes from the project metadata if available.
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
        """
        Remove selected z-slices from a 3D image stack based on `stack_selection`.

        Notes
        -----
        Only works if `input_image` is a 3D ndarray. The slices with indices listed
        in `stack_selection` are excluded from the stack. Updates `input_image`
        in-place.

        Prints a warning if `stack_selection` is empty.
        """

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
        """
        Project a 3D image stack into a 2D image using the method defined by `typ`.

        Notes
        -----
        Updates the `image` attribute with the projected 2D result.

        Supported projection types (`typ`):
        - "avg" : mean intensity across slices
        - "median" : median intensity across slices
        - "std" : standard deviation across slices
        - "var" : variance across slices
        - "max" : maximum intensity across slices
        - "min" : minimum intensity across slices

        Raises
        ------
        AttributeError
            If `input_image` is not defined.
        """

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
        """
        Detect whether the input image is 2D or 3D and perform appropriate preprocessing.

        Notes
        -----
        - For 3D images, applies `stack_selection_()` and then `projection()`.
        - For 2D images, no projection is applied.
        - Prints status messages indicating the type of image and applied operations.

        Raises
        ------
        AttributeError
            If `input_image` is not defined.
        """
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
        Load a 3D image stack from a TIFF file.

        Parameters
        ----------
        path : str
            Path to the 3D image file (*.tiff) to be loaded.

        Notes
        -----
        The loaded image is stored in the `input_image` attribute as a 3D ndarray.
        """
        path = os.path.abspath(path)

        self.input_image = self.load_3D_tiff(path)

    def load_image_(self, path):
        """
        Load a 2D image into the class.

        Parameters
        ----------
        path : str
            Path to the image file to be loaded.

        Notes
        -----
        The loaded image is stored in the `input_image` attribute as a 2D ndarray.
        """
        path = os.path.abspath(path)

        self.input_image = self.load_image(path)

    def load_mask_(self, path):
        r"""
        Load a binary mask into the class and optionally set it as the normalization mask.

        Parameters
        ----------
        path : str
            Path to the mask image file. Supported formats include 8-bit or 16-bit images
            with extensions such as `.png` or `.jpeg`. The mask must be binary
            (e.g., 0/255, 0/2**16-1, 0/1).

        Notes
        -----
        - If `load_normalization_mask_()` is not called, this mask is also used as the
          background mask for intensity normalization.
        - Normalization is applied per pixel using the formula:

          .. math::

              R_{i,j} = T_{i,j} - ( \mu_B (1 + c) )

          where
          * ``R_{i,j}`` — normalized pixel intensity
          * ``T_{i,j}`` — pixel intensity in the target mask
          * ``μ_B`` — mean intensity of the background (reversed mask)
          * ``c`` — correction factor
        """

        path = os.path.abspath(path)

        self.mask = self.load_mask(path)

        print(
            "\nThis mask was also set as the reverse background mask. If you want a different background mask for normalization, use 'load_normalization_mask()'."
        )
        self.background_mask = self.load_mask(path)

    def load_normalization_mask_(self, path):
        r"""
        Load a binary mask for normalization into the class.

        Parameters
        ----------
        path : str
            Path to the mask image file. Supported formats include 8-bit or 16-bit
            images (e.g., `.png`, `.jpeg`). The mask must be binary (0/255, 0/2**16-1, 0/1).

        Notes
        -----
        - The mask defines the area of interest. Normalization is applied to the inverse
          of this area (reversed mask).
        - Normalization formula applied per pixel:

          .. math::

              R_{i,j} = T_{i,j} - ( \mu_B (1 + c) )

          where
          * ``R_{i,j}`` — normalized pixel intensity
          * ``T_{i,j}`` — pixel intensity in the target mask
          * ``μ_B`` — mean intensity of the background (reversed mask)
          * ``c`` — correction factor
        """

        path = os.path.abspath(path)

        self.background_mask = self.load_mask(path)

    def intensity_calculations(self):
        """
        Calculate normalized and raw intensity statistics from the image based on masks.

        This method performs intensity calculations using the main mask (`self.mask`)
        and the background mask (`self.background_mask`). The pixel intensities within
        the mask of interest are normalized by subtracting a threshold derived from the
        background region and applying a correction factor (`self.correction_factor`).
        Negative values after normalization are clipped to zero.

        The following statistics are computed for both normalized and raw values:
        - Minimum
        - Maximum
        - Mean
        - Median
        - Standard deviation
        - Variance
        - List of all normalized values (only for normalized data)

        Notes
        -----
        - The method updates the instance attribute `self.normalized_image_values`
          with a dictionary containing both normalized and raw statistics.
        - Normalization formula applied for each pixel in the selected mask:
            final_val = selected_value - (threshold + threshold * correction_factor)
          where threshold is the mean intensity in the background mask.
        - Negative values after normalization are set to zero.
        """

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
        """
        Calculates the size and bounding dimensions of the masked region in the image.

        This method computes the following metrics based on the current mask:
            - Total number of pixels in the mask (`px_size`)
            - Real-world size if a scale is provided (`size`)
            - Maximum lengths along x and y axes (`max_length_x_axis`, `max_length_y_axis`)

        If `self.scale` is defined (unit per pixel), the real-world size is calculated.
        If not, `size` will be `None` and a warning message is printed.

        Returns:
            Updates the following attributes in the class:
                - self.size_info (dict) containing:
                    - 'size' (float or None): real-world size of the mask
                    - 'px_size' (int): number of pixels in the masked region
                    - 'max_length_x_axis' (int): length of the bounding box along the x-axis
                    - 'max_length_y_axis' (int): length of the bounding box along the y-axis

        Example:
            analysis.size_calculations()
            print(analysis.size_info)
        """

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
        Run the full analysis pipeline on the loaded image using the provided masks.

        Notes
        -----
        - The input image must be loaded via `load_image_()` or `load_image_3D()`.
        - The ROI mask must be loaded via `load_mask_()`. Optionally, a normalization
          mask can be loaded via `load_normalization_mask_()`.
        - Parameters such as projection type and correction factor can be set with
          `set_projection()` and `set_correction_factor()`.
        - Scale and stack selection can also influence calculations if defined.
        - To view current parameters, use the `current_metadata` property.

        Returns
        -------
        None
            The results are stored internally and can be retrieved using
            `get_results()`.
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
        Return the results from the analysis performed by `run_calculations()`.

        Returns
        -------
        results_dict : dict or None
            Dictionary containing intensity and size results. Structure:
            - 'intensity' : dict with normalized and raw intensity statistics
            - 'size' : dict with ROI size metrics

        Notes
        -----
        If analysis has not been run yet, prints a message and returns None.
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
        Save the analysis results to a `.int` (JSON) file.

        Parameters
        ----------
        path : str, optional
            Directory path where the file will be saved. Defaults to the current working directory.

        mask_region : str
            Name or identifier of the mask region (e.g., tissue, part of tissue).

        feature_name : str
            Name of the feature being analyzed. Underscores or spaces are replaced with periods.

        individual_number : int
            Unique identifier for the individual in the analysis (e.g., 1, 2, 3).

        individual_name : str
            Name of the individual (e.g., species name, tissue, organoid).

        Notes
        -----
        - The method validates that all required parameters are provided and that
          analysis results exist (`normalized_image_values` and `size_info`).
        - Creates the directory if it does not exist.
        - File name format:
          '<individual_name>_<individual_number>_<mask_region>_<feature_name>.int'

        Raises
        ------
        FileNotFoundError
            If the specified path cannot be created or accessed.

        ValueError
            If any of `mask_region`, `feature_name`, `individual_number`, or
            `individual_name` are missing or invalid.
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
        Concatenate intensity data from multiple `.int` files and save as CSV.

        Parameters
        ----------
        directory : str, optional
            Path to the directory containing `.int` files. Defaults to the current working directory.

        name : str
            Prefix for the output CSV file names. CSV files are saved in the format
            '<name>_<gene>_<region>.csv'.

        Raises
        ------
        FileNotFoundError
            If the directory cannot be accessed or no `.int` files are found.

        ValueError
            If an `.int` file is missing expected data or has an incorrect format.

        Notes
        -----
        - The method groups intensity data by gene (feature) and mask region.
        - Outputs one CSV file per unique gene-region combination, saved in the specified directory.
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
    """
    Class for performing percentile-based statistical analysis on grouped data.

    This class provides methods to calculate percentiles, remove outliers, aggregate
    data into percentile bins, perform Welch's ANOVA, Kolmogorov-Smirnov and Fisher's exact tests,
    evaluate Wasserstein distances, calculate Fold Change, and visualize results via comparative
    histograms. It is designed to handle both single-column and multi-column combinations
    of values for group-based analysis.

    Methods
    -------
    drop_up_df(data, group_col, values_col)
        Removes upper outliers from a DataFrame based on a grouping column.

    percentiles_calculation(values, sep_perc=1)
        Calculates percentiles and creates loopable percentile ranges.

    to_percentil(values, percentiles, percentiles_loop, values_col, replication_col)
        Aggregates statistics based on percentile ranges.

    df_to_percentiles(data, group_col, values_col, replication_col, sep_perc=1, drop_outlires=True)
        Computes percentile statistics for grouped DataFrame data.

    round_to_scientific_notation(num)
        Formats a number in scientific notation or standard format.

    aov(data, testes_col, comb="*")
        Performs Welch's ANOVA on percentile-based group data.

    post_aov(data, testes_col, comb="*")
        Performs Welch's ANOVA with pairwise t-tests.

    ks_percentiles(input_hist)
        Perform pairwise Kolmogorov-Smirnov (KS) tests on percentile data across all groups.

    fisher_percentiles(input_hist)
        Perform pairwise Fisher's exact tests on percentile data across all groups.

    to_wasserstein_distance(data)
        Calculates scaled pairwise Wasserstein distances for grouped distributions.

    to_fold_change(data, tested_value)
        Calculates the Fold Change (FC) between all directed permutations of groups.

    get_stats(data, tested_value)
        Calculates and aggregates overall statistical metrics (ANOVA, Fisher's exact, Kolmogorov-Smirnov, FC, Wasserstein distance).

    hist_compare_plot(data, queue=None, p_adj=True, txt_size=20)
        Generates comparative histograms with statistical test results and metrics.
    """

    def drop_up_df(self, data: pd.DataFrame, group_col: str, values_col: str):
        """
        Remove upper outliers from a DataFrame based on a specified value column, grouped by a grouping column.

        Outliers are calculated and removed separately for each group defined by `group_col`.
        The upper outliers are defined using the interquartile range (IQR) method:
            values greater than Q3 + 1.5 * IQR are considered outliers.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the data.

        group_col : str
            The name of the column used for grouping the data.

        values_col : str
            The column containing the values from which upper outliers will be removed.

        Returns
        -------
        filtered_data : pd.DataFrame
            A filtered DataFrame with the upper outliers removed for each group.

        Notes
        -----
        - Outliers are removed separately within each group.
        - The original DataFrame is not modified; a new filtered DataFrame is returned.
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
        Calculate percentiles for a set of values and generate consecutive percentile ranges.

        This function computes percentiles from 0 to 100 at intervals defined by `sep_perc`.
        It also generates a list of consecutive percentile ranges that can be used for further analysis or binning.

        Parameters
        ----------
        values : array-like
            The input data values for which the percentiles are calculated.

        sep_perc : int, optional
            Separation interval between percentiles (default is 1, meaning percentiles are calculated every 1%).

        Returns
        -------
        percentiles : np.ndarray
            Array of calculated percentile values.

        percentiles_loop : list of tuple
            List of consecutive percentile ranges as tuples, e.g., [(0, 1), (1, 2), ..., (99, 100)].

        Notes
        -----
        - The first percentile is set to 0 to avoid issues with zero values.
        - `percentiles_loop` is useful for iterating through percentile ranges when aggregating statistics.
        """

        per_vector = values.copy()

        percentiles = np.percentile(per_vector, np.arange(0, 101, sep_perc))
        percentiles[0] = 0

        percentiles_loop = [(i, i + 1) for i in range(int(100 / sep_perc))]

        return percentiles, percentiles_loop

    def to_percentil(
        self, values, percentiles, percentiles_loop, values_col, replication_col
    ):
        """
        Aggregate statistics for a set of values based on percentile ranges, including replications.

        This function calculates summary statistics (count, proportion, mean, median,
        standard deviation, variance) for each percentile range defined in `percentiles_loop`.
        It computes these statistics both for the combined data ('mutual') and separately
        for each individual replication. It also calculates overall metrics per replication.

        Parameters
        ----------
        values : pd.DataFrame
            Input DataFrame containing the data to be analyzed.
        percentiles : np.ndarray
            Array of percentile values used to define the boundaries of each range.
        percentiles_loop : list of tuple
            List of consecutive percentile ranges as index tuples, e.g., [(0, 1), (1, 2), ..., (99, 100)].
        values_col : str
            The column name in `values` containing the numeric data to aggregate.
        replication_col : str
            The column name in `values` used to identify distinct replications or samples.

        Returns
        -------
        full_data : dict
            A nested dictionary containing the calculated statistics with the following structure:
            - 'percentiles' : dict
                - 'mutual' : dict
                    Lists of statistics ('n', 'n_standarized', 'avg', 'median', 'std', 'var')
                    aggregated across all replications for each percentile bin.
                - 'replications' : dict
                    Keys are replication names. Values are dictionaries of statistics (same as above)
                    calculated specifically for that replication within each bin.
            - 'values' : dict
                Lists of overall statistics ('avg', 'median', 'std', 'var', 'replication')
                calculated for each replication as a whole (ignoring bins).

        Notes
        -----
        - If a percentile range contains no elements, statistics are set to 0 and count is set to 1 to avoid empty lists.
        """

        full_data = {}
        per_vector = values[values_col]
        amount = len(per_vector)

        data_mutual = {
            "n": [],
            "n_standarized": [],
            "avg": [],
            "median": [],
            "std": [],
            "var": [],
        }

        for x in percentiles_loop:
            subset = per_vector[
                (per_vector > percentiles[x[0]]) & (per_vector <= percentiles[x[1]])
            ]
            n_subset = len(subset)

            if n_subset > 0:
                data_mutual["n"].append(n_subset)
                data_mutual["n_standarized"].append(n_subset / amount)
                data_mutual["avg"].append(np.mean(subset))
                data_mutual["median"].append(np.median(subset))
                data_mutual["std"].append(np.std(subset))
                data_mutual["var"].append(np.var(subset))
            else:
                data_mutual["n"].append(0)
                data_mutual["n_standarized"].append(0)
                data_mutual["avg"].append(0)
                data_mutual["median"].append(0)
                data_mutual["std"].append(0)
                data_mutual["var"].append(0)

        full_data["percentiles"] = {"mutual": data_mutual, "replications": {}}

        unique_names = set(values[replication_col])

        for nam in unique_names:
            per_vector_rep = values[values_col][values[replication_col] == nam]

            data_rep = {
                "n": [],
                "n_standarized": [],
                "avg": [],
                "median": [],
                "std": [],
                "var": [],
            }

            for x in percentiles_loop:
                subset = per_vector_rep[
                    (per_vector_rep > percentiles[x[0]])
                    & (per_vector_rep <= percentiles[x[1]])
                ]
                n_subset = len(subset)

                if n_subset > 0:
                    data_rep["n"].append(n_subset)
                    data_rep["n_standarized"].append(n_subset / amount)
                    data_rep["avg"].append(np.mean(subset))
                    data_rep["median"].append(np.median(subset))
                    data_rep["std"].append(np.std(subset))
                    data_rep["var"].append(np.var(subset))
                else:
                    data_rep["n"].append(0)
                    data_rep["n_standarized"].append(0)
                    data_rep["avg"].append(0)
                    data_rep["median"].append(0)
                    data_rep["std"].append(0)
                    data_rep["var"].append(0)

            full_data["percentiles"]["replications"][nam] = data_rep

        unique_names = set(values[replication_col])

        data_rep = {"avg": [], "median": [], "std": [], "var": [], "replication": []}

        for nam in unique_names:
            per_vector_rep = values[values_col][values[replication_col] == nam]

            data_rep["avg"].append(np.mean(per_vector_rep))
            data_rep["median"].append(np.median(per_vector_rep))
            data_rep["std"].append(np.std(per_vector_rep))
            data_rep["var"].append(np.var(per_vector_rep))
            data_rep["replication"].append(nam)

        full_data["values"] = data_rep

        return full_data

    def df_to_percentiles(
        self,
        data: pd.DataFrame,
        group_col: str = "individual_name",
        values_col: str = "norm_intensity",
        replication_col: str = "individual_number",
        sep_perc: int = 1,
        drop_outlires: bool = True,
    ):
        """
        Calculate summary statistics based on percentile ranges for each group in a DataFrame.

        This method groups the input DataFrame by `group_col`, computes global percentile ranges
        based on `values_col`, and then aggregates statistics for each percentile bin. The aggregation
        is performed both mutually for the group and individually per replication. Optionally,
        upper outliers can be removed before the calculations.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame containing the grouped data.
        group_col : str, optional
            Column name used to define groups (default is 'individual_name').
        values_col : str, optional
            Column name containing the numeric values for percentile calculations
            (default is 'norm_intensity').
        replication_col : str, optional
            Column name used to identify separate replications within the groups
            (default is 'individual_number').
        sep_perc : int, optional
            Separation interval for percentiles (default is 1, meaning 1% steps).
        drop_outlires : bool, optional
            If True, removes upper outliers from the data using the IQR method before
            performing calculations (default is True).

        Returns
        -------
        full_data : dict
            A dictionary where each key is a unique group name (from `group_col`).
            The corresponding value is the nested dictionary returned by `to_percentil()`,
            which includes bin-wise statistics ('mutual' and 'replications') and overall
            metrics ('values').

        Notes
        -----
        - Outlier removal uses the IQR method within each group if `drop_outlires` is True.
        """

        full_data = {}

        if drop_outlires == True:
            data = self.drop_up_df(
                data=data, group_col=group_col, values_col=values_col
            )

        groups = set(data[group_col])
        val_dat = [x for x in data[values_col] if x > 0]

        percentiles, percentiles_loop = self.percentiles_calculation(
            val_dat, sep_perc=sep_perc
        )

        for g in groups:

            print(f"Group: {g} ...")

            tmp_values = data[data[group_col] == g]

            per_dat = self.to_percentil(
                tmp_values, percentiles, percentiles_loop, values_col, replication_col
            )

            full_data[g] = per_dat

        return full_data

    def round_to_scientific_notation(self, num):
        """
        Round a number to scientific notation if very small, otherwise to one decimal place.

        Parameters
        ----------
        num : float
            The number to round.

        Returns
        -------
        str
            The rounded number as a string.
            - If `num` is 0, returns "0.0".
            - If `abs(num) < 1e-4`, returns scientific notation with 1 decimal and 1-digit exponent.
            - Otherwise, returns the number rounded to one decimal place.
        """

        if num == 0:
            return "0.0"

        if abs(num) < 0.0001:
            rounded_num = np.format_float_scientific(num, precision=1, exp_digits=1)
            return rounded_num
        else:
            return f"{num:.1f}"

    def aov(self, data, testes_col, comb: str = "*"):
        """
        Perform a Welch's ANOVA analysis.

        This function calculates group values by aggregating specified columns (testes_col)
        via the comb method and then conducts a Welch's ANOVA. This approach is ideal for
        comparing group means when data exhibits unequal variances across groups.

        Parameters
        ----------
        data : dict of pd.DataFrame
            Dictionary where keys are group names and values are DataFrames containing the data.

        testes_col : str or list of str
            Column name(s) from which the group values are derived. If a list is provided, columns
            will be combined based on the `comb` operation.

        comb : str, optional
            Operation used to combine multiple columns if `testes_col` is a list. Options include:
                '*' : multiplication
                '+' : addition
                '**': exponentiation
                '-' : subtraction
                '/' : division
            Default is '*'.

        Returns
        -------
        F : float
            F-statistic from Welch's ANOVA.

        p_val : float
            Uncorrected p-value from Welch's ANOVA, testing for significant differences between groups.

        Notes
        -----
        - If `testes_col` is a single string, no combination is performed, and the group values
          are taken directly from that column.
        - Welch's ANOVA is used as it accounts for unequal variances between groups.
        - The `df.melt()` method is used to reshape the data, allowing the ANOVA to be applied to all groups.

        Examples
        --------
        >>> welch_F, welch_p = self.aov(data, testes_col=['col1', 'col2'], comb='+')
        >>> print(f"Welch's ANOVA F-statistic: {welch_F}, p-value: {welch_p}")
        """

        groups = []

        for d in data.keys():

            if isinstance(testes_col, str):
                g = data[d]["values"][testes_col]
            elif isinstance(testes_col, list):
                g = [1] * len(data[d]["values"][testes_col[0]])
                for t in testes_col:
                    if comb == "*":
                        g = [a * b for a, b in zip(g, data[d]["values"][t])]
                    elif comb == "+":
                        g = [a + b for a, b in zip(g, data[d]["values"][t])]
                    elif comb == "**":
                        g = [a**b for a, b in zip(g, data[d]["values"][t])]
                    elif comb == "-":
                        g = [a - b for a, b in zip(g, data[d]["values"][t])]
                    elif comb == "/":
                        g = [a / b for a, b in zip(g, data[d]["values"][t])]

            groups.append(g)

        df = pd.DataFrame({f"group_{i}": group for i, group in enumerate(groups)})

        df_melted = df.melt(var_name="group", value_name="value")

        welch_results = pg.welch_anova(data=df_melted, dv="value", between="group")

        return welch_results["F"].values[0], welch_results["p-unc"].values[0]

    def post_aov(self, data, testes_col, comb: str = "*"):
        """
        Perform Welch's ANOVA and pairwise Welch's t-tests on grouped data.

        This method first conducts a Welch's ANOVA to detect significant differences
        in group means. It then performs pairwise Welch's t-tests across all group
        combinations to identify specific differences. All p-values are adjusted using
        the Bonferroni correction to account for multiple comparisons.

        Parameters
        ----------
        data : dict of pd.DataFrame
            Dictionary where keys are group names and values are DataFrames containing the data.

        testes_col : str or list of str
            Column name(s) from which the group values are derived. If a list is provided,
            columns will be combined according to the `comb` operation.

        comb : str, optional
            Operation used to combine multiple columns if `testes_col` is a list. Options include:
                '*' : multiplication
                '+' : addition
                '**': exponentiation
                '-' : subtraction
                '/' : division
            Default is '*'.

        Returns
        -------
        p_val : float
            Uncorrected p-value from the Welch's ANOVA.

        final_results : dict
            Dictionary containing results of pairwise Welch's t-tests with keys:
                'group1' : list of first group names in each comparison
                'group2' : list of second group names in each comparison
                'stat' : list of t-statistics for each comparison
                'p_val' : list of uncorrected p-values for each comparison
                'adj_p_val' : list of Bonferroni-adjusted p-values for multiple comparisons
        """

        p_val = self.aov(data=data, testes_col=testes_col, comb=comb)[1]

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
                g1 = data[group1]["values"][testes_col]
            elif isinstance(testes_col, list):
                g1 = [1] * len(data[group1]["values"][testes_col[0]])
                for t in testes_col:
                    if comb == "*":
                        g1 = [a * b for a, b in zip(g1, data[group1]["values"][t])]
                    elif comb == "+":
                        g1 = [a + b for a, b in zip(g1, data[group1]["values"][t])]
                    elif comb == "**":
                        g1 = [a**b for a, b in zip(g1, data[group1]["values"][t])]
                    elif comb == "-":
                        g1 = [a - b for a, b in zip(g1, data[group1]["values"][t])]
                    elif comb == "/":
                        g1 = [a / b for a, b in zip(g1, data[group1]["values"][t])]

            if isinstance(testes_col, str):
                g2 = data[group2]["values"][testes_col]
            elif isinstance(testes_col, list):
                g2 = [1] * len(data[group2]["values"][testes_col[0]])
                for t in testes_col:
                    if comb == "*":
                        g2 = [a * b for a, b in zip(g2, data[group2]["values"][t])]
                    elif comb == "+":
                        g2 = [a + b for a, b in zip(g2, data[group2]["values"][t])]
                    elif comb == "**":
                        g2 = [a**b for a, b in zip(g2, data[group2]["values"][t])]
                    elif comb == "-":
                        g2 = [a - b for a, b in zip(g2, data[group2]["values"][t])]
                    elif comb == "/":
                        g2 = [a / b for a, b in zip(g2, data[group2]["values"][t])]

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

    def ks_percentiles(self, input_hist):
        """
        Perform pairwise Kolmogorov-Smirnov (KS) tests on percentile data across all groups.

        This method extracts the percentile levels and computes the average value for
        each percentile to obtain a lower-dimensional representation of the data, thereby
        reducing the Big Data scale problem for each group. Using these metrics, it reconstructs
        the underlying empirical distributions to evaluate both structural proportions and scale.

        To further mitigate the large sample size problem ("curse of Big Data") where inflating
        pixel counts yields artificially significant results, a controlled downsampling (resampling)
        is applied to standardize the sample sizes across groups.

        A two-sample Kolmogorov-Smirnov (KS) test is then performed for every possible pair
        of groups to detect differences in distribution shapes. Finally, p-values are adjusted
        using the Bonferroni correction method to account for multiple comparisons and control
        the family-wise error rate.

        Parameters
        ----------
        input_hist : dict
            A nested dictionary where keys are group names. Each group must contain
            the following structure: input_hist[group]["percentiles"]["mutual"]["n"],
            which holds an iterable (e.g., list or Series) of counts per percentile/bin.

        Returns
        -------
        final_results : dict
            A dictionary containing the results of the pairwise comparisons with keys:
            - 'group1': list of the first group names in the pairs.
            - 'group2': list of the second group names in the pairs.
            - 'K-S': list of Kolmogorov-Smirnov test statistics.
            - 'p_val': list of unadjusted p-values.
            - 'adj_p_val': list of Bonferroni-adjusted p-values (capped at 1.0).

        Example
        -------
        >>> results = self.ks_percentiles(input_hist)
        """

        ks_data = {}

        for d in input_hist.keys():
            tmp_dic = {}

            for n, c in enumerate(input_hist[d]["percentiles"]["mutual"]["avg"]):
                tmp_dic[f"p{n+1}"] = c

            ks_data[d] = tmp_dic

        df_cleaned = pd.DataFrame(ks_data).T

        pairs = list(combinations(df_cleaned.index, 2))

        final_results = {
            "group1": [],
            "group2": [],
            "K-S": [],
            "p_val": [],
            "adj_p_val": [],
        }

        for group1, group2 in pairs:

            g = sorted([group1, group2])

            table_pair = pd.DataFrame(df_cleaned).T[[group1, group2]].copy()

            res = stats.ks_2samp(table_pair.iloc[:, 0], table_pair.iloc[:, 1])

            final_results["group1"].append(g[0])
            final_results["group2"].append(g[1])
            final_results["K-S"].append(res.statistic)
            final_results["p_val"].append(res.pvalue)
            adj = res.pvalue * len(pairs)
            if adj > 1:
                final_results["adj_p_val"].append(1)
            else:
                final_results["adj_p_val"].append(adj)

        return final_results

    def fisher_percentiles(self, input_hist):
        """
        Perform pairwise Fisher's exact tests on percentile data across all groups.

        This method extracts the raw counts (N) for each percentile bin across all
        groups to construct a contingency table representation of the data. By utilizing
        the discrete frequency counts per bin rather than continuous average values, it
        evaluates both structural distribution proportions and sample size scaling
        differences simultaneously.

        An exact testing approach is applied to every unique pair of groups by extracting
        their corresponding sub-tables. For each pair, a Fisher's exact test (or its
        extension for larger contingency tables) is performed to detect statistically
        significant deviations in distribution profiles.

        Finally, p-values are manually adjusted using the Bonferroni correction method
        by multiplying the raw p-values by the total number of comparisons to control
        the family-wise error rate across multiple pair-wise tests.
        the family-wise error rate.

        Parameters
        ----------
        input_hist : dict
            A nested dictionary where keys are group names. Each group must contain
            the following structure: input_hist[group]["percentiles"]["mutual"]["n"],
            which holds an iterable (e.g., list or Series) of counts per percentile/bin.

        Returns
        -------
        final_results : dict
            A dictionary containing the results of the pairwise comparisons with keys:
            - 'group1': list of the first group names in the pairs.
            - 'group2': list of the second group names in the pairs.
            - 'fish': list of Fisher's exact test statistics.
            - 'p_val': list of unadjusted p-values.
            - 'adj_p_val': list of Bonferroni-adjusted p-values (capped at 1.0).

        Example
        -------
        >>> results = self.fisher_percentiles(input_hist)
        """

        fish_data = {}

        for d in input_hist.keys():
            tmp_dic = {}

            for n, c in enumerate(input_hist[d]["percentiles"]["mutual"]["n"]):
                tmp_dic[f"p{n+1}"] = c

            fish_data[d] = tmp_dic

        df_cleaned = pd.DataFrame(fish_data).T

        pairs = list(combinations(df_cleaned.index, 2))

        final_results = {
            "group1": [],
            "group2": [],
            "fish": [],
            "p_val": [],
            "adj_p_val": [],
        }

        for group1, group2 in pairs:

            g = sorted([group1, group2])

            table_pair = pd.DataFrame(df_cleaned).T[[group1, group2]].copy()

            res = stats.fisher_exact(table_pair)

            final_results["group1"].append(g[0])
            final_results["group2"].append(g[1])
            final_results["fish"].append(res.statistic)
            final_results["p_val"].append(res.pvalue)
            adj = res.pvalue * len(pairs)
            if adj > 1:
                final_results["adj_p_val"].append(1)
            else:
                final_results["adj_p_val"].append(adj)

        return final_results

    def to_wasserstein_distance(self, data):
        """
        Calculate scaled pairwise Wasserstein distances for grouped distributions.

        This method computes the 1D Wasserstein distance (Earth Mover's Distance)
        between all possible combinations of groups in the provided dataset.
        Before calculating the distance, the standardized frequencies are scaled
        by a factor representing the average total count (sample size) of the
        two compared groups.

        Parameters
        ----------
        data : dict
            A nested dictionary where keys are group names. For each group, the
            method expects the following internal data structure:
            - `data[group_name]['percentiles']['mutual']['n']` : list-like
                Absolute counts or sample sizes for the distribution.
            - `data[group_name]['percentiles']['mutual']['n_standarized']` : list-like
                Standardized frequencies or probabilities to be compared.

        Returns
        -------
        final_results : dict
            A dictionary containing the results of the pairwise distance calculations:
            - 'group1' : list of str
                The name of the first group in the comparison.
            - 'group2' : list of str
                The name of the second group in the comparison.
            - 'wasserstein_distance' : list of float
                The computed scaled Wasserstein distance for each pair.
        """

        pairs = list(combinations(data.keys(), 2))

        final_results = {"group1": [], "group2": [], "wasserstein_distance": []}

        for group1, group2 in pairs:

            factor = (
                sum(data[group1]["percentiles"]["mutual"]["n"])
                + sum(data[group2]["percentiles"]["mutual"]["n"])
            ) / 2

            dist = wasserstein_distance(
                [
                    x * factor
                    for x in data[group1]["percentiles"]["mutual"]["n_standarized"]
                ],
                [
                    x * factor
                    for x in data[group2]["percentiles"]["mutual"]["n_standarized"]
                ],
            )

            g = sorted([group1, group2])
            final_results["group1"].append(g[0])
            final_results["group2"].append(g[1])
            final_results["wasserstein_distance"].append(dist)

        return final_results

    def to_fold_change(self, data, tested_value):
        """
        Calculate the Fold Change (FC) between all permutations of groups.

        This method computes the ratio of the mean values of a specified feature
        (`tested_value`) for every directed pair of groups. Because permutations
        are used, the calculation is directional (i.e., both Group A / Group B
        and Group B / Group A are computed).

        Parameters
        ----------
        data : dict
            A nested dictionary where keys are group names. For each group, the
            method expects the following internal structure:
            - `data[group_name]['values'][tested_value]` : array-like
                Numeric values used to compute the mean for the group.

        tested_value : str
            The specific key or column name within the 'values' dictionary
            indicating which feature's fold change should be calculated.

        Returns
        -------
        final_results : dict
            A dictionary containing the results of the pairwise fold change calculations:
            - 'group1' : list of str
                The name of the numerator group in the comparison.
            - 'group2' : list of str
                The name of the denominator group in the comparison.
            - 'FC' : list of float
                The calculated fold change (mean of group1 / mean of group2).
        """

        pairs = list(permutations(data.keys(), 2))

        final_results = {"group1": [], "group2": [], "FC": []}

        values = []
        for group1, group2 in pairs:

            values = values + data[group1]["values"][tested_value]
            values = values + data[group2]["values"][tested_value]

        values_min = min([x for x in values if x > 0])
        values_min = values_min / 2

        for group1, group2 in pairs:

            g1 = np.mean(data[group1]["values"][tested_value])
            g2 = np.mean(data[group2]["values"][tested_value])

            if g1 == 0:
                g1 = g1 + values_min

            if g2 == 0:
                g2 = g2 + values_min

            fc = g1 / g2

            final_results["group1"].append(group1)
            final_results["group2"].append(group2)
            final_results["FC"].append(fc)

        return final_results

    def get_stats(self, data, tested_value):
        """
        Calculate and aggregate statistical metrics (ANOVA, Fisher's Exact,
        Kolmogorov-Smirnov, Fold Change, Wasserstein distance).

        This method computes overall statistics and pairwise comparisons for grouped data.
        To properly capture both structural proportions and total count variations across
        percentiles while avoiding the curse of Big Data, it runs two distinct tests:
        1. Fisher's exact test on discrete percentile counts to evaluate absolute scale
           and profile differences.
        2. Two-sample Kolmogorov-Smirnov (KS) test on reconstructed empirical
           distributions to evaluate discrepancies in distribution shapes.

        Additionally, it calculates the Fold Change (FC) and evaluates Wasserstein
        distances. If the average number of replicates per group is at least 3,
        it conducts Welch's ANOVA. The input dictionary is modified in-place to
        include a new 'statistics' key containing all results.

        Parameters
        ----------
        data : dict
            A nested dictionary where keys are group names. Each group's dictionary
            must contain the structure `['values']['replication']` to verify sample sizes,
            along with the necessary data structures required by downstream statistical methods.

        tested_value : str
            The key or column name representing the specific variable to evaluate
            (e.g., used for ANOVA and Fold Change calculations).

        Returns
        -------
        data : dict
            The original input dictionary, extended with a new `data['statistics']` key
            that houses the computed statistical results, including `percintiles_fish`
            and `percintiles_ks`.

        Example
        -------
        stats = self.get_stats(
            data,
            tested_value='n',
        )
        """

        # parametric selected value
        sum_k = 0
        n = 0
        for k in data.keys():
            if k != "statistics":
                n += 1
                sum_k += len(data[k]["values"]["replication"])

        sum_k = sum_k / n

        if sum_k >= 3:
            pk, dfk = self.post_aov(data, testes_col=tested_value)

        # fish
        fish = self.fisher_percentiles(data)

        # K_S
        ks = self.ks_percentiles(data)

        dw = self.to_wasserstein_distance(data)

        fc = self.to_fold_change(data, tested_value)

        data["statistics"] = {}

        data["statistics"]["percintiles_fish"] = fish

        data["statistics"]["percintiles_ks"] = ks

        if sum_k >= 3:
            data["statistics"]["ANOVA"] = {}

            data["statistics"]["ANOVA"]["p_value"] = pk
            data["statistics"]["ANOVA"]["pair-comparison"] = dfk
        else:
            import warnings

            warnings.warn(
                f"Insufficient replicates for statistical analysis. "
                f"At least 3 replicates per group (3 vs 3) are required. "
                f"The average number of samples per probe in this dataset was {n}.",
                RuntimeWarning,
            )

        data["statistics"]["FC"] = fc

        data["statistics"]["wasserstein_distance"] = dw

        data["statistics"]["tested_value"] = tested_value

        return data

    def hist_compare_plot(
        self, data, queue=None, p_adj: bool = True, txt_size: int = 20
    ):
        """
        Generate comparative histograms and display results of statistical tests
        (ANOVA / T-test [if min 3 vs. 3 comparison available], Kolmogorov-Smirnov, Fisher percentiles)
        and statistics (FC, Wasserstein distance).


        Parameters
        ----------
        data : dict
            Dictionary where keys are group names and values are containing histogram data.
            Each DataFrame should include the column specified by `tested_value`.

        queue : list of str or None
            Defines the order of groups to be plotted.

        p_adj : bool, optional
            If True, applies Bonferroni correction for multiple comparisons (default is True).

        txt_size : int, optional
            Font size for text annotations in the plot (default is 20).

        Returns
        -------
        fig : matplotlib.figure.Figure
            Matplotlib figure object containing the generated histograms and statistical test results.

        Example
        -------
        fig = self.hist_compare_plot(
            data,
            queue=['group1', 'group2', 'group3'],
            p_adj=True,
            txt_size=18
        )
        plt.show()
        """

        if queue is None:
            queue = [x for x in sorted(data.keys()) if x != "statistics"]

        if sorted(queue) != [x for x in sorted(data.keys()) if x != "statistics"]:
            print(
                "\n Wrong queue provided! The queue will be sorted with default settings!"
            )
            queue = [x for x in sorted(data.keys()) if x != "statistics"]

        # parametric selected value
        tested_value = data["statistics"]["tested_value"]

        ##############################################################################

        standarized_max, standarized_min, value_max, value_min = [], [], [], []
        for d in queue:
            standarized_max.append(
                max(data[d]["percentiles"]["mutual"]["n_standarized"])
            )
            standarized_min.append(
                min(data[d]["percentiles"]["mutual"]["n_standarized"])
            )
            value_max.append(max(data[d]["percentiles"]["mutual"][tested_value]))
            value_min.append(min(data[d]["percentiles"]["mutual"][tested_value]))

        num_columns = len(queue) + 1

        fig, axs = plt.subplots(
            3,
            num_columns,
            figsize=(8 * num_columns, 10),
            gridspec_kw={"width_ratios": [1] * len(queue) + [0.5], "wspace": 0.05},
        )

        for i, d in enumerate(queue):
            tmp_data = data[d]["percentiles"]["mutual"]

            axs[0, i].bar(
                [str(n) for n in range(len(tmp_data["n_standarized"]))],
                tmp_data["n_standarized"],
                width=0.95,
                color="gold",
            )

            # line
            n_groups = len(data[d]["percentiles"]["replications"].keys())
            colors = plt.cm.OrRd(np.linspace(0.3, 0.9, n_groups))

            for ix, dn in enumerate(data[d]["percentiles"]["replications"].keys()):

                color = colors[ix]

                y = data[d]["percentiles"]["replications"][dn]["n_standarized"]
                x = np.arange(len(y))

                axs[0, i].plot(x, y, color="black", linewidth=1.5, label="_nolegend_")

                axs[0, i].plot(
                    x,
                    y,
                    color=color,
                    linewidth=1,
                    marker="o",
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

            # line

            for ix, dn in enumerate(data[d]["percentiles"]["replications"].keys()):

                color = colors[ix]

                y = data[d]["percentiles"]["replications"][dn][tested_value]
                x = np.arange(len(y))

                axs[1, i].plot(x, y, color="black", linewidth=1.5, label="_nolegend_")

                axs[1, i].plot(
                    x,
                    y,
                    color=color,
                    linewidth=1,
                    marker="o",
                )

            mean_value = np.mean(data[d]["values"][tested_value])
            axs[1, i].axhline(y=mean_value, color="red", linestyle="--")

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

            # line
            for ix, dn in enumerate(data[d]["percentiles"]["replications"].keys()):

                color = colors[ix]

                y = [
                    a * b
                    for a, b in zip(
                        data[d]["percentiles"]["replications"][dn][tested_value],
                        data[d]["percentiles"]["replications"][dn]["n_standarized"],
                    )
                ]
                x = np.arange(len(y))

                axs[2, i].plot(x, y, color="black", linewidth=1.5, label="_nolegend_")

                axs[2, i].plot(
                    x,
                    y,
                    color=color,
                    linewidth=1,
                    marker="o",
                )

            mean_value = np.mean(
                data[d]["values"][data["statistics"]["tested_value"]]
            ) * np.mean(tmp_data["n_standarized"])

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

        # statistics

        # ANOVA / t-test

        if "ANOVA" in data["statistics"].keys():
            pk = data["statistics"]["ANOVA"]["p_value"]
            dfk = data["statistics"]["ANOVA"]["pair-comparison"]
            dfk = pd.DataFrame(dfk)

            dfk = dfk.sort_values(
                by=["group1", "group2"],
                key=lambda col: [
                    queue.index(val) if val in queue else -1 for val in col
                ],
            ).reset_index(drop=True)

            sign = "ns"
            if float(self.round_to_scientific_notation(pk)) < 0.001:
                sign = "***"
            elif float(self.round_to_scientific_notation(pk)) < 0.01:
                sign = "**"
            elif float(self.round_to_scientific_notation(pk)) < 0.05:
                sign = "*"

            text = f"Test Welch's ANOVA\non '{tested_value}' values\np-value: {self.round_to_scientific_notation(pk)} - {sign}\n"

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

            axs[2, -1].text(
                0.5,
                0.5,
                text,
                ha="center",
                va="center",
                fontsize=txt_size * 0.7,
                wrap=True,
            )
            axs[2, -1].set_axis_off()
        else:
            axs[2, -1].set_axis_off()

        # FC / Distance

        ranking_FC = pd.DataFrame(data["statistics"]["FC"])

        ranking_dw = pd.DataFrame(data["statistics"]["wasserstein_distance"])

        ranking_combined = pd.merge(
            ranking_FC, ranking_dw, on=["group1", "group2"], how="right"
        )

        ranking_combined = ranking_combined.sort_values(
            by=["group1", "group2"],
            key=lambda col: [queue.index(val) if val in queue else -1 for val in col],
        ).reset_index(drop=True)

        text = "FC / Wasserstein distance\n"
        for i in range(len(ranking_combined)):
            group1 = ranking_combined["group1"][i]
            group2 = ranking_combined["group2"][i]
            fc_val = ranking_combined["FC"][i]
            wasserstein_val = ranking_combined["wasserstein_distance"][i]

            text += f"{group1} vs. {group2}\n {fc_val:.2f} | {wasserstein_val:.2f}\n"

        axs[1, -1].text(
            0.5, 0.5, text, ha="center", va="center", fontsize=txt_size * 0.7, wrap=True
        )
        axs[1, -1].set_axis_off()

        # fish

        fish = pd.DataFrame(data["statistics"]["percintiles_fish"])

        # K-S

        ks = pd.DataFrame(data["statistics"]["percintiles_ks"])

        fish = fish.sort_values(
            by=["group1", "group2"],
            key=lambda col: [queue.index(val) if val in queue else -1 for val in col],
        ).reset_index(drop=True)

        ks = ks.sort_values(
            by=["group1", "group2"],
            key=lambda col: [queue.index(val) if val in queue else -1 for val in col],
        ).reset_index(drop=True)

        text = f"Tests\nFisher's exact / Kolmogorov-Smirnov\n"

        if p_adj == True:
            for i in range(len(fish["group1"])):
                sign1 = "ns"
                if fish["adj_p_val"][i] < 0.001:
                    sign1 = "***"
                elif fish["adj_p_val"][i] < 0.01:
                    sign1 = "**"
                elif fish["adj_p_val"][i] < 0.05:
                    sign1 = "*"

                sign2 = "ns"
                if ks["adj_p_val"][i] < 0.001:
                    sign2 = "***"
                elif ks["adj_p_val"][i] < 0.01:
                    sign2 = "**"
                elif ks["adj_p_val"][i] < 0.05:
                    sign2 = "*"

                text += f"{fish['group1'][i]} vs. {fish['group2'][i]}\np-value: {sign1} / {sign2}\n"

        else:
            for i in range(len(fish["group1"])):
                sign1 = "ns"
                if fish["p_val"][i] < 0.001:
                    sign1 = "***"
                elif fish["p_val"][i] < 0.01:
                    sign1 = "**"
                elif fish["p_val"][i] < 0.05:
                    sign1 = "*"

                sign2 = "ns"
                if ks["p_val"][i] < 0.001:
                    sign2 = "***"
                elif ks["p_val"][i] < 0.01:
                    sign2 = "**"
                elif ks["p_val"][i] < 0.05:
                    sign2 = "*"

                text += f"{fish['group1'][i]} vs. {fish['group2'][i]}\np-value: {sign1} / {sign2}\n"

        axs[0, -1].text(
            0.5, 0.5, text, ha="center", va="center", fontsize=txt_size * 0.7, wrap=True
        )
        axs[0, -1].set_axis_off()

        plt.tight_layout()

        if cfg._DISPLAY_MODE:
            plt.show()

        return fig
