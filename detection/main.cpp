/*
   Copyright DMLAB at Georgia State University

   This file is part of An Extended Noise-Aware PIL Dataset creation project.

   PIL dataset creation is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   PIL dataset creation is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with [PROJECT NAME]. If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * Full-disk active region / PIL detection for Surya-bench HMI magnetograms.
 *
 * Adapted from SuryaBench ar_segmentation (see LICENSE-upstream).
 * Changes from upstream:
 *   - CPU only: cv::cuda calls replaced with CPU equivalents.
 *   - Disk centre and radius measured per frame from the data. Header
 *     geometry is not used (converted Surya FITS carry placeholder headers).
 *   - Configurable limb cut (limb_mode / limb_value in config.yaml).
 *   - overlapUnion and the size filter rewritten to avoid allocating one
 *     full-size Mat per component.
 *   - Final rasters are cut again in case a component straddles the circle.
 *
 * limb_mode:
 *   none         no cut
 *   formula      r = R_sun + 900 / 0.6 px (upstream formula; larger than the frame)
 *   radius_px    r = limb_value
 *   radius_frac  r = limb_value * R_sun
 *   angle        r = R_sun * sin(limb_value deg)
 *
 * Build with -DFDARP_DEBUG=ON for per-step output.
 */

#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <limits>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include "fitsio.h"
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <dirent.h>
#include <H5Cpp.h>

#include "io.hpp"
#include "utils.hpp"
#include <yaml-cpp/yaml.h>

#ifdef FDARP_DEBUG
#define DBG(msg) std::cout << "[DEBUG] " << msg << std::endl << std::flush
#else
#define DBG(msg) do { } while (0)
#endif

struct Config
{
    double pos_gauss;
    double neg_gauss;
    double low_threshold;
    double high_threshold;
    int dilation_size;
    int gap_size;
    int size_threshold;
    int pil_threshold;
    double strength_threshold;

    // --- limb cutting ---
    std::string limb_mode;      // none | formula | radius_px | radius_frac | angle
    double limb_value;          // meaning depends on limb_mode
    double limb_rsun_px;        // fallback radius when the measurement fails
    double disk_threshold;      // |B| above which a pixel counts as on-disk
    bool measure_disk;          // measure the disk per frame

    std::string data_dir;
    std::string output_dir;
    std::string log_dir;
};

template <typename T>
static T yamlGet(const YAML::Node &node, const std::string &key, const T &fallback)
{
    if (node[key])
    {
        try { return node[key].as<T>(); }
        catch (...) { }
    }
    return fallback;
}

Config loadConfig(const std::string &config_path)
{
    Config config;
    YAML::Node root = YAML::LoadFile(config_path);
    YAML::Node p = root["parameters"];

    config.pos_gauss = p["pos_gauss"].as<double>();
    config.neg_gauss = p["neg_gauss"].as<double>();
    config.low_threshold = p["low_threshold"].as<double>();
    config.high_threshold = p["high_threshold"].as<double>();
    config.dilation_size = p["dilation_size"].as<int>();
    config.gap_size = p["gap_size"].as<int>();
    config.size_threshold = p["size_threshold"].as<int>();
    config.strength_threshold = p["strength_threshold"].as<double>();
    config.pil_threshold = p["pil_threshold"].as<double>();

    config.limb_mode      = yamlGet<std::string>(p, "limb_mode", std::string("radius_frac"));
    config.limb_value     = yamlGet<double>(p, "limb_value", 1.0);
    config.limb_rsun_px   = yamlGet<double>(p, "limb_rsun_px", 1625.4);
    config.disk_threshold = yamlGet<double>(p, "disk_threshold", 1.0);
    config.measure_disk   = yamlGet<bool>(p, "measure_disk", true);

    config.data_dir = root["directories"]["data_dir"].as<std::string>();
    config.output_dir = root["directories"]["output_dir"].as<std::string>();
    config.log_dir = root["directories"]["log_dir"].as<std::string>();
    return config;
}

std::unordered_map<std::string, std::string> configToParameterMap(const Config &config)
{
    std::unordered_map<std::string, std::string> parameters;
    parameters["pos_gauss"] = std::to_string(config.pos_gauss);
    parameters["neg_gauss"] = std::to_string(config.neg_gauss);
    parameters["low_threshold"] = std::to_string(config.low_threshold);
    parameters["high_threshold"] = std::to_string(config.high_threshold);
    parameters["dilation_size"] = std::to_string(config.dilation_size);
    parameters["gap_size"] = std::to_string(config.gap_size);
    parameters["size_threshold"] = std::to_string(config.size_threshold);
    parameters["strength_threshold"] = std::to_string(config.strength_threshold);
    parameters["pil_threshold"] = std::to_string(config.pil_threshold);
    parameters["limb_mode"] = config.limb_mode;
    parameters["limb_value"] = std::to_string(config.limb_value);
    parameters["limb_rsun_px"] = std::to_string(config.limb_rsun_px);
    parameters["disk_threshold"] = std::to_string(config.disk_threshold);
    parameters["measure_disk"] = config.measure_disk ? "true" : "false";
    return parameters;
}

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Disk geometry, measured from the data
// ---------------------------------------------------------------------------

struct DiskGeometry
{
    double cx = 0.0;
    double cy = 0.0;
    double r = 0.0;
    bool ok = false;
};

// Disk centre/radius from the non-zero extent along the central row and
// column. A bounding-box estimate is computed as a cross-check.
DiskGeometry measureDisk(const cv::Mat &mag, double threshold, double tol = 5.0)
{
    DiskGeometry g;
    if (mag.empty())
        return g;

    const int rows = mag.rows;
    const int cols = mag.cols;
    const int midRow = rows / 2;
    const int midCol = cols / 2;

    // --- estimator 1: central row and column ---
    int x0 = -1, x1 = -1, y0 = -1, y1 = -1;
    for (int x = 0; x < cols; ++x)
    {
        if (std::abs(mag.at<double>(midRow, x)) > threshold)
        {
            if (x0 < 0) x0 = x;
            x1 = x;
        }
    }
    for (int y = 0; y < rows; ++y)
    {
        if (std::abs(mag.at<double>(y, midCol)) > threshold)
        {
            if (y0 < 0) y0 = y;
            y1 = y;
        }
    }

    if (x0 < 0 || y0 < 0)
    {
        DBG("  disk: no on-disk pixels along the central row/column");
        return g;
    }

    g.cx = (x0 + x1) / 2.0;
    g.cy = (y0 + y1) / 2.0;
    g.r = ((x1 - x0) + (y1 - y0)) / 4.0;
    g.ok = true;

    // --- estimator 2: bounding box of all valid pixels ---
    int bx0 = cols, bx1 = -1, by0 = rows, by1 = -1;
    for (int y = 0; y < rows; ++y)
    {
        const double *row = mag.ptr<double>(y);
        for (int x = 0; x < cols; ++x)
        {
            if (std::abs(row[x]) > threshold)
            {
                if (x < bx0) bx0 = x;
                if (x > bx1) bx1 = x;
                if (y < by0) by0 = y;
                if (y > by1) by1 = y;
            }
        }
    }

    if (bx1 >= 0 && by1 >= 0)
    {
        double bcx = (bx0 + bx1) / 2.0;
        double bcy = (by0 + by1) / 2.0;
        double br = ((bx1 - bx0) + (by1 - by0)) / 4.0;
        double diff = std::max({std::abs(g.cx - bcx), std::abs(g.cy - bcy), std::abs(g.r - br)});
        DBG("  disk: scan (" << g.cx << ", " << g.cy << ") R=" << g.r
            << " | bbox (" << bcx << ", " << bcy << ") R=" << br);
        if (diff > tol)
            std::cerr << "WARNING: disk estimators disagree by " << diff
                      << " px" << std::endl;
    }

    return g;
}

// ---------------------------------------------------------------------------

class pos_neg_detection
{
public:
    struct posneg_map
    {
        cv::Mat pos_map, neg_map;
    } posneg_struct;

    void identify_pos_neg_region(cv::Mat fits_image, int nRows, int nColumns, const Config &config)
    {
        cv::Mat pos_map, neg_map;
        fits_image.copyTo(pos_map);
        fits_image.copyTo(neg_map);

        cv::Mat pos_mask = (fits_image < config.pos_gauss);
        cv::Mat neg_mask = (fits_image > config.neg_gauss);

        pos_map.setTo(0, pos_mask);
        cv::Mat negatedMask_pos;
        cv::bitwise_not(pos_mask, negatedMask_pos);
        pos_map.setTo(1, negatedMask_pos);
        cv::Mat pos_8bit;
        pos_map.convertTo(pos_8bit, CV_8U);

        neg_map.setTo(0, neg_mask);
        cv::Mat negatedMask_neg;
        cv::bitwise_not(neg_mask, negatedMask_neg);
        neg_map.setTo(1, negatedMask_neg);
        cv::Mat neg_8bit;
        neg_map.convertTo(neg_8bit, CV_8U);

        this->posneg_struct.neg_map = neg_8bit;
        this->posneg_struct.pos_map = pos_8bit;
    }

    cv::Mat edge_detection(cv::Mat input, const Config &config)
    {
        cv::Mat output, input_8bit;
        if (input.type() != CV_8UC1)
            input.convertTo(input_8bit, CV_8U);
        else
            input_8bit = input;
        cv::Canny(input_8bit, output, config.low_threshold, config.high_threshold);
        return output;
    }

    cv::Mat buff_edge(cv::Mat edges, const Config &config)
    {
        cv::Mat dilated_edges;
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(config.dilation_size, config.dilation_size));
        cv::dilate(edges, dilated_edges, kernel);
        cv::dilate(dilated_edges, dilated_edges, kernel);
        return dilated_edges;
    }

    cv::Mat PIL_extraction(const cv::Mat &pos_dil_edge, const cv::Mat &neg_dil_edge)
    {
        cv::Mat a, b, intersection;
        if (pos_dil_edge.type() != CV_8UC1) pos_dil_edge.convertTo(a, CV_8U); else a = pos_dil_edge;
        if (neg_dil_edge.type() != CV_8UC1) neg_dil_edge.convertTo(b, CV_8U); else b = neg_dil_edge;
        cv::bitwise_and(a, b, intersection);
        return intersection;
    }

    cv::Mat overlapUnion(const cv::Mat &pos_dil, const cv::Mat &neg_dil)
    {
        cv::Mat a, b;
        if (pos_dil.type() != CV_8UC1) pos_dil.convertTo(a, CV_8U); else a = pos_dil;
        if (neg_dil.type() != CV_8UC1) neg_dil.convertTo(b, CV_8U); else b = neg_dil;

        cv::Mat union_mask, overlap_mask;
        cv::bitwise_or(a, b, union_mask);
        cv::bitwise_and(a, b, overlap_mask);

        DBG("  union non-zero: " << cv::countNonZero(union_mask));
        DBG("  overlap non-zero: " << cv::countNonZero(overlap_mask));

        cv::Mat labels, stats, centroids;
        int ncomp = cv::connectedComponentsWithStats(union_mask, labels, stats, centroids, 8, CV_32S);
        DBG("  components: " << ncomp - 1);

        std::vector<unsigned char> keep(ncomp, 0);
        for (int y = 0; y < labels.rows; ++y)
        {
            const int *lrow = labels.ptr<int>(y);
            const unsigned char *orow = overlap_mask.ptr<unsigned char>(y);
            for (int x = 0; x < labels.cols; ++x)
            {
                int lb = lrow[x];
                if (orow[x] && lb > 0)
                    keep[lb] = 1;
            }
        }

        cv::Mat out = cv::Mat::zeros(labels.size(), CV_8U);
        for (int y = 0; y < labels.rows; ++y)
        {
            const int *lrow = labels.ptr<int>(y);
            unsigned char *orow = out.ptr<unsigned char>(y);
            for (int x = 0; x < labels.cols; ++x)
            {
                int lb = lrow[x];
                if (lb > 0 && keep[lb])
                    orow[x] = 1;
            }
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// Limb cutting
// ---------------------------------------------------------------------------

// Cut radius in pixels; negative means no cut.
double computeLimbRadiusPx(const Config &config, double rsun_px,
                           int imgRows, int imgCols)
{
    const std::string &mode = config.limb_mode;

    if (mode == "none")
    {
        DBG("  limb: mode=none -> no cut");
        return -1.0;
    }

    double r = -1.0;

    if (mode == "formula")
    {
        // upstream (RSUN_OBS + 900) / CDELT1, with the measured radius
        r = rsun_px + 900.0 / 0.6;
    }
    else if (mode == "radius_px")
    {
        r = config.limb_value;
    }
    else if (mode == "radius_frac")
    {
        r = config.limb_value * rsun_px;
    }
    else if (mode == "angle")
    {
        double theta_deg = config.limb_value;
        if (theta_deg <= 0.0 || theta_deg >= 90.0)
        {
            DBG("  limb: angle " << theta_deg << " outside (0,90) -> no cut");
            return -1.0;
        }
        r = rsun_px * std::sin(theta_deg * M_PI / 180.0);
    }
    else
    {
        DBG("  limb: unknown mode '" << mode << "' -> no cut");
        return -1.0;
    }

    double halfDiag = 0.5 * std::sqrt((double)imgRows * imgRows + (double)imgCols * imgCols);
    if (r >= halfDiag)
        DBG("  limb: radius " << r << " px >= half-diagonal " << halfDiag
            << " px -> the circle covers the frame, nothing is removed");

    return r;
}

// Zero everything outside the circle.
cv::Mat removeOutside(const cv::Mat &X, double radius_px, double cx, double cy,
                      double set_value = 0.0)
{
    if (radius_px < 0.0)
        return X.clone();

    cv::Mat mask = cv::Mat::zeros(X.rows, X.cols, CV_8U);
    cv::circle(mask, cv::Point(static_cast<int>(std::lround(cx)),
                               static_cast<int>(std::lround(cy))),
               static_cast<int>(std::lround(radius_px)), cv::Scalar(1), -1);

    cv::Mat out = X.clone();
    out.setTo(set_value, mask == 0);
    return out;
}

// ---------------------------------------------------------------------------

cv::Mat connectedComp_with_sizeFilter(cv::Mat thinComponents, const std::string &type, const Config &config)
{
    int threshold_size;
    if (type == "pil")
        threshold_size = config.pil_threshold;
    else if (type == "ar")
        threshold_size = config.size_threshold;
    else
        throw std::invalid_argument("Invalid type provided. Expected 'pil' or 'ar'.");

    cv::Mat input8;
    if (thinComponents.type() != CV_8UC1)
        thinComponents.convertTo(input8, CV_8U);
    else
        input8 = thinComponents;

    cv::Mat labels, stats, centroids;
    int ncomp = cv::connectedComponentsWithStats(input8, labels, stats, centroids, 8, CV_32S);
    DBG("  sizeFilter(" << type << "): components=" << ncomp - 1);

    std::vector<unsigned char> keep(ncomp, 0);
    for (int idx = 1; idx < ncomp; ++idx)
    {
        if (stats.at<int>(idx, cv::CC_STAT_AREA) >= threshold_size)
            keep[idx] = 1;
    }

    cv::Mat out = cv::Mat::zeros(labels.size(), CV_8UC1);
    for (int y = 0; y < labels.rows; ++y)
    {
        const int *lrow = labels.ptr<int>(y);
        unsigned char *orow = out.ptr<unsigned char>(y);
        for (int x = 0; x < labels.cols; ++x)
        {
            int lb = lrow[x];
            if (lb > 0 && keep[lb])
                orow[x] = 1;
        }
    }
    return out;
}

std::vector<std::string> list_directory_magnetogram(const std::string &directory_path)
{
    std::vector<std::string> file_list;
    DIR *dir = opendir(directory_path.c_str());
    if (dir == nullptr)
    {
        std::cerr << "Error: Failed to open directory '" << directory_path << "'." << std::endl;
        return file_list;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (filename != "." && filename != "..")
        {
            if (filename.rfind("magnetogram.fits") != std::string::npos)
                file_list.push_back(filename);
        }
    }
    std::sort(file_list.begin(), file_list.end());
    closedir(dir);
    return file_list;
}

std::string get_file_basename(std::string file_path)
{
    return file_path.substr(file_path.find_last_of("/") + 1);
}

std::string get_output_file_name(const std::string file_path, const std::string output_dir)
{
    std::string base_name = get_file_basename(file_path);
    return output_dir + base_name.substr(0, base_name.find_last_of(".")) + ".h5";
}

void writeToLog(std::ofstream &log, const std::string &message)
{
    log << message << std::endl;
}

bool compute_main(const std::string &filename, const std::string &out_dir,
                  std::ofstream &logfile, const Config &config)
{
    std::string output_fn_path = get_output_file_name(filename, out_dir);
    if (std::filesystem::exists(output_fn_path))
    {
        std::cout << "File exists, skipping.\n";
        return true;
    }

    DBG("compute_main: start");

    int nColumns = 0, nRows = 0;
    int nColumns_psf = 0, nRows_psf = 0;

    if (utils::GetImageSize(filename, &nColumns, &nRows, false) != 0)
    {
        std::cerr << "ERROR: GetImageSize failed for " << filename << std::endl;
        writeToLog(logfile, "GetImageSize failed: " + filename);
        return false;
    }
    DBG("  image size: " << nRows << " x " << nColumns);

    if (nRows <= 0 || nColumns <= 0)
    {
        std::cerr << "ERROR: invalid image dimensions" << std::endl;
        writeToLog(logfile, "Invalid dimensions: " + filename);
        return false;
    }

    double *pixelVector = io::ReadImageAsVector(filename, &nColumns_psf, &nRows_psf, false);
    if (pixelVector == nullptr)
    {
        std::cerr << "ERROR: ReadImageAsVector returned NULL for " << filename << std::endl;
        writeToLog(logfile, "ReadImageAsVector failed: " + filename);
        return false;
    }
    DBG("  pixels read ok");

    std::vector<double> fits_map(pixelVector, pixelVector + (size_t)nColumns * (size_t)nRows);
    free(pixelVector);
    cv::Mat fits_image(nRows, nColumns, CV_64FC1, fits_map.data());
    DBG("  cv::Mat built");

    // --- measure the disk before anything else ---
    double cx = nColumns / 2.0;
    double cy = nRows / 2.0;
    double rsun_px = config.limb_rsun_px;

    if (config.measure_disk)
    {
        DiskGeometry disk = measureDisk(fits_image, config.disk_threshold);
        if (disk.ok)
        {
            cx = disk.cx;
            cy = disk.cy;
            rsun_px = disk.r;
            DBG("  disk: using measured centre (" << cx << ", " << cy
                << ")  R = " << rsun_px << " px");
        }
        else
        {
            DBG("  disk: measurement failed -> centre (" << cx << ", " << cy
                << ")  R = " << rsun_px << " px from config");
        }
    }
    else
    {
        DBG("  disk: measurement disabled -> centre (" << cx << ", " << cy
            << ")  R = " << rsun_px << " px from config");
    }

    try
    {
        H5::H5File h5_fid = io::create_h5_file(output_fn_path);
        DBG("  h5 created");

        std::unordered_map<std::string, std::string> metadata = io::read_header(filename);
        DBG("  header entries: " << metadata.size());

        std::unordered_map<std::string, std::string> params = configToParameterMap(config);
        // record the geometry actually used for this frame
        params["disk_cx"] = std::to_string(cx);
        params["disk_cy"] = std::to_string(cy);
        params["disk_r_px"] = std::to_string(rsun_px);

        io::write_table(metadata, h5_fid, "lineage_metadata");
        io::write_table(params, h5_fid, "parameters");
        DBG("  metadata tables written");

        // 1. DETECT POLARITY REGIONS
        pos_neg_detection pos_neg_instance;
        pos_neg_instance.identify_pos_neg_region(fits_image, nRows, nColumns, config);
        cv::Mat neg_map = pos_neg_instance.posneg_struct.neg_map;
        cv::Mat pos_map = pos_neg_instance.posneg_struct.pos_map;
        DBG("  polarity regions done");

        // 1.1 SIZE THRESHOLD ON CONNECTED COMPONENTS
        cv::Mat pos_v1 = connectedComp_with_sizeFilter(pos_map, "ar", config);
        cv::Mat neg_v1 = connectedComp_with_sizeFilter(neg_map, "ar", config);
        DBG("  size filter done");

        // 2. DILATION
        cv::Mat neg_dil = pos_neg_instance.buff_edge(neg_v1, config);
        cv::Mat pos_dil = pos_neg_instance.buff_edge(pos_v1, config);
        DBG("  dilation done");

        // 3. LIMB CUT
        double limb_r = computeLimbRadiusPx(config, rsun_px, nRows, nColumns);
        DBG("  limb: mode=" << config.limb_mode << " value=" << config.limb_value
            << " -> cut radius = " << limb_r << " px  at centre (" << cx << ", " << cy << ")");

        cv::Mat pos_nolim = removeOutside(pos_dil, limb_r, cx, cy, 0.0);
        cv::Mat neg_nolim = removeOutside(neg_dil, limb_r, cx, cy, 0.0);
        DBG("  removeOutside done");

        cv::Mat union_with_intersect = pos_neg_instance.overlapUnion(pos_nolim, neg_nolim);
        DBG("  overlapUnion done");

        cv::Mat intersection = pos_neg_instance.PIL_extraction(pos_nolim, neg_nolim);
        DBG("  PIL_extraction done");

        // Components can straddle the cut circle; cut the final rasters again.
        if (limb_r > 0.0)
        {
            union_with_intersect = removeOutside(union_with_intersect, limb_r, cx, cy, 0.0);
            intersection = removeOutside(intersection, limb_r, cx, cy, 0.0);
        }

        io::write_hdf5(h5_fid, intersection, "intersection", true,
                       "intersection of pos and neg regions", "intersection of pos and neg.");
        io::write_hdf5(h5_fid, union_with_intersect, "union_with_intersect", true,
                       "Union of pos and neg regions with overlapping",
                       "Union of pos and neg regions with overlapping.");
        DBG("  rasters written");

        io::close_hdf5(h5_fid);
    }
    catch (...)
    {
        // HDF5 handles are closed before removing this frame's partial output.
        std::error_code ec;
        fs::remove(output_fn_path, ec);
        if (ec)
        {
            const std::string message = "Failed to remove partial output " + output_fn_path + ": " + ec.message();
            std::cerr << message << std::endl;
            writeToLog(logfile, message);
        }
        throw;
    }
    DBG("compute_main: done");
    return true;
}

// create_directories() can return false for a path with a trailing slash even
// after creating it, so strip the slash and check the result directly.
int create_directory(const std::string &out_dir)
{
    fs::path p = fs::path(out_dir).lexically_normal();
    if (!p.has_filename())
        p = p.parent_path();

    std::error_code ec;
    fs::create_directories(p, ec);
    std::error_code status_ec;
    if (!fs::is_directory(p, status_ec))
    {
        const std::error_code reason = ec ? ec :
            (status_ec ? status_ec : std::make_error_code(std::errc::not_a_directory));
        std::cerr << "Failed to create directory " << p << ": " << reason.message() << std::endl;
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    std::cout << "Running in CPU mode." << std::endl;

    Config config = loadConfig("config.yaml");
    std::cout << "Size Threshold: " << config.size_threshold << std::endl;
    std::cout << "Positive Gauss Threshold: " << config.pos_gauss << std::endl;
    std::cout << "Negative Gauss Threshold: " << config.neg_gauss << std::endl;
    std::cout << "Limb mode: " << config.limb_mode
              << "  value: " << config.limb_value
              << "  measure_disk: " << (config.measure_disk ? "true" : "false")
              << "  fallback R: " << config.limb_rsun_px << " px"
              << std::endl;

    if (argc < 2)
    {
        std::cerr << "Please provide year/month/day as a command-line argument." << std::endl;
        return 1;
    }

    std::string month = argv[1];
    std::cout << "Processing full-disk(year/month/day): " << month << std::endl;

    std::string data_dir = config.data_dir + month + "/";
    std::string out_dir = config.output_dir + month + "/";
    std::string log_fn = config.log_dir + month + "/";

    if (create_directory(out_dir) != 0 || create_directory(log_fn) != 0)
        return 1;

    std::vector<std::string> file_names = list_directory_magnetogram(data_dir);
    std::cout << "Found " << file_names.size() << " magnetogram file(s) in " << data_dir << std::endl;
    if (file_names.empty())
    {
        std::cerr << "No magnetogram files found in " << data_dir << std::endl;
        return 1;
    }

    std::ofstream logfile(log_fn + "run.log");
    if (!logfile.is_open())
    {
        std::cerr << "Failed to open log file at: " << log_fn << std::endl;
        return 1;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    int i = 0;
    int n_failed = 0;
    for (const auto &file : file_names)
    {
        std::string in_file_path = data_dir + file;
        std::cout << "\n=== [" << (i + 1) << "/" << file_names.size() << "] " << in_file_path << " ===" << std::endl;
        try
        {
            if (!compute_main(in_file_path, out_dir, logfile, config))
                n_failed++;
        }
        catch (const H5::Exception &e)
        {
            std::cerr << "HDF5 EXCEPTION on " << in_file_path << ": " << e.getDetailMsg() << std::endl;
            writeToLog(logfile, "HDF5 exception: " + e.getDetailMsg() + " on " + in_file_path);
            n_failed++;
        }
        catch (const std::exception &e)
        {
            std::cerr << "EXCEPTION on " << in_file_path << ": " << e.what() << std::endl;
            writeToLog(logfile, std::string("Exception: ") + e.what() + " on " + in_file_path);
            n_failed++;
        }
        i++;
    }

    logfile.close();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "\nFull-disk: " << month << " Execution time: " << duration.count() << " ms" << std::endl;

    if (n_failed > 0)
    {
        std::cerr << n_failed << " of " << file_names.size() << " file(s) failed" << std::endl;
        return 1;
    }
    return 0;
}
