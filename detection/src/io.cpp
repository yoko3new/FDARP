/*
   Copyright [2023] [Ziba Khani]

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

   Modified (see comments on read_header and write_table).
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <opencv2/core.hpp>
#include <H5Cpp.h>

#include <stdexcept>

#include "fitsio.h"

#include "io.hpp"
#include "utils.hpp"
#include "model_parameters.hpp"

namespace io
{
    /* ---------------- FUNCTION: ReadImageAsVector ------------------------ */
    /*    Given a filename, it opens the file, reads the size of the image and
     * stores that size in *nRows and *nColumns, then allocates memory for a 1-D
     * array to hold the image and reads the image from the file into the
     * array.  Finally, it returns the image array -- or, more precisely, it
     * returns a pointer to the array; it also stores the image dimensions
     * in the pointer-parameters nRows and nColumns.
     *    Note that this function does *not* use Numerical Recipes functions; instead
     * it allocates a standard 1-D C vector [this means that the first index will
     * be 0, not 1].
     *
     *    Returns 0 for successful operation, -1 if a CFITSIO-related error occurred.
     *
     */
    double *ReadImageAsVector(std::string filename, int *nColumns, int *nRows,
                              bool verbose)
    {
        fitsfile *imfile_ptr;
        double *imageVector;
        int status, nfound;
        int problems;
        long naxes[2];
        int nPixelsTot;
        long firstPixel[2] = {1, 1};
        int n_rows, n_columns;

        status = problems = 0;
        std::string fn = filename + "[1]";
        /* Open the FITS file: */
        problems = fits_open_file(&imfile_ptr, fn.c_str(), READONLY, &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems opening FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            return NULL;
        }

        /* read the NAXIS1 and NAXIS2 keyword to get image size */
        problems = fits_read_keys_lng(imfile_ptr, "ZNAXIS", 1, 2, naxes, &nfound,
                                      &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems reading FITS keywords from file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            fits_close_file(imfile_ptr, &status); // Close file before returning
            return NULL;
        }
        if (verbose)
            printf("ReadImageAsVector: Image keywords: NAXIS1 = %ld, NAXIS2 = %ld\n", naxes[0], naxes[1]);

        n_columns = naxes[0]; // FITS keyword NAXIS1 = # columns
        *nColumns = n_columns;
        n_rows = naxes[1]; // FITS keyword NAXIS2 = # rows
        *nRows = n_rows;
        nPixelsTot = n_columns * n_rows; // number of pixels in the image

        // Allocate memory for the image-data vector:
        imageVector = (double *)malloc(nPixelsTot * sizeof(double));
        // Read in the image data
        problems = fits_read_pix(imfile_ptr, TDOUBLE, firstPixel, nPixelsTot, NULL, imageVector,
                                 NULL, &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems reading pixel data from FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            free(imageVector);                    // Free allocated memory
            fits_close_file(imfile_ptr, &status); // Close file before returning
            return NULL;
        }

        if (verbose)
            printf("\nReadImageAsVector: Image read.\n");

        problems = fits_close_file(imfile_ptr, &status);
        if (problems)
        {
            fprintf(stderr, "\n*** WARNING: Problems closing FITS file \"%s\"!\n    FITSIO error messages follow:", filename.c_str());
            utils::PrintError(status);
            return NULL;
        }
        // Print(imageVector);
        return imageVector;
    }

    H5::DataType get_hdf5_data_type(const cv::Mat &data, bool use_int)
    {
        switch (data.type())
        {
        case CV_8UC1:
            return H5::PredType::NATIVE_UINT8;
        case CV_8SC1:
            return H5::PredType::NATIVE_INT8;
        case CV_16UC1:
            return H5::PredType::NATIVE_UINT16;
        case CV_16SC1:
            return H5::PredType::NATIVE_INT16;
        case CV_32SC1:
            return H5::PredType::NATIVE_INT32;
        case CV_32FC1:
            return H5::PredType::NATIVE_FLOAT;
        case CV_64FC1:
            return H5::PredType::NATIVE_DOUBLE;
        default:
            throw std::runtime_error("Unsupported cv::Mat data type for HDF5.");
        }
    }

    H5::H5File create_h5_file(const std::string &filename)
    {
        try
        {
            H5::H5File file(filename, H5F_ACC_TRUNC);
            return file;
        }
        catch (const H5::FileIException &e)
        {
            std::cerr << "Error: Could not create HDF5 file: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }

    void add_attribute_to_dataset(H5::DataSet &dataset, const std::string &attr_name, const std::string &attr_value)
    {
        H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);
        H5::StrType attr_strtype(H5::PredType::C_S1, attr_value.size());
        H5::Attribute attribute = dataset.createAttribute(attr_name, attr_strtype, attr_dataspace);
        attribute.write(attr_strtype, attr_value);
    }

    void write_hdf5(H5::H5File &file, cv::Mat data, const std::string &var_name, bool use_int_type, const std::string &dataset_long_name, const std::string &dataset_description)
    {
        try
        {
            cv::Mat continuous_data;
            continuous_data = data.clone();
            if (!continuous_data.isContinuous())
            {
                continuous_data = continuous_data.clone();
            }
            int width = continuous_data.rows;
            int height = continuous_data.cols;

            hsize_t chunk_dims[2] = {
                static_cast<hsize_t>(width),
                static_cast<hsize_t>(height)};

            hsize_t dims[2] = {static_cast<hsize_t>(continuous_data.rows),
                               static_cast<hsize_t>(continuous_data.cols)};
            H5::DataSpace dataspace(2, dims);

            H5::DSetCreatPropList plist;
            plist.setDeflate(6);
            plist.setChunk(2, chunk_dims);

            H5::DataSet dataset = file.createDataSet(var_name, get_hdf5_data_type(data, use_int_type), dataspace, plist);

            add_attribute_to_dataset(dataset, "Description", dataset_description);
            add_attribute_to_dataset(dataset, "LongName", dataset_long_name);

            dataset.write(continuous_data.data, get_hdf5_data_type(data, use_int_type));

            dataset.close();
        }
        catch (const H5::DataSetIException &e)
        {
            std::cerr << "Error: Could not create dataset: " << e.getCDetailMsg() << std::endl;
            throw;
        }
        catch (const H5::DataSpaceIException &e)
        {
            std::cerr << "Error: Could not create dataspace: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }

    void close_hdf5(H5::H5File &file)
    {
        try
        {
            file.close();
        }
        catch (const H5::FileIException &e)
        {
            std::cerr << "Error: Could not close HDF5 file: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }

    bool containsAny(const std::string &target, const std::vector<std::string> &strings)
    {
        for (const std::string &str : strings)
        {
            if (target.find(str) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    // Fixed: the old parser read cards with `ss >> key >> ws >> eq >> value`.
    // In FITS the '=' is attached to the keyword ("RSUN_OBS="), so `key`
    // swallowed it and `eq` took the first digit of the value
    // (RSUN_OBS = 976.0 was read as 76.0). Now splits on the first '=' and
    // matches keywords exactly (the old substring test let CRPIX1 match CRPIX10).
    std::unordered_map<std::string, std::string> read_header(std::string filename)
    {
        std::unordered_map<std::string, std::string> key_value_pairs;

        fitsfile *fitsFilePtr = nullptr;
        int status = 0;

        if (fits_open_file(&fitsFilePtr, filename.c_str(), READONLY, &status))
        {
            std::cerr << "read_header: cannot open " << filename << std::endl;
            utils::PrintError(status);
            return key_value_pairs;
        }

        // Move to HDU 2 (the image extension; HDU 1 is the empty primary)
        int hdutype = 0;
        if (fits_movabs_hdu(fitsFilePtr, 2, &hdutype, &status))
        {
            std::cerr << "read_header: cannot move to HDU 2 in " << filename << std::endl;
            utils::PrintError(status);
            fits_close_file(fitsFilePtr, &status);
            return key_value_pairs;
        }

        int nkeys = 0;
        if (fits_get_hdrspace(fitsFilePtr, &nkeys, NULL, &status))
        {
            utils::PrintError(status);
            fits_close_file(fitsFilePtr, &status);
            return key_value_pairs;
        }

        const std::vector<std::string> keys_to_extract = {
            "QUALITY", "ORIGIN", "CONTENT", "BUNIT", "HARPNUM", "TELESCOP",
            "WCSNAME", "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "T_REC",
            "SIZE_ACR", "AREA_ACR", "CRPIX1", "CRPIX2", "RSUN_OBS", "CDELT1"};

        auto trim = [](const std::string &s) {
            const char *ws = " \t\r\n";
            size_t b = s.find_first_not_of(ws);
            if (b == std::string::npos)
                return std::string();
            size_t e = s.find_last_not_of(ws);
            return s.substr(b, e - b + 1);
        };

        char card[FLEN_CARD];
        for (int i = 1; i <= nkeys; ++i)
        {
            if (fits_read_record(fitsFilePtr, i, card, &status))
            {
                utils::PrintError(status);
                status = 0;
                continue;
            }

            std::string line(card);

            // Split on the FIRST '='. In a FITS card the keyword occupies
            // columns 1-8 and the value indicator "= " columns 9-10.
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos)
                continue; // COMMENT / HISTORY / END cards carry no '='

            std::string key = trim(line.substr(0, eq_pos));
            std::string value = line.substr(eq_pos + 1);

            // Drop the trailing "/ comment", but keep a '/' inside a quoted
            // string (dates and paths contain slashes).
            bool in_quotes = false;
            for (size_t p = 0; p < value.size(); ++p)
            {
                if (value[p] == '\'')
                    in_quotes = !in_quotes;
                else if (value[p] == '/' && !in_quotes)
                {
                    value = value.substr(0, p);
                    break;
                }
            }

            value = trim(value);

            // Strip surrounding single quotes from string-valued cards
            if (value.size() >= 2 && value.front() == '\'' && value.back() == '\'')
                value = trim(value.substr(1, value.size() - 2));

            if (key.empty() || value.empty())
                continue;

            // Exact keyword match (substring matching would confuse e.g.
            // CRPIX1 with CRPIX10)
            for (const std::string &wanted : keys_to_extract)
            {
                if (key == wanted)
                {
                    key_value_pairs[key] = value;
                    break;
                }
            }
        }

        fits_close_file(fitsFilePtr, &status);
        if (status != 0)
            utils::PrintError(status);

        return key_value_pairs;
    }

    std::unordered_map<std::string, std::string> get_params(ModelParameters params)
    {
        try
        {
            struct ParamInfo
            {
                std::string name;
                std::string value;
            };

            std::vector<ParamInfo> paramInfos = {
                {"pos_gauss", std::to_string(params.pos_gauss)},
                {"neg_gauss", std::to_string(params.neg_gauss)},
                {"dilation_size", std::to_string(params.dilation_size)},
                {"strength_threshold", std::to_string(params.strength_threshold)},
                {"size_threshold", std::to_string(params.size_threshold)},
                {"gap_size", std::to_string(params.gap_size)}};

            std::unordered_map<std::string, std::string> key_value_pairs;
            for (const ParamInfo &paramInfo : paramInfos)
            {
                key_value_pairs.insert(std::make_pair(paramInfo.name, paramInfo.value));
            }

            return key_value_pairs;
        }
        catch (const H5::Exception &e)
        {
            std::cerr << "Error: " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }

    void writeKeyValuePairsToCSV(const std::string &csvFilePath, const std::unordered_map<std::string, std::string> &keyValuePairs)
    {
        std::ofstream csvFile(csvFilePath);
        if (!csvFile)
        {
            std::cerr << "Error creating CSV file: " << csvFilePath << std::endl;
            return;
        }
        csvFile << "Key,Value\n";
        for (const auto &kvp : keyValuePairs)
        {
            csvFile << kvp.first << "," << kvp.second << "\n";
        }
        csvFile.close();
    }

    void write_properties(const std::string &csvFilePath, const std::map<std::string, std::string> &keyValuePairs)
    {
        std::ofstream csvFile(csvFilePath);
        if (!csvFile)
        {
            std::cerr << "Error creating CSV file: " << csvFilePath << std::endl;
            return;
        }
        csvFile << "Key,Value\n";
        for (const auto &kvp : keyValuePairs)
        {
            csvFile << kvp.first << "," << kvp.second << "\n";
        }
        csvFile.close();
    }

    // Fixed: std::string members were written through an HDF5 variable-length
    // string type, which expects char*; this segfaults on Clang/macOS. Also
    // replaced a variable-length array with std::vector.
    void write_table(std::unordered_map<std::string, std::string> properties, H5::H5File &file, std::string datasetName)
    {
        if (properties.empty())
            return;

        struct TableRow
        {
            const char *key;
            const char *value;
        };

        // Keep the actual strings alive while HDF5 reads through the pointers
        std::vector<std::string> keys, values;
        keys.reserve(properties.size());
        values.reserve(properties.size());
        for (const auto &kv : properties)
        {
            keys.push_back(kv.first);
            values.push_back(kv.second);
        }

        std::vector<TableRow> rows(properties.size());
        for (size_t i = 0; i < keys.size(); ++i)
        {
            rows[i].key = keys[i].c_str();
            rows[i].value = values[i].c_str();
        }

        try
        {
            H5::StrType varStr(H5::PredType::C_S1, H5T_VARIABLE);
            H5::CompType rowType(sizeof(TableRow));
            rowType.insertMember("key", HOFFSET(TableRow, key), varStr);
            rowType.insertMember("value", HOFFSET(TableRow, value), varStr);

            hsize_t dims[1] = {static_cast<hsize_t>(rows.size())};
            H5::DataSpace dataSpace(1, dims);
            H5::DataSet dataSet = file.createDataSet(datasetName, rowType, dataSpace);
            dataSet.write(rows.data(), rowType);
            dataSet.close();
        }
        catch (const H5::Exception &e)
        {
            std::cerr << "Error writing table '" << datasetName << "': " << e.getCDetailMsg() << std::endl;
            throw;
        }
    }
}
