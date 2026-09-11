#include "BoWGDetector.h"
#include "Parameters.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct NativeConfig {
    std::string pattern_path;
    std::string vocabulary_path;
    int feature_count;
    bool multi_scale;
    int multi_scale_levels;
    float feature_size;
    int word_weight_type;
    int word_scoring_type;
    int word_group_weight_type;
    int word_group_scoring_type;
    bool use_word_groups;
    bool use_distribution;
    double word_weight;
    double word_group_weight;
    bool use_temporal_score;
    double previous_weight_threshold;
    double temporal_parameter;
    double similarity_threshold;
    int temporal_consistency;
    int max_intraisland_gap;
    int max_distance_between_groups;
    int max_distance_between_queries;
    bool use_native_geometry;
    bool direct_index_geometry;
    int direct_index_level;
    int native_min_feature_points;
    double native_max_reprojection_error;
    double native_ransac_probability;
    int native_ransac_max_iterations;
    double native_max_neighbor_ratio;
    double min_previous_word_score;
    double min_previous_word_group_score;
    double min_previous_distribution_score;
};

std::mutex global_bowg_mutex;

template <typename T>
T required(const py::dict& config, const char* name) {
    if (!config.contains(name)) {
        throw std::invalid_argument(std::string("missing BoWG config field: ") + name);
    }
    return config[name].cast<T>();
}

NativeConfig parseConfig(const py::dict& config) {
    NativeConfig value;
    value.pattern_path = required<std::string>(config, "pattern_path");
    value.vocabulary_path = required<std::string>(config, "vocabulary_path");
    value.feature_count = required<int>(config, "feature_count");
    value.multi_scale = required<bool>(config, "multi_scale");
    value.multi_scale_levels = required<int>(config, "multi_scale_levels");
    value.feature_size = required<float>(config, "feature_size");
    value.word_weight_type = required<int>(config, "word_weight_type");
    value.word_scoring_type = required<int>(config, "word_scoring_type");
    value.word_group_weight_type = required<int>(config, "word_group_weight_type");
    value.word_group_scoring_type = required<int>(config, "word_group_scoring_type");
    value.use_word_groups = required<bool>(config, "use_word_groups");
    value.use_distribution = required<bool>(config, "use_distribution");
    value.word_weight = required<double>(config, "word_weight");
    value.word_group_weight = required<double>(config, "word_group_weight");
    value.use_temporal_score = required<bool>(config, "use_temporal_score");
    value.previous_weight_threshold = required<double>(config, "previous_weight_threshold");
    value.temporal_parameter = required<double>(config, "temporal_parameter");
    value.similarity_threshold = required<double>(config, "similarity_threshold");
    value.temporal_consistency = required<int>(config, "temporal_consistency");
    value.max_intraisland_gap = required<int>(config, "max_intraisland_gap");
    value.max_distance_between_groups = required<int>(config, "max_distance_between_groups");
    value.max_distance_between_queries = required<int>(config, "max_distance_between_queries");
    value.use_native_geometry = required<bool>(config, "use_native_geometry");
    value.direct_index_geometry = required<bool>(config, "direct_index_geometry");
    value.direct_index_level = required<int>(config, "direct_index_level");
    value.native_min_feature_points = required<int>(config, "native_min_feature_points");
    value.native_max_reprojection_error = required<double>(config, "native_max_reprojection_error");
    value.native_ransac_probability = required<double>(config, "native_ransac_probability");
    value.native_ransac_max_iterations = required<int>(config, "native_ransac_max_iterations");
    value.native_max_neighbor_ratio = required<double>(config, "native_max_neighbor_ratio");
    value.min_previous_word_score = required<double>(config, "min_previous_word_score");
    value.min_previous_word_group_score = required<double>(config, "min_previous_word_group_score");
    value.min_previous_distribution_score = required<double>(config, "min_previous_distribution_score");
    return value;
}

void applyConfig(const NativeConfig& config) {
    BRIEF_PATTERN_FILE = config.pattern_path;
    VOCABULARY_PATH = config.vocabulary_path;
    FEATURE_CNT = config.feature_count;
    MS_FEATURE = config.multi_scale;
    MS_LEVELS = config.multi_scale_levels;
    FEATURE_SIZE = config.feature_size;
    W_WEIGHT_TYPE = config.word_weight_type;
    W_SCORING_TYPE = config.word_scoring_type;
    WG_WEIGHT_TYPE = config.word_group_weight_type;
    WG_SCORING_TYPE = config.word_group_scoring_type;
    USE_WG = config.use_word_groups;
    USE_DISTRIBUTION = config.use_distribution;
    W_WEIGHT = config.word_weight;
    WG_WEIGHT = config.word_group_weight;
    USE_TEMPORAL_SCORE = config.use_temporal_score;
    PREV_WEIGHT_TH = config.previous_weight_threshold;
    TEMPORAL_PARAM = config.temporal_parameter;
    SIMILARITY_TH = config.similarity_threshold;
    TEMPORAL_K = config.temporal_consistency;
    MAX_INTRAISLAND_GAP = config.max_intraisland_gap;
    MAX_DISTANCE_BETWEEN_GROUPS = config.max_distance_between_groups;
    MAX_DISTANCE_BETWEEN_QUERIES = config.max_distance_between_queries;
    USE_GEOM = config.use_native_geometry;
    GEOM_DI = config.direct_index_geometry;
    DI_LEVEL = config.direct_index_level;
    MIN_FPOINTS = config.native_min_feature_points;
    MAX_REPROJECTION_ERROR = config.native_max_reprojection_error;
    RANSAC_PROBABILITY = config.native_ransac_probability;
    MAX_RANSAC_ITERATIONS = config.native_ransac_max_iterations;
    MAX_NEIGHBOR_RATIO = config.native_max_neighbor_ratio;
    MIN_PREV_W_SCORE = config.min_previous_word_score;
    MIN_PREV_WG_SCORE = config.min_previous_word_group_score;
    MIN_PREV_DIST_SCORE = config.min_previous_distribution_score;
}

void packDescriptor(const DVision::BRIEF::bitset& descriptor, std::uint8_t* output) {
    if (descriptor.size() != 256) {
        throw std::runtime_error("BoWG returned a BRIEF descriptor that is not 256 bits");
    }
    for (std::size_t bit = 0; bit < 256; ++bit) {
        if (descriptor.test(bit)) {
            output[bit / 8] |= static_cast<std::uint8_t>(1u << (bit % 8));
        }
    }
}

py::dict resultToPython(const BoWG::DetectionResult& result) {
    py::array_t<float> keypoints({result.features.keypoints.size(), std::size_t(2)});
    py::array_t<std::uint8_t> descriptors(
        {result.features.descriptors.size(), std::size_t(32)});
    auto points = keypoints.mutable_unchecked<2>();
    auto bytes = descriptors.mutable_unchecked<2>();
    for (std::size_t row = 0; row < result.features.keypoints.size(); ++row) {
        points(row, 0) = result.features.keypoints[row].pt.x;
        points(row, 1) = result.features.keypoints[row].pt.y;
        std::uint8_t* output = &bytes(row, 0);
        for (std::size_t column = 0; column < 32; ++column) {
            output[column] = 0;
        }
        packDescriptor(result.features.descriptors[row], output);
    }

    py::list candidates;
    for (const BoWG::Candidate& candidate : result.candidates) {
        candidates.append(py::make_tuple(candidate.entry_id, candidate.score));
    }
    py::dict output;
    output["entry_id"] = result.entry_id;
    output["candidates"] = candidates;
    output["keypoints_xy"] = std::move(keypoints);
    output["descriptors"] = std::move(descriptors);
    output["extract_ms"] = result.extract_ms;
    output["query_ms"] = result.query_ms;
    return output;
}

class NativeEngine {
public:
    explicit NativeEngine(const py::dict& config) : config_(parseConfig(config)) {
        std::lock_guard<std::mutex> lock(global_bowg_mutex);
        applyConfig(config_);
        detector_.reset(new BoWG::BoWGDetector(false, 0));
    }

    py::dict queryThenAdd(
        py::array_t<std::uint8_t, py::array::c_style> image,
        int top_k,
        int temporal_exclusion) {
        const py::buffer_info info = image.request();
        if (info.ndim != 3 || info.shape[0] <= 0 || info.shape[1] <= 0 ||
            info.shape[2] != 3) {
            throw std::invalid_argument("image_bgr must be a non-empty uint8 HxWx3 array");
        }
        cv::Mat view(
            static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_8UC3,
            info.ptr, static_cast<std::size_t>(info.strides[0]));
        BoWG::DetectionResult result;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(global_bowg_mutex);
            applyConfig(config_);
            result = detector_->queryThenAdd(view, top_k, temporal_exclusion);
        }
        return resultToPython(result);
    }

    void reset() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(global_bowg_mutex);
        applyConfig(config_);
        detector_->reset();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(global_bowg_mutex);
        return detector_->size();
    }

private:
    NativeConfig config_;
    std::unique_ptr<BoWG::BoWGDetector> detector_;
};

}  // namespace

PYBIND11_MODULE(_bowg, module) {
    module.doc() = "Native BoWG retrieval facade for FGS-SLAM";
    py::class_<NativeEngine>(module, "Engine")
        .def(py::init<const py::dict&>())
        .def("query_then_add", &NativeEngine::queryThenAdd,
             py::arg("image_bgr"), py::arg("top_k"), py::arg("temporal_exclusion"))
        .def("reset", &NativeEngine::reset)
        .def_property_readonly("size", &NativeEngine::size);
    module.def("_pack_descriptor_bits_for_test", [](const std::vector<std::size_t>& bits) {
        DVision::BRIEF::bitset descriptor(256);
        for (std::size_t bit : bits) {
            if (bit >= 256) {
                throw std::invalid_argument("descriptor bit index must be below 256");
            }
            descriptor.set(bit);
        }
        py::array_t<std::uint8_t> packed(32);
        auto output = packed.mutable_unchecked<1>();
        for (std::size_t index = 0; index < 32; ++index) {
            output(index) = 0;
        }
        packDescriptor(descriptor, &output(0));
        return packed;
    });
}
