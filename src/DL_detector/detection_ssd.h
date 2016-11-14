#ifndef DETECTION_SSD_
#define DETECTION_SSD_

#include <opencv2/core/core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include "caffe/caffe.hpp"
#include "util/timing_profiler.h"

enum DetectionType {
	vehicle_detection = 0,
	window_detection = 1,
	marker_detection = 2,
	face_detection = 3
};

enum WindowType {
	window_background = 0, window = 1
};

enum MarkerType {
	marker_background = 0,
	nianjianbiao = 1,
	zheyangban = 2,
	baijian = 3,
	anquandai = 4,
	guazhui = 5,
	zhijinhe = 6
};

enum VehicleType {
	vehicle_background = 0, car = 1, person = 2, bicycle = 3, tricycle = 4
};

enum FaceType {
	face_background = 0, face = 1
};

class DetectionSSD {
public:
	DetectionSSD(const DetectionType &dt);
	void Predict(const std::vector<cv::Mat> &imgs,
			std::vector<std::vector<cv::Rect> > &position,
			std::vector<std::vector<float> > &confidence,
			std::vector<std::vector<unsigned char> > &type);
	char *get_detection_time_cost() {
		if (profile_time_)
			return time_profiler_.getSmoothedTimeProfileString();
		else
			return "TimeProfiling is not opened!";
	}
	int get_batch_size() {
		return batch_size_;
	}
private:
	DetectionType dt_;
	caffe::shared_ptr<caffe::Net<float> > net_;
	caffe::Blob<float>* input_layer_;
	std::vector<caffe::Blob<float>*> outputs_;
	cv::Size input_geometry_;
	std::string model_file_;
	std::string trained_file_;
	int num_channels_;
	int batch_size_;
	cv::Mat mean_;
	bool useGPU_;
	std::vector<float> conf_thr_;
	bool profile_time_;
	std::string time_profiler_str_;
	timing_profiler time_profiler_;
	void PredictBatch(const std::vector<cv::Mat> &imgs);
};
#endif
