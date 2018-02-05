#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include "DL_detector/detection_ssd.h"

using namespace caffe;
using namespace cv;
using namespace std;

DetectionSSD::DetectionSSD(const DetectionType &dt) {
	int input_height, input_width;
	useGPU_ = true;
	profile_time_ = true;
	// Set model, depoly and confidence threshold
	dt_ = dt;
	if (dt_ == vehicle_detection) {
		model_file_ =
				"./model/car_detection/ModifySmallVGG_deepv_car_person_SSD_600x400_iter_250000.caffemodel";
		trained_file_ = "./model/car_detection/deploy.prototxt";
		input_height = 400;
		input_width = 600;
		batch_size_ = 1;
		conf_thr_.clear();
		conf_thr_.push_back(0.8);
		conf_thr_.push_back(0.8);
		conf_thr_.push_back(0.6);
		conf_thr_.push_back(0.6);
		conf_thr_.push_back(0.6);
	} else if (dt_ == marker_detection) {
		model_file_ =
				"./model/marker_detection/VGG_deepv_SSD_384x256_csise5_iter_117000.caffemodel";
		trained_file_ = "./model/marker_detection/tiny_deploy.prototxt";
		input_height = 256;
		input_width = 384;
		batch_size_ = 1;
		conf_thr_.clear();
		conf_thr_.push_back(1.0);
		conf_thr_.push_back(0.36);
		conf_thr_.push_back(0.6);
		conf_thr_.push_back(0.6);
		conf_thr_.push_back(0.5);
		conf_thr_.push_back(0.6);
		conf_thr_.push_back(0.6);

	} else if (dt_ == window_detection) {
		model_file_ =
				"./model/window_detection/VGG_window_night_SSD_80x160_oct_hist_dim150_batch_less2_extra_iter_44000.caffemodel";
		trained_file_ = "./model/window_detection/window_deploy.prototxt";
		input_height = 160;
		input_width = 80;
		batch_size_ = 1;
		conf_thr_.clear();
		conf_thr_.push_back(0.4);
		conf_thr_.push_back(0.4);
	} else if (dt_ == face_detection) {
		model_file_ =
				"./model/face_detection/VGG_ssd_original_iter_90000.caffemodel";
		trained_file_ =
				"./model/face_detection/test_ssd_original.prototxt";
		input_height = 180;
		input_width = 320;
		batch_size_ = 1;
		conf_thr_.clear();
		conf_thr_.push_back(0.3);
		conf_thr_.push_back(0.3);
	}
	// set GPU mode
	if (useGPU_) {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(0);
	} else
		Caffe::set_mode(Caffe::CPU);
	// Load the network.
	net_.reset(new Net<float>(trained_file_, TEST));
	net_->CopyTrainedLayersFrom(model_file_);
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	input_layer_ = net_->input_blobs()[0];
	num_channels_ = input_layer_->channels();
	input_geometry_ = Size(input_layer_->width(), input_layer_->height());
	input_geometry_.height = input_height;
	input_geometry_.width = input_width;
	input_layer_->Reshape(batch_size_, num_channels_, input_geometry_.height,
			input_geometry_.width);

	const vector< caffe::shared_ptr< caffe::Layer<float> > >& layers = net_->layers();
	const vector<vector<Blob<float>*> >& bottom_vecs = net_->bottom_vecs();
	const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();
	for (int i = 0; i < layers.size(); ++i)
		layers[i]->Forward(bottom_vecs[i], top_vecs[i]);

}

void DetectionSSD::PredictBatch(const vector<Mat> &imgs) {
	float* input_data = input_layer_->mutable_cpu_data();
	int cnt = 0;
	for (int i = 0; i < imgs.size(); i++) {
		Mat sample;
		Mat img = imgs[i].clone();
		if (dt_ == window_detection) {
			cvtColor(img, img, CV_BGR2GRAY);
			equalizeHist(img, img);
		}
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, CV_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cvtColor(img, sample, CV_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cvtColor(img, sample, CV_GRAY2BGR);
		else
			sample = img;
		if ((sample.rows != input_geometry_.height)
				|| (sample.cols != input_geometry_.width)) {
			resize(sample, sample,
					Size(input_geometry_.width, input_geometry_.height));
		}
		float mean[3] = { 104, 117, 123 };
		for (int k = 0; k < sample.channels(); k++) {
			for (int i = 0; i < sample.rows; i++) {
				for (int j = 0; j < sample.cols; j++) {
					input_data[cnt] = (float(sample.at < uchar > (i, j * 3 + k))
							- mean[k]);
					cnt += 1;
				}
			}
		}
	}
	// Forward dimension change to all layers.
	net_->Reshape();
	net_->ForwardPrefilled();
	if (useGPU_) {
		cudaDeviceSynchronize();
	}
	// Copy the output layer to a vector
	for (int i = 0; i < net_->num_outputs(); i++) {
		Blob<float>* output_layer = net_->output_blobs()[i];
		outputs_.push_back(output_layer);
	}
}

void DetectionSSD::Predict(const vector<Mat> &imgs,
		vector<vector<cv::Rect> > &position, vector<vector<float> > &confidence,
		vector<vector<unsigned char> > &type) {
	if (profile_time_)
		time_profiler_.reset();
	PredictBatch(imgs);
	position.clear();
	confidence.clear();
	type.clear();
	int box_num = outputs_[0]->height();
	int box_length = outputs_[0]->width();
	const float* top_data = outputs_[0]->cpu_data();
	for (int i = 0; i < batch_size_; i++) {
		vector < Rect > sgl_img_position;
		vector<float> sgl_confidence;
		vector<unsigned char> sgl_type;
		position.push_back(sgl_img_position);
		confidence.push_back(sgl_confidence);
		type.push_back(sgl_type);
	}
	for (int j = 0; j < box_num; j++) {
		int i = top_data[j * 7 + 0];
		int cls = top_data[j * 7 + 1];
		float score = top_data[j * 7 + 2];
		if ((i < batch_size_) && (cls != 0 && score > conf_thr_[cls])) {
			type[i].push_back(cls);
			confidence[i].push_back(score);
			Rect sgl_position;
			sgl_position.x = top_data[j * 7 + 3] * imgs[i].cols;
			sgl_position.y = top_data[j * 7 + 4] * imgs[i].rows;
			sgl_position.width = top_data[j * 7 + 5] * imgs[i].cols
					- sgl_position.x + 1;
			sgl_position.height = top_data[j * 7 + 6] * imgs[i].rows
					- sgl_position.y + 1;
			position[i].push_back(sgl_position);
		}
	}
	if (profile_time_) {
		time_profiler_str_ = "DetectionCost";
		time_profiler_.update(time_profiler_str_);
	}
}
