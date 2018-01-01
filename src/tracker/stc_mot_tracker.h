/*
 * stc_mot_tracker.h
 *
 *  Created on: Nov 3, 2016
 *      Author: haoquan
 */

#ifndef SRC_TRACKER_STC_MOT_TRACKER_H_
#define SRC_TRACKER_STC_MOT_TRACKER_H_

#include <sys/time.h>
#include <deque>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "util/timing_profiler.h"
#include "fftw3.h"
using namespace std;
using namespace cv::gpu;
//using namespace cv;

// -------------------------------------- Binary Matching -----------------------------------------------------
#define MAX_MATCHING_NUM 101
#define INF 999999999
class BinaryMatching {
public:
	BinaryMatching() {
	}
	;
	~BinaryMatching() {
	}
	;
	float match(const int n, const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM],
			int inv_link[MAX_MATCHING_NUM]);
private:
	int n_;
	int link_[MAX_MATCHING_NUM];
	float lx_[MAX_MATCHING_NUM], ly_[MAX_MATCHING_NUM],
			slack_[MAX_MATCHING_NUM];
	bool visx_[MAX_MATCHING_NUM], visy_[MAX_MATCHING_NUM];
	bool DFS(const int x, const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM]);
};
// --------------------------------------------------------------------------------------------------------------

// -------------------------------------- STC Tracker -----------------------------------------------------------
class STCTracker {
public:
	STCTracker();
	~STCTracker();
	void init(const Mat frame, const Rect inbox, const float resize_rto,
			const float *line);
	void tracking(const Mat frame, Rect &in_out_trackBox);
private:
	void createHammingWin();
	void complexOperation(const Mat src1, const Mat src2, Mat &dst,
			int flag = 0);
	void getCxtPriorPosteriorModel(const Mat image);
	void learnSTCModel(const Mat image);
	float linefit(const int &y);
	void FDFT_inplace(Mat& matin);
	void BDFT(Mat& matin,Mat& matout);
private:
	timing_profiler t_profiler_2;
	string prof_str;
	float line_k;
	float line_b;
	float resize_rto_;
	int FrameNum;
	double sigma;			// scale parameter (variance)
	double alpha;			// scale parameter
	double beta;			// shape parameter
	double rho;				// learning parameter
	double scale;			//	scale ratio
	double lambda;		//	scale learning parameter
	int num;					//	the number of frames for updating the scale
	int box_w, box_h, box_init_y;
	vector<double> maxValue;
	Point center;			//	the object position
	Rect cxtRegion;		// context region
	int padding;

	Mat cxtPriorPro;		// prior probability
	Mat cxtPosteriorPro;	// posterior probability
	Mat STModel;			// conditional probability
	Mat STCModel;			// spatio-temporal context model
	Mat hammingWin;			// Hamming window
};
// --------------------------------------------------------------------------------------------------------------

enum SpeedLevel {
	FAST_SPEED = 0, MED_SPEED = 1, SLOW_SPEED = 2, UNKNOWN_SPEED = 3,
};

struct Direction {
	short up_down_dir;
	short left_right_dir;
};

enum VehicleScore {
	KILL_THRESHOLD = 0, LOW_SCORE = 1, HIGH_SCORE = 2

};

typedef struct {
	unsigned long id;
	int pyr_level;
	STCTracker tracker;
} STC;

struct TObj {
	unsigned long obj_id;
	Rect loc;
	unsigned char type;
	float vehicle_pic_score;
	VehicleScore score;
	bool is_new;
	float vx, vy;
	int first_x, first_y;
	SpeedLevel sl;
	float speed;
	Direction dir;
};

struct TFrame {
	unsigned long frm_id;
	bool is_key;
	int img_w, img_h;
	vector<Mat> img_pyr;
	vector<TObj> obj;
};

struct ObjResult {
	unsigned long obj_id;
	Rect loc;
	unsigned char type;
	float score;
	SpeedLevel sl;
	Direction dir;
};

struct FrameResult {
	unsigned long frm_id;
	vector<ObjResult> obj;
};

typedef vector<FrameResult> TrackingResult;

class STC_MOT_Tracker {
public:
	STC_MOT_Tracker();
	~STC_MOT_Tracker();
	void Update(const Mat &img, const unsigned long &frm_id,
			const bool &is_key_frame, const vector<Rect> &det_box,
			const vector<unsigned char> &det_type, TrackingResult &result);
private:
	// parameters
	bool display_config_;
	bool profile_time_;
	int height_;
	int width_;
	float min_intersection;
	int track_size_;
	int min_box_size_;
	int max_wh_rto_;
	float max_size_rto_;
	float delta_v_threshold_;
	float slow_v_threshold_;
	float fast_v_threshold_;
	unsigned long id_count_;
	vector<Rect> disable_area_;
	float tl_x_, tl_y_, br_x_, br_y_;
	Size vehicle_pic_size_;
	double pi_;
	unsigned int pyr_size_;
	unsigned int tracker_base_size_h_;
	unsigned int tracker_base_size_w_;
	// VOT tracker
	vector<STC> stc_;
	// record time cost
	string t_profiler_str_;
	timing_profiler t_profiler_;

	// for binary matching use
	BinaryMatching matching_;
	float graph_[MAX_MATCHING_NUM][MAX_MATCHING_NUM];
	int match_pair_[MAX_MATCHING_NUM];
	bool is_det_matched_[MAX_MATCHING_NUM];
	// poly fit to get spatial size of the car
	int max_line_pnt_num_;
	float *line_;
	CvMemStorage* line_fit_storage_;
	CvSeq* line_fit_point_seq_;
	// local storage;
	deque<TFrame> frame_;
private:
	// main procedure
	void process_key_frame(const vector<Rect> &in_det_box,
			const vector<unsigned char> &det_type, TrackingResult &result);
	// main algorithm
	void update_line(const vector<Rect> &detections);
	int build_graph(vector<Rect> &det_box,
			const vector<unsigned char> &det_type);
	void update_vehicle_location(const int &last_key_frame_offset,
			const vector<Rect> &det_box);
	void smooth_update_vehicle_location(const int &last_key_frame_offset,
			const unsigned long &v_id, const Rect &new_loc);
	void add_new_vehicle(vector<Rect> &det_box,
			const vector<unsigned char> &det_type);
	int find_pyr_level(vector<Mat> rgb_pyr, Rect box);
	void save_matched_low_score_tracker();
	void degrade_unmatched_new_detection();
	void delete_low_score_tracker();
	void track_foreward();
	void activate_status(int last_key_frame_offset, TrackingResult &result);
	void kill_tracker(const unsigned long v_id);
	// tools
	bool is_valid_rect(Rect &b);
	float calc_intersection(const Rect &b1, const Rect &b2);
	int find_last_key_frame();
	int find_vehicle(const vector<TObj> &obj, const unsigned long v_id);
	int find_stc_idx(const unsigned long id);
	void update_vehicle_pic(TObj &sgl_obj);
	bool is_in_disable_area(Rect &b);
	float calc_angle(float &x, float &y);
	float calc_delta(float &x1, float &y1, float &x2, float &y2);
	float calc_img_dis(const Rect b1, const Rect b2, float &dx, float &dy);
	// utils
	int max(int i, int j);
	float max(float i, float j);
	int min(int i, int j);
	float min(float i, float j);
	float calc_size_rto(int i, int j);
	float length(float i, float j);
};

#endif /* SRC_TRACKER_STC_MOT_TRACKER_H_ */
