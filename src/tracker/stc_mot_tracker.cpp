/*
 * stc_mot_tracker.cpp
 *
 *  Created on: Nov 3, 2016
 *      Author: haoquan
 */

#include <glog/logging.h>
#include "tracker/stc_mot_tracker.h"
ofstream testfile("tt");
STC_MOT_Tracker::STC_MOT_Tracker() {
	// set parameters
	display_config_ = false;
	profile_time_ = true;
	min_intersection = 50;
	track_size_ = 20;
	min_box_size_ = 5;
	max_wh_rto_ = 10;
	delta_v_threshold_ = 10;
	slow_v_threshold_ = 0.02;
	fast_v_threshold_ = 0.12;
	max_line_pnt_num_ = 200;
	max_size_rto_ = 4;
	vehicle_pic_size_.height = 100;
	vehicle_pic_size_.width = 100;
	// constant parameters
	pi_ = 3.141592653;
	pyr_size_ = 1;
	tracker_base_size_h_ = 240;
	tracker_base_size_w_ = 320;
	// initialization
	height_ = tracker_base_size_h_;
	width_ = tracker_base_size_w_;
	matching_ = BinaryMatching();
	stc_.clear();
	id_count_ = 0;
	// initialize auto size adjustment
	line_ = new float[4];
	line_[0] = 0;
	line_fit_storage_ = cvCreateMemStorage(0);
	line_fit_point_seq_ = cvCreateSeq(CV_32FC2, sizeof(CvSeq),
			sizeof(CvPoint2D32f), line_fit_storage_);
	disable_area_.clear();
//    Rect disable_area1;
//    disable_area1.x = 0;
//    disable_area1.y = 0;
//    disable_area1.width = 320;
//    disable_area1.height = 40;
//    disable_area_.push_back(disable_area1);
	// clear local storage
	frame_.clear();
}

STC_MOT_Tracker::~STC_MOT_Tracker() {
	cvClearSeq(line_fit_point_seq_);
	cvReleaseMemStorage(&line_fit_storage_);
	delete line_;
}

void STC_MOT_Tracker::Update(const Mat &img, const unsigned long &frm_id,
		const bool &is_key_frame, const vector<Rect> &det_box,
		const vector<unsigned char> &det_type, TrackingResult &result) {
	// copy inputs to local storage

	TFrame new_frame;
	new_frame.frm_id = frm_id;
	new_frame.img_w = img.cols;
	new_frame.img_h = img.rows;
	new_frame.is_key = is_key_frame;
	Size new_size;
	new_size.width = tracker_base_size_w_;
	new_size.height = tracker_base_size_h_;
	t_profiler_.reset();
	for (int i = 0; i < pyr_size_; i++) {
		Mat new_rgb;

//		cout<<"origin"<<img.cols<<" "<<img.rows<<endl;
		resize(img, new_rgb, new_size, 0, 0, CV_INTER_LINEAR);
//		cout<<"new"<<new_size.height<<" "<<new_size.width<<endl;

		new_size.width = new_size.width * 0.75;
		new_size.height = new_size.height * 0.75;
		new_frame.img_pyr.push_back(new_rgb);
	}
	t_profiler_str_ = "pyr Resize";
	t_profiler_.update(t_profiler_str_);
	frame_.push_back(new_frame);
	// clear result
	result.clear();
	if (display_config_) {
		printf("init track update\n");
		printf("Frame %lu ---------------------------------------------\n",
				frame_[frame_.size() - 1].frm_id);
		printf("track update\n");
	}

	if (frame_.size() > 1)
		track_foreward();
	t_profiler_str_ = "STC";
	t_profiler_.update(t_profiler_str_);
	if (display_config_) {
		printf("after track forward **track:\n");
		for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size();
				v_idx++)
			printf("after track forward **%d %d %d %d\n",
					frame_[frame_.size() - 1].obj[v_idx].loc.x,
					frame_[frame_.size() - 1].obj[v_idx].loc.y,
					frame_[frame_.size() - 1].obj[v_idx].loc.width,
					frame_[frame_.size() - 1].obj[v_idx].loc.height);
	}
	if (frame_[frame_.size() - 1].is_key)
		process_key_frame(det_box, det_type, result);
	t_profiler_str_ = "Matching";
	t_profiler_.update(t_profiler_str_);
//	if (profile_time_)
//		LOG(INFO) << t_profiler_.getSmoothedTimeProfileString();
}

void STC_MOT_Tracker::process_key_frame(const vector<Rect> &in_det_box,
		const vector<unsigned char> &det_type, TrackingResult &result) {
	vector<Rect> det_box;
	for (int i = 0; i < in_det_box.size(); i++) {
		float w_rto = 1.0 * tracker_base_size_w_
				/ frame_[frame_.size() - 1].img_w;
		float h_rto = 1.0 * tracker_base_size_h_
				/ frame_[frame_.size() - 1].img_h;
		Rect sgl_det_box;
		sgl_det_box.x = max(1, (int) (w_rto * in_det_box[i].x));
		sgl_det_box.y = max(1, (int) (h_rto * in_det_box[i].y));
		sgl_det_box.width = min((int) (w_rto * in_det_box[i].width),
				tracker_base_size_w_ - sgl_det_box.x - 1);
		sgl_det_box.height = min((int) (h_rto * in_det_box[i].height),
				tracker_base_size_h_ - sgl_det_box.y - 1);
		det_box.push_back(sgl_det_box);
	}
	for (int d_idx = 0;
			d_idx
					< max((int) frame_[frame_.size() - 1].obj.size(),
							(int) det_box.size()); d_idx++)
		is_det_matched_[d_idx] = false;
	int last_key_frame_offset = find_last_key_frame();
	if (line_fit_point_seq_->total <= max_line_pnt_num_)
		update_line(det_box);
	int graph_size = build_graph(det_box, det_type);
	matching_.match(graph_size, graph_, match_pair_);
	if (display_config_) {
		printf("matching pair : ");
		for (int i = 0; i < graph_size; i++)
			printf("%d ", match_pair_[i]);
		printf("\n");
	}
	if (last_key_frame_offset >= 0)
		update_vehicle_location(last_key_frame_offset, det_box, in_det_box);
	if (display_config_) {
		printf("**track:\n");
		for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size();
				v_idx++)
			printf("**%d %d %d %d\n",
					frame_[frame_.size() - 1].obj[v_idx].loc.x,
					frame_[frame_.size() - 1].obj[v_idx].loc.y,
					frame_[frame_.size() - 1].obj[v_idx].loc.width,
					frame_[frame_.size() - 1].obj[v_idx].loc.height);
	}
	save_matched_low_score_tracker();
	degrade_unmatched_new_detection();
	delete_low_score_tracker();
	add_new_vehicle(det_box, det_type, in_det_box);
	if (display_config_) {
		if (frame_.size() > 1)
			printf("previous V num %lu\n",
					frame_[frame_.size() - 2].obj.size());
		printf("current V num %lu\n", frame_[frame_.size() - 1].obj.size());
		for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size();
				v_idx++)
			printf("*V id = %lu, V score = %d\n",
					frame_[frame_.size() - 1].obj[v_idx].obj_id,
					frame_[frame_.size() - 1].obj[v_idx].score);
		for (int v_idx = 0; v_idx < stc_.size(); v_idx++)
			printf("*V id = %lu\n", stc_[v_idx].id);
	}
	if (last_key_frame_offset > 0)
		activate_status(last_key_frame_offset, result);
}

void STC_MOT_Tracker::update_line(const vector<Rect> &detections) {
	for (int i = 0; i < detections.size(); i++)
		if ((detections[i].x > 10) && (detections[i].y > 10)
				&& (detections[i].x + detections[i].width < width_ - 10)
				&& (detections[i].y + detections[i].height < height_ - 10)) {
			float tmp[2];
			tmp[0] = detections[i].y + detections[i].height / 2;
			tmp[1] = (detections[i].width + detections[i].height) / 2;
			CvPoint2D32f tmp2 = cvPoint2D32f(tmp[0], tmp[1]);
			cvSeqPush(line_fit_point_seq_, &tmp2);
		}
	if (line_fit_point_seq_->total > 30) {
		cvFitLine(line_fit_point_seq_, CV_DIST_L2, 0, 0.01, 0.01, line_);
		if (display_config_)
			printf("v=(%f,%f), (x,y)=(%f,%f)\n", line_[0], line_[1], line_[2],
					line_[3]);
	}
}

void STC_MOT_Tracker::activate_status(int last_key_frame_offset,
		TrackingResult &result) {
	int ed = last_key_frame_offset;
	int st = ed - 1;
	while ((st > 0) && (!frame_[st].is_key))
		st--;
	result.clear();
	for (int offset = st; offset < ed; offset++) {
		FrameResult sgl_result;
		sgl_result.frm_id = frame_[offset].frm_id;
		for (int i = 0; i < frame_[offset].obj.size(); i++)
			if (frame_[offset].obj[i].score != KILL_THRESHOLD) {
				ObjResult sgl_obj_result;
				float w_rto = 1.0 * frame_[offset].img_w / tracker_base_size_w_;
				float h_rto = 1.0 * frame_[offset].img_h / tracker_base_size_h_;
				Rect& acc_loc = frame_[offset].obj[i].accuracy_original_location;
//				if (acc_loc.x != 0 || acc_loc.y != 0 || acc_loc.width != 0
//						|| acc_loc.height != 0) {
				sgl_obj_result.loc = acc_loc;
//				} else {
//					sgl_obj_result.loc.x = max(1,
//							(int) (w_rto * frame_[offset].obj[i].loc.x));
//					sgl_obj_result.loc.y = max(1,
//							(int) (h_rto * frame_[offset].obj[i].loc.y));
//					sgl_obj_result.loc.width = min(
//							(int) (w_rto * frame_[offset].obj[i].loc.width),
//							frame_[offset].img_w - sgl_obj_result.loc.x - 1);
//					sgl_obj_result.loc.height = min(
//							(int) (h_rto * frame_[offset].obj[i].loc.height),
//							frame_[offset].img_h - sgl_obj_result.loc.y - 1);
//				}
				sgl_obj_result.obj_id = frame_[offset].obj[i].obj_id;
				sgl_obj_result.score = frame_[offset].obj[i].vehicle_pic_score;
				sgl_obj_result.type = frame_[offset].obj[i].type;
				sgl_obj_result.sl = frame_[offset].obj[i].sl;
				sgl_obj_result.dir = frame_[offset].obj[i].dir;
				sgl_result.obj.push_back(sgl_obj_result);
			}
		result.push_back(sgl_result);
		if (display_config_)
			printf("output frame %lu result.\n", frame_[offset].frm_id);
	}
	if (st != 0)
		printf("ERROR: activate status error.\n");
	for (int offset = st; offset < ed; offset++)
		frame_.pop_front();
}

void STC_MOT_Tracker::save_matched_low_score_tracker() {
	for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size(); v_idx++)
		if ((frame_[frame_.size() - 1].obj[v_idx].score == LOW_SCORE)
				&& (match_pair_[v_idx] >= 0)
				&& (graph_[v_idx + 1][match_pair_[v_idx] + 1] > 0)) {
			frame_[frame_.size() - 1].obj[v_idx].score = HIGH_SCORE;
			unsigned long v_id = frame_[frame_.size() - 1].obj[v_idx].obj_id;
			kill_tracker(v_id); // todo ?
			int offset = frame_.size() - 2;
			while ((offset >= 0) && (!frame_[offset].is_key)) {
				int frame_vehicle_idx = find_vehicle(frame_[offset].obj, v_id);
				if (frame_vehicle_idx < 0)
					break;
				frame_[offset].obj[frame_vehicle_idx].score = HIGH_SCORE;
				offset--;
			}
		}
}

void STC_MOT_Tracker::degrade_unmatched_new_detection() {
	for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size(); v_idx++)
		if ((frame_[frame_.size() - 1].obj[v_idx].is_new)
				&& ((match_pair_[v_idx] < 0)
						|| (graph_[v_idx + 1][match_pair_[v_idx] + 1] == 0))) //todo ?
			frame_[frame_.size() - 1].obj[v_idx].score = KILL_THRESHOLD;
}

void STC_MOT_Tracker::delete_low_score_tracker() {
	for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size(); v_idx++) {
		if ((frame_[frame_.size() - 1].obj[v_idx].score == KILL_THRESHOLD)
				|| (frame_[frame_.size() - 1].obj[v_idx].score == LOW_SCORE)) {
			frame_[frame_.size() - 1].obj[v_idx].score = KILL_THRESHOLD;
			unsigned long v_id = frame_[frame_.size() - 1].obj[v_idx].obj_id;
			kill_tracker(v_id);
			int offset = frame_.size() - 2;
			while ((offset >= 0) && (!frame_[offset].is_key)) {
				int frame_vehicle_idx = find_vehicle(frame_[offset].obj, v_id);
				if (frame_vehicle_idx < 0)
					break;
				frame_[offset].obj[frame_vehicle_idx].score = KILL_THRESHOLD;
				offset--;
			}
		}
	}
}

void STC_MOT_Tracker::kill_tracker(const unsigned long v_id) {
	// kill the tracker
	int stc_idx = find_stc_idx(v_id);
	if (stc_idx < 0)
		return;
	vector<STC>::iterator stc_it = stc_.begin() + stc_idx;
	stc_.erase(stc_it);
}

void STC_MOT_Tracker::track_foreward() {
	float dx, dy;
	for (int v_idx = 0; v_idx < frame_[frame_.size() - 2].obj.size(); v_idx++) {
		Rect box(frame_[frame_.size() - 2].obj[v_idx].loc.x,
				frame_[frame_.size() - 2].obj[v_idx].loc.y,
				frame_[frame_.size() - 2].obj[v_idx].loc.width,
				frame_[frame_.size() - 2].obj[v_idx].loc.height);
		int stc_idx = find_stc_idx(frame_[frame_.size() - 2].obj[v_idx].obj_id);
		if (stc_idx < 0)
			continue;
		stc_[stc_idx].tracker.tracking(
				frame_[frame_.size() - 1].img_pyr[stc_[stc_idx].pyr_level],
				box);
		TObj new_obj;
		new_obj.accuracy_original_location = Rect(0, 0, 0, 0);
		new_obj.obj_id = frame_[frame_.size() - 2].obj[v_idx].obj_id;
		new_obj.type = frame_[frame_.size() - 2].obj[v_idx].type;
		new_obj.first_x = frame_[frame_.size() - 2].obj[v_idx].first_x;
		new_obj.first_y = frame_[frame_.size() - 2].obj[v_idx].first_y;
		new_obj.speed = frame_[frame_.size() - 2].obj[v_idx].speed;
		new_obj.loc.x = box.x;
		new_obj.loc.y = box.y;
		new_obj.loc.width = box.width;
		new_obj.loc.height = box.height;
		new_obj.is_new = false;
		new_obj.vx = frame_[frame_.size() - 2].obj[v_idx].vx;
		new_obj.vy = frame_[frame_.size() - 2].obj[v_idx].vy;
		update_vehicle_pic(new_obj);
		if (display_config_)
			printf("pre box %d %d %d %d, cur box %d %d %d %d\n",
					frame_[frame_.size() - 2].obj[v_idx].loc.x,
					frame_[frame_.size() - 2].obj[v_idx].loc.y,
					frame_[frame_.size() - 2].obj[v_idx].loc.width,
					frame_[frame_.size() - 2].obj[v_idx].loc.height,
					new_obj.loc.x, new_obj.loc.y, new_obj.loc.width,
					new_obj.loc.height);
		float dx = 0;
		float dy = 0;
		//todo ?
		int pvk_idx = find_vehicle(frame_[frame_.size() - 2].obj,
				new_obj.obj_id);
		if (pvk_idx >= 0)
			calc_img_dis(new_obj.loc,
					frame_[frame_.size() - 2].obj[pvk_idx].loc, dx, dy);
		float dv = length(dx, dy);
		if (display_config_)
			printf("pln constraint %lu, dv %f, cur %f %f , pre %f %f\n",
					frame_[frame_.size() - 2].obj[v_idx].obj_id, dv, dx, dy,
					frame_[frame_.size() - 2].obj[v_idx].vx,
					frame_[frame_.size() - 2].obj[v_idx].vy);
		float dir_angle = calc_delta(dx, dy,
				frame_[frame_.size() - 2].obj[v_idx].vx,
				frame_[frame_.size() - 2].obj[v_idx].vy);

		if ((dv > delta_v_threshold_) && (dir_angle > 100)) {
			//printf("pre v %f %f\n", frame_[frame_.size() - 2].obj[v_idx].vx,
			//		frame_[frame_.size() - 2].obj[v_idx].vy);
			//printf("cur v %f %f\n", dx, dy);
			//printf("dir_angle %f\n", dir_angle);
			if (display_config_)
				printf("pln constraint kill id %lu, dv %f\n",
						frame_[frame_.size() - 2].obj[v_idx].obj_id, dv);
			new_obj.loc.x = frame_[frame_.size() - 2].obj[v_idx].loc.x;
			new_obj.loc.y = frame_[frame_.size() - 2].obj[v_idx].loc.y;
			new_obj.loc.width = frame_[frame_.size() - 2].obj[v_idx].loc.width;
			new_obj.type = frame_[frame_.size() - 2].obj[v_idx].type;
			new_obj.loc.height =
					frame_[frame_.size() - 2].obj[v_idx].loc.height;
			new_obj.vx = 0;
			new_obj.vy = 0;
			new_obj.speed = -999;
			new_obj.sl = UNKNOWN_SPEED;
			new_obj.dir.up_down_dir = 0;
			new_obj.dir.left_right_dir = 0;
			new_obj.first_x = new_obj.loc.x + new_obj.loc.width / 2;
			new_obj.first_y = new_obj.loc.y + new_obj.loc.height / 2;
			new_obj.score = KILL_THRESHOLD;
			frame_[frame_.size() - 1].obj.push_back(new_obj);
			continue;
		}
		float x_dis = new_obj.loc.x + new_obj.loc.width / 2 - new_obj.first_x;
		float y_dis = new_obj.loc.y + new_obj.loc.height / 2 - new_obj.first_y;
		if (fabs(x_dis) > 50) {
			if (x_dis < 0)
				new_obj.dir.left_right_dir = 1;
			else
				new_obj.dir.left_right_dir = 2;
		} else
			new_obj.dir.left_right_dir = 0;
		if (fabs(y_dis) > 50) {
			if (y_dis < 0)
				new_obj.dir.up_down_dir = 1;
			else
				new_obj.dir.up_down_dir = 2;
		} else
			new_obj.dir.up_down_dir = 0;
		float v = length(dx, dy);
		if (new_obj.speed != -999)
			new_obj.speed = new_obj.speed * 0.8 + v * 0.2;
		else
			new_obj.speed = v;
		if (new_obj.speed < slow_v_threshold_)
			new_obj.sl = SLOW_SPEED;
		else if (new_obj.speed > fast_v_threshold_)
			new_obj.sl = FAST_SPEED;
		else
			new_obj.sl = MED_SPEED;
		if (is_valid_rect(box))
			new_obj.score = frame_[frame_.size() - 2].obj[v_idx].score;
		else {
			if (display_config_)
				printf("pln constraint kill id %lu\n",
						frame_[frame_.size() - 2].obj[v_idx].obj_id);
			printf("is_valid_box kill\n");
			new_obj.score = KILL_THRESHOLD;
		}
		frame_[frame_.size() - 1].obj.push_back(new_obj);
	}
}

int STC_MOT_Tracker::find_stc_idx(const unsigned long id) {
	for (int stc_idx = 0; stc_idx < stc_.size(); stc_idx++)
		if (stc_[stc_idx].id == id)
			return stc_idx;
	return -1;
}

void STC_MOT_Tracker::add_new_vehicle(vector<Rect> &det_box,
		const vector<unsigned char> &det_type,
		const vector<Rect> &accuracy_original_det_box) {
	for (int d_idx = 0; d_idx < det_box.size(); d_idx++) {
		if (is_det_matched_[d_idx] == false) {
			Rect det = det_box[d_idx];
			if (!is_valid_rect(det))
				continue;
			TObj new_obj;
			new_obj.loc = det;
			new_obj.type = det_type[d_idx];
			update_vehicle_pic(new_obj);
			new_obj.obj_id = id_count_++;
			new_obj.is_new = true;
			new_obj.score = HIGH_SCORE;
			new_obj.vx = 0;
			new_obj.vy = 0;
			new_obj.speed = -999;
			new_obj.sl = UNKNOWN_SPEED;
			new_obj.dir.up_down_dir = 0;
			new_obj.dir.left_right_dir = 0;
			new_obj.first_x = new_obj.loc.x + new_obj.loc.width / 2;
			new_obj.first_y = new_obj.loc.y + new_obj.loc.height / 2;
			new_obj.accuracy_original_location =
					accuracy_original_det_box[d_idx];
			STC new_stc;
			new_stc.id = new_obj.obj_id;
			new_stc.pyr_level = find_pyr_level(
					frame_[frame_.size() - 1].img_pyr, det);
			float resize_rto = 1.0
					* frame_[frame_.size() - 1].img_pyr[new_stc.pyr_level].cols
					/ max(frame_[frame_.size() - 1].img_pyr[0].cols, 1);
			//printf("$$ %d %d %d %d %d %d", frame_[frame_.size() - 1].img_pyr[new_stc.pyr_level].rows, frame_[frame_.size() - 1].img_pyr[new_stc.pyr_level].cols, box.x, box.y, box.width, box.height);
			new_stc.tracker.init(
					frame_[frame_.size() - 1].img_pyr[new_stc.pyr_level], det,
					resize_rto, line_);
			stc_.push_back(new_stc);
			frame_[frame_.size() - 1].obj.push_back(new_obj);
		}
	}
}

int STC_MOT_Tracker::find_pyr_level(vector<Mat> rgb_pyr, Rect box) {
	int base_size = min(box.width, box.height);
	int pyr_level = 0;
	while ((pyr_level < rgb_pyr.size() - 1)
			&& (1.0 * base_size * rgb_pyr[pyr_level + 1].cols
					/ max(rgb_pyr[0].cols, 1) > track_size_)) {
		pyr_level++;
	}
	return pyr_level;
}

int STC_MOT_Tracker::build_graph(vector<Rect> &det_box,
		const vector<unsigned char> &det_type) {
	float dx, dy;
	int num = max((int) frame_[frame_.size() - 1].obj.size(),
			(int) det_box.size());
	if (display_config_) {
		printf("track:\n");
		for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size();
				v_idx++)
			printf("%d %d %d %d\n", frame_[frame_.size() - 1].obj[v_idx].loc.x,
					frame_[frame_.size() - 1].obj[v_idx].loc.y,
					frame_[frame_.size() - 1].obj[v_idx].loc.width,
					frame_[frame_.size() - 1].obj[v_idx].loc.height);
		printf("det_box:\n");
		for (int v_idx = 0; v_idx < det_box.size(); v_idx++)
			printf("%d %d %d %d\n", det_box[v_idx].x, det_box[v_idx].y,
					det_box[v_idx].width, det_box[v_idx].height);
	}
	for (int v_idx = 0; v_idx < num; v_idx++)
		for (int d_idx = 0; d_idx < num; d_idx++) {
			if ((v_idx >= frame_[frame_.size() - 1].obj.size())
					|| (d_idx >= det_box.size())
					|| (frame_[frame_.size() - 1].obj[v_idx].type
							!= det_type[d_idx])) {
				graph_[v_idx + 1][d_idx + 1] = 0;
				continue;
			}
			float intersection = calc_intersection(
					frame_[frame_.size() - 1].obj[v_idx].loc, det_box[d_idx]);
			float min_area = min(
					frame_[frame_.size() - 1].obj[v_idx].loc.width
							* frame_[frame_.size() - 1].obj[v_idx].loc.height,
					det_box[d_idx].width * det_box[d_idx].height);
			calc_img_dis(det_box[d_idx],
					frame_[frame_.size() - 1].obj[v_idx].loc, dx, dy);
			if ((length(dx, dy) > delta_v_threshold_ * 3)
					&& (intersection / min_area < 0.5)) {
				graph_[v_idx + 1][d_idx + 1] = 0;
				continue;
			}
			float size_rto = calc_size_rto(
					frame_[frame_.size() - 1].obj[v_idx].loc.height
							* frame_[frame_.size() - 1].obj[v_idx].loc.width,
					det_box[d_idx].height * det_box[d_idx].width);
			if ((intersection > min_intersection) && (size_rto < max_size_rto_))
				graph_[v_idx + 1][d_idx + 1] = intersection;

			else
				graph_[v_idx + 1][d_idx + 1] = 0;
//			if(v_idx==0){
//				cout<<"graph:"<<d_idx<<"|"<<graph_[v_idx + 1][d_idx + 1];
//			}
		}
	if (display_config_) {
		printf("graph:\n");
		for (int v_idx = 0; v_idx < num; v_idx++) {
			for (int d_idx = 0; d_idx < num; d_idx++)
				printf("%f ", graph_[v_idx + 1][d_idx + 1]);
			printf("\n");
		}
	}
	return num;
}

float STC_MOT_Tracker::calc_intersection(const Rect &b1, const Rect &b2) {
	int up_left_x = max(b1.x, b2.x);
	int up_left_y = max(b1.y, b2.y);
	int low_right_x = min(b1.x + b1.width, b2.x + b2.width);
	int low_right_y = min(b1.y + b1.height, b2.y + b2.height);
	if ((low_right_x < up_left_x) || (low_right_y < up_left_y))
		return 0;
	else
		return ((low_right_x - up_left_x) * (low_right_y - up_left_y));
}

bool STC_MOT_Tracker::is_valid_rect(Rect &b) {
	b.width = min(b.x + b.width, width_ - 1);
	b.height = min(b.y + b.height, height_ - 1);
	b.x = max(b.x, 0);
	b.y = max(b.y, 0);
	b.width = b.width - b.x;
	b.height = b.height - b.y;
	if ((b.width < min_box_size_) || (b.height < min_box_size_)
			|| calc_size_rto(b.width, b.height) > max_wh_rto_)
		return false;
	if (is_in_disable_area(b))
		return false;
	return true;
}

bool STC_MOT_Tracker::is_in_disable_area(Rect &b) {
	for (int i = 0; i < disable_area_.size(); i++) {
		float b_size = b.width * b.height;
		float disable_area_size = disable_area_[i].width
				* disable_area_[i].height;
		if (calc_intersection(b, disable_area_[i])
				/ max(1.0, min(b_size, disable_area_size)) > 0.1)
			return true;
	}
	return false;
}

int STC_MOT_Tracker::find_last_key_frame() {
	int idx = frame_.size() - 2;
	while ((idx >= 0) && (!frame_[idx].is_key))
		idx--;
	return idx;
}

void STC_MOT_Tracker::update_vehicle_location(const int &last_key_frame_offset,
		const vector<Rect> &det_box,
		const vector<Rect> &accuracy_original_det_box) {
	for (int v_idx = 0; v_idx < frame_[frame_.size() - 1].obj.size(); v_idx++) {
		if ((match_pair_[v_idx] >= 0)
				&& (graph_[v_idx + 1][match_pair_[v_idx] + 1] > 0)) {
			smooth_update_vehicle_location(last_key_frame_offset,
					frame_[frame_.size() - 1].obj[v_idx].obj_id,
					det_box[match_pair_[v_idx]],accuracy_original_det_box[match_pair_[v_idx]]);
			frame_[frame_.size() - 1].obj[v_idx].score = HIGH_SCORE;
			float dx = 0;
			float dy = 0;
			if (last_key_frame_offset >= 0) {
				int pv_idx = find_vehicle(frame_[last_key_frame_offset].obj,
						frame_[frame_.size() - 1].obj[v_idx].obj_id);
				if (pv_idx >= 0)
					calc_img_dis(frame_[frame_.size() - 1].obj[v_idx].loc,
							frame_[last_key_frame_offset].obj[pv_idx].loc, dx,
							dy);
			}
			frame_[frame_.size() - 1].obj[v_idx].vx = dx;
			frame_[frame_.size() - 1].obj[v_idx].vy = dy;
			is_det_matched_[match_pair_[v_idx]] = true;
			int stc_idx = find_stc_idx(
					frame_[frame_.size() - 1].obj[v_idx].obj_id);
			frame_[frame_.size() - 1].obj[v_idx].accuracy_original_location =
					accuracy_original_det_box[match_pair_[v_idx]];
			Rect box(det_box[match_pair_[v_idx]].x,
					det_box[match_pair_[v_idx]].y,
					det_box[match_pair_[v_idx]].width,
					det_box[match_pair_[v_idx]].height);
			stc_[stc_idx].pyr_level = find_pyr_level(
					frame_[frame_.size() - 1].img_pyr,
					det_box[match_pair_[v_idx]]);
			float resize_rto =
					static_cast<float>(1.0
							* frame_[frame_.size() - 1].img_pyr[stc_[stc_idx].pyr_level].cols
							/ max(frame_[frame_.size() - 1].img_pyr[0].cols, 1));
			stc_[stc_idx].tracker.init(
					frame_[frame_.size() - 1].img_pyr[stc_[stc_idx].pyr_level],
					box, resize_rto, line_);
		} else if (frame_[frame_.size() - 1].obj[v_idx].score == HIGH_SCORE)
			frame_[frame_.size() - 1].obj[v_idx].score = LOW_SCORE;
		else
			frame_[frame_.size() - 1].obj[v_idx].score = KILL_THRESHOLD;
	}
}

void STC_MOT_Tracker::smooth_update_vehicle_location(
		const int &last_key_frame_offset, const unsigned long &v_id,
		const Rect &new_loc, const Rect &accuracy_new_loc) {
	int pre_v_idx = find_vehicle(frame_[last_key_frame_offset].obj, v_id);
	int det_fps = frame_.size() - last_key_frame_offset - 1;

	{
		float up_left_offset_x = static_cast<float>(1.0
				* (new_loc.x
						- frame_[last_key_frame_offset].obj[pre_v_idx].loc.x)
				/ det_fps);
		float up_left_offset_y = static_cast<float>(1.0
				* (new_loc.y
						- frame_[last_key_frame_offset].obj[pre_v_idx].loc.y)
				/ det_fps);
		float low_right_offset_x =
				static_cast<float>(1.0
						* (new_loc.x + new_loc.width
								- frame_[last_key_frame_offset].obj[pre_v_idx].loc.x
								- frame_[last_key_frame_offset].obj[pre_v_idx].loc.width)
						/ det_fps);
		float low_right_offset_y =
				static_cast<float>(1.0
						* (new_loc.y + new_loc.height
								- frame_[last_key_frame_offset].obj[pre_v_idx].loc.y
								- frame_[last_key_frame_offset].obj[pre_v_idx].loc.height)
						/ det_fps);
		for (int frame_offset = frame_.size() - 1;
				frame_offset > last_key_frame_offset; frame_offset--) {
			int times = frame_offset - last_key_frame_offset;
			int v_idx = find_vehicle(frame_[frame_offset].obj, v_id);
			if (v_idx >= 0) {
				frame_[frame_offset].obj[v_idx].loc.width =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].loc.x
								+ frame_[last_key_frame_offset].obj[pre_v_idx].loc.width
								+ low_right_offset_x * times);
				frame_[frame_offset].obj[v_idx].loc.height =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].loc.y
								+ frame_[last_key_frame_offset].obj[pre_v_idx].loc.height
								+ low_right_offset_y * times);
				frame_[frame_offset].obj[v_idx].loc.x =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].loc.x
								+ up_left_offset_x * times);
				frame_[frame_offset].obj[v_idx].loc.y =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].loc.y
								+ up_left_offset_y * times);
				frame_[frame_offset].obj[v_idx].loc.width =
						frame_[frame_offset].obj[v_idx].loc.width
								- frame_[frame_offset].obj[v_idx].loc.x;
				frame_[frame_offset].obj[v_idx].loc.height =
						frame_[frame_offset].obj[v_idx].loc.height
								- frame_[frame_offset].obj[v_idx].loc.y;
				update_vehicle_pic(frame_[frame_offset].obj[v_idx]);
			}
		}
	}

	{
		float up_left_offset_x =
				static_cast<float>(1.0
						* (accuracy_new_loc.x
								- frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.x)
						/ det_fps);
		float up_left_offset_y =
				static_cast<float>(1.0
						* (accuracy_new_loc.y
								- frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.y)
						/ det_fps);
		float low_right_offset_x =
				static_cast<float>(1.0
						* (accuracy_new_loc.x + accuracy_new_loc.width
								- frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.x
								- frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.width)
						/ det_fps);
		float low_right_offset_y =
				static_cast<float>(1.0
						* (accuracy_new_loc.y + accuracy_new_loc.height
								- frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.y
								- frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.height)
						/ det_fps);
		for (int frame_offset = frame_.size() - 1;
				frame_offset > last_key_frame_offset; frame_offset--) {
			int times = frame_offset - last_key_frame_offset;
			int v_idx = find_vehicle(frame_[frame_offset].obj, v_id);
			if (v_idx >= 0) {
				frame_[frame_offset].obj[v_idx].accuracy_original_location.width =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.x
								+ frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.width
								+ low_right_offset_x * times);
				frame_[frame_offset].obj[v_idx].accuracy_original_location.height =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.y
								+ frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.height
								+ low_right_offset_y * times);
				frame_[frame_offset].obj[v_idx].accuracy_original_location.x =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.x
								+ up_left_offset_x * times);
				frame_[frame_offset].obj[v_idx].accuracy_original_location.y =
						static_cast<int>(frame_[last_key_frame_offset].obj[pre_v_idx].accuracy_original_location.y
								+ up_left_offset_y * times);
				frame_[frame_offset].obj[v_idx].accuracy_original_location.width =
						frame_[frame_offset].obj[v_idx].accuracy_original_location.width
								- frame_[frame_offset].obj[v_idx].accuracy_original_location.x;
				frame_[frame_offset].obj[v_idx].accuracy_original_location.height =
						frame_[frame_offset].obj[v_idx].accuracy_original_location.height
								- frame_[frame_offset].obj[v_idx].accuracy_original_location.y;
				update_vehicle_pic(frame_[frame_offset].obj[v_idx]);
			}
		}
	}
}

int STC_MOT_Tracker::find_vehicle(const vector<TObj> &obj,
		const unsigned long v_id) {
	for (int v_idx = 0; v_idx < obj.size(); v_idx++)
		if (obj[v_idx].obj_id == v_id)
			return v_idx;
	return -1;
}

void STC_MOT_Tracker::update_vehicle_pic(TObj &sgl_obj) {
	float area_size = sgl_obj.loc.width * sgl_obj.loc.height;
	if ((area_size > 0) && (sgl_obj.loc.x > width_ * 0.05)
			&& (sgl_obj.loc.y > height_ * 0.05)
			&& (sgl_obj.loc.x + sgl_obj.loc.width < width_ * 0.95)
			&& (sgl_obj.loc.y + sgl_obj.loc.height < height_ * 0.95))
		sgl_obj.vehicle_pic_score = 10000 + area_size;
	else
		sgl_obj.vehicle_pic_score = area_size;
}

float STC_MOT_Tracker::calc_delta(float &x1, float &y1, float &x2, float &y2) {
	float a1 = calc_angle(x1, y1);
	float a2 = calc_angle(x2, y2);
	if ((a1 == -999) || (a2 == -999))
		return 0;
	float delta = abs(a1 - a2);
	delta = min(delta, 360 - delta);
	return delta;
}

float STC_MOT_Tracker::calc_angle(float &x, float &y) {
	float vec_norm = sqrt(x * x + y * y);
	if (vec_norm == 0) {
		return -999;
	}
	float angle = acos(x / vec_norm) / pi_ * 180;
	if (y < -1e-6)
		angle = 360 - angle;
	return angle;
}

float STC_MOT_Tracker::calc_img_dis(const Rect b1, const Rect b2, float &dx,
		float &dy) {
	float img_pnt1[3] = { float(b1.x + 0.5 * b1.width), float(
			b1.y + 0.5 * b1.height), 1 };
	float img_pnt2[3] = { float(b2.x + 0.5 * b2.width), float(
			b2.y + 0.5 * b2.height), 1 };
	dx = img_pnt1[0] - img_pnt2[0];
	dy = img_pnt1[1] - img_pnt2[1];
}

int STC_MOT_Tracker::max(int i, int j) {
	if (i > j)
		return i;
	else
		return j;
}

float STC_MOT_Tracker::max(float i, float j) {
	if (i > j)
		return i;
	else
		return j;
}

int STC_MOT_Tracker::min(int i, int j) {
	if (i < j)
		return i;
	else
		return j;
}

float STC_MOT_Tracker::min(float i, float j) {
	if (i < j)
		return i;
	else
		return j;
}

float STC_MOT_Tracker::calc_size_rto(int i, int j) {
	return 1.0 * max(i, j) / max(min(i, j), 1);
}

float STC_MOT_Tracker::length(float i, float j) {
	return sqrt(i * i + j * j);
}

// -------------------------------------- Binary Matching -----------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
bool BinaryMatching::DFS(const int x,
		const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM]) {
	visx_[x] = true;
	for (int y = 1; y <= n_; y++) {
		if (visy_[y])
			continue;
		int t = lx_[x] + ly_[y] - w[x][y];
		if (t == 0) {
			visy_[y] = true;
			if (link_[y] == -1 || DFS(link_[y], w)) {
				link_[y] = x;
				return true;
			}
		} else if (slack_[y] > t)
			slack_[y] = t;
	}
	return false;
}
float BinaryMatching::match(const int n,
		const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM],
		int inv_link[MAX_MATCHING_NUM]) {
	n_ = n;
	for (int i = 1; i <= n_; i++) {
		lx_[i] = -INF;
		ly_[i] = 0;
		link_[i] = -1;
		inv_link[i - 1] = -1;
		for (int j = 1; j <= n_; j++)
			if (w[i][j] > lx_[i])
				lx_[i] = w[i][j];
	}
	for (int x = 1; x <= n_; x++) {
		for (int i = 1; i <= n_; i++)
			slack_[i] = INF;
		while (1) {
			for (int i = 1; i <= n_; i++)
				visx_[i] = visy_[i] = false;
			if (DFS(x, w))
				break;
			int d = INF;
			for (int i = 1; i <= n_; i++)
				if (!visy_[i] && d > slack_[i])
					d = slack_[i];
			for (int i = 1; i <= n_; i++) {
				if (visx_[i])
					lx_[i] -= d;
				if (visy_[i])
					ly_[i] += d;
				else
					slack_[i] -= d;
			}
		}
	}
	float res = 0;
	for (int i = 1; i <= n_; i++)
		if (link_[i] > -1) {
			res += w[link_[i]][i];
			inv_link[link_[i] - 1] = i - 1;
		}
	return res;
}
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------

// -------------------------------------- STC Tracker -----------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
STCTracker::STCTracker() {
}
STCTracker::~STCTracker() {
}
// Create a Hamming window
void STCTracker::createHammingWin() {
	for (int i = 0; i < hammingWin.rows; i++) {
		for (int j = 0; j < hammingWin.cols; j++) {
			hammingWin.at<double>(i, j) = (0.54
					- 0.46 * cos(2 * CV_PI * i / hammingWin.rows))
					* (0.54 - 0.46 * cos(2 * CV_PI * j / hammingWin.cols));
		}
	}
}
// Define two complex-value operation
void STCTracker::complexOperation(const Mat src1, const Mat src2, Mat &dst,
		int flag) {
	CV_Assert(src1.size == src2.size);
	CV_Assert(src1.channels() == 2);

	Mat A_Real, A_Imag, B_Real, B_Imag, R_Real, R_Imag;
	vector<Mat> planes;
	split(src1, planes);
	planes[0].copyTo(A_Real);
	planes[1].copyTo(A_Imag);

	split(src2, planes);
	planes[0].copyTo(B_Real);
	planes[1].copyTo(B_Imag);

	dst.create(src1.rows, src1.cols, CV_64FC2);
	split(dst, planes);
	R_Real = planes[0];
	R_Imag = planes[1];

//	for (int i = 0; i < A_Real.rows; i++) {
//		for (int j = 0; j < A_Real.cols; j++) {
//			double a = A_Real.at<double>(i, j);
//			double b = A_Imag.at<double>(i, j);
//			double c = B_Real.at<double>(i, j);
//			double d = B_Imag.at<double>(i, j);
//
//			if (flag) {
//				// division: (a+bj) / (c+dj)
//				R_Real.at<double>(i, j) = (a * c + b * d)
//						/ (c * c + d * d + 0.000001);
//				R_Imag.at<double>(i, j) = (b * c - a * d)
//						/ (c * c + d * d + 0.000001);
//			} else {
//				// multiplication: (a+bj) * (c+dj)
//				R_Real.at<double>(i, j) = a * c - b * d;
//				R_Imag.at<double>(i, j) = b * c + a * d;
//			}
//
//		}
//	}
	if (flag) {
		Mat dibu = double(1)
				/ (B_Real.mul(B_Real) + B_Imag.mul(B_Imag) + 0.000001);
		R_Real = (A_Real.mul(B_Real) + A_Imag.mul(B_Imag)).mul(dibu);
		R_Imag = (A_Imag.mul(B_Real) - A_Real.mul(B_Imag)).mul(dibu);
	} else {
		R_Real = A_Real.mul(B_Real) - A_Imag.mul(B_Imag);
		R_Imag = A_Imag.mul(B_Real) + A_Real.mul(B_Imag);
	}
//	for (int i = 0; i < A_Real.rows; i++) {
//		for (int j = 0; j < A_Real.cols; j++) {
//			testfile << R_Real.at<double>(i, j) << endl;
//		}
//	}

	merge(planes, dst);
}
// Get context prior and posterior probability
void STCTracker::getCxtPriorPosteriorModel(const Mat image) {
	//cout<<"cxtPriorPro "<<cxtPriorPro.rows<<" "<<cxtPriorPro.cols<<endl;
	//cout<<"img"<<image.rows<<" "<<image.cols<<endl;
	CV_Assert(image.size == cxtPriorPro.size);

	double sum_prior(0), sum_post(0);
	for (int i = 0; i < cxtRegion.height; i++) {
		for (int j = 0; j < cxtRegion.width; j++) {
			double x = j + cxtRegion.x;
			double y = i + cxtRegion.y;
			double dist = sqrt(
					(center.x - x) * (center.x - x)
							+ (center.y - y) * (center.y - y));

			// equation (5) in the paper
			cxtPriorPro.at<double>(i, j) = exp(
					-dist * dist / (2 * sigma * sigma));
			sum_prior += cxtPriorPro.at<double>(i, j);

			// equation (6) in the paper
			cxtPosteriorPro.at<double>(i, j) = exp(
					-pow(dist / sqrt(alpha), beta));
			sum_post += cxtPosteriorPro.at<double>(i, j);
		}
	}
	cxtPriorPro.convertTo(cxtPriorPro, -1, 1.0 / sum_prior);
	cxtPriorPro = cxtPriorPro.mul(image);
	cxtPosteriorPro.convertTo(cxtPosteriorPro, -1, 1.0 / sum_post);
}

void STCTracker::FDFT_inplace(Mat& matin) {
	dft(matin, matin);
}
void STCTracker::BDFT(Mat& matin, Mat& matout) {
	dft(matin, matout, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
}

//void STCTracker::FDFT_inplace(Mat& matin){
//
//
//	fftw_complex *in;
//	fftw_plan p;
//
////	Mat A_single;
////	matin.convertTo(A_single,CV_32FC2,1,0);
//
//	in = (fftw_complex*) matin.data;
//
//	p = fftw_plan_dft_2d(matin.rows, matin.cols, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
//	fftw_execute(p);
////	A_single.convertTo(matin,CV_64FC2,1,0);
//	fftw_destroy_plan(p);
//}
//
//
//void STCTracker::BDFT(Mat& matin,Mat& matout){
//	fftw_complex *in,*out;
//	fftw_plan p;
////	Mat in_single_mat,
//	Mat out_single_mat;
////	matin.convertTo(in_single_mat,CV_32FC2,1,0);
//	out_single_mat=Mat::zeros(matin.size(),CV_64FC2);
//
//	in = (fftw_complex*) matin.data;
//	out= (fftw_complex*) out_single_mat.data;
//	p = fftw_plan_dft_2d(matin.rows, matin.cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
//	fftw_execute(p);
//
//	vector < Mat > planes;
//	split(out_single_mat, planes);
//	planes[0].convertTo(matout,CV_64FC1,1.0/matin.rows/matin.cols,0);
//
//	fftw_destroy_plan(p);
//}

// Learn Spatio-Temporal Context Model
void STCTracker::learnSTCModel(const Mat image) {
	t_profiler_2.reset();

	// step 1: Get context prior and posterior probability
	getCxtPriorPosteriorModel(image);
//	cout<<"cxtPriorPro.size()"<<cxtPriorPro.size().height<<" "<<cxtPriorPro.size().width<<endl;
	prof_str = "getModel";
	t_profiler_2.update(prof_str);

	// step 2-1: Execute 2D DFT for prior probability
	Mat priorFourier;
	Mat planes1[] = { cxtPriorPro, Mat::zeros(cxtPriorPro.size(), CV_64F) };
	merge(planes1, 2, priorFourier);
	prof_str = "Merge";
	t_profiler_2.update(prof_str);

	FDFT_inplace(priorFourier);
	prof_str = "First Dft";
	t_profiler_2.update(prof_str);
	// step 2-2: Execute 2D DFT for posterior probability
	Mat postFourier;
	Mat planes2[] = { cxtPosteriorPro, Mat::zeros(cxtPosteriorPro.size(),
	CV_64F) };
	merge(planes2, 2, postFourier);
	FDFT_inplace(postFourier);
	prof_str = "Second Dft";
	t_profiler_2.update(prof_str);
	// step 3: Calculate the division
	Mat conditionalFourier;
	complexOperation(postFourier, priorFourier, conditionalFourier, 1);

	// step 4: Execute 2D inverse DFT for conditional probability and we obtain STCModel
	BDFT(conditionalFourier, STModel);
	prof_str = "Three Dft";
	t_profiler_2.update(prof_str);
	// step 5: Use the learned spatial context model to update spatio-temporal context model
	addWeighted(STCModel, 1.0 - rho, STModel, rho, 0.0, STCModel);
	prof_str = "Weight done";
	t_profiler_2.update(prof_str);
//	if (profile_time_)
//	LOG(INFO) << t_profiler_2.getSmoothedTimeProfileString();
}
// Initialize the hyper parameters and models
void STCTracker::init(const Mat frame, const Rect inbox, const float resize_rto,
		const float *line) {
	Rect box;
//	printf("+%d %d %d %d\n", inbox.x, inbox.y, inbox.width, inbox.height);
	box.x = inbox.x * resize_rto;
	box.y = inbox.y * resize_rto;
	box.width = inbox.width * resize_rto;
	box.height = inbox.height * resize_rto;

	Rect boxRegion;
	FrameNum = 1;
	resize_rto_ = resize_rto;
	// initial some parameters
	padding = 1;
	num = 5;         //num consecutive frames
	alpha = 2.25;         //parameter \alpha in Eq.(6)
	beta = 1;		 //Eq.(6)
	rho = 0.075;		 //learning parameter \rho in Eq.(12)
	lambda = 0.25;
	sigma = 0.5 * (box.width + box.height);		 //sigma init
	scale = 1.0;
	sigma = sigma * scale;

	// the object position
	center.x = box.x + 0.5 * box.width;
	center.y = box.y + 0.5 * box.height;

	// the context region
	cxtRegion.width = (1 + padding) * box.width;
	cxtRegion.height = (1 + padding) * box.height;
	cxtRegion.x = center.x - cxtRegion.width * 0.5;
	cxtRegion.y = center.y - cxtRegion.height * 0.5;
	cxtRegion &= Rect(0, 0, frame.cols, frame.rows);
	boxRegion = cxtRegion;		 //output box region

	// the prior, posterior and conditional probability and spatio-temporal context model
	cxtPriorPro = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	cxtPosteriorPro = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	STModel = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	STCModel = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);

	// create a Hamming window
	hammingWin = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	createHammingWin();

	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);

	// normalized by subtracting the average intensity of that region
	Scalar average = mean(gray(cxtRegion));
	Mat context;
	gray(cxtRegion).convertTo(context, CV_64FC1, 1.0, -average[0]);

	// multiplies a Hamming window to reduce the frequency effect of image boundary
	context = context.mul(hammingWin);

	// learn Spatio-Temporal context model from first frame
	learnSTCModel(context);
//	printf("-%d %d %d %d\n", inbox.x, inbox.y, inbox.width, inbox.height);
	box_w = box.width;
	box_h = box.height;
	box_init_y = box.y + box.height / 2;
	if ((line[0] != 0) && (box.y > frame.rows * 0.04)
			&& (box.y + box.height < frame.rows * 0.96)) {
		line_k = line[1] / line[0];
		line_b = line[3] * resize_rto - line[2] * resize_rto * line_k;
	} else {
		line_k = 0;
		line_b = 0;
	}
}
// STCTracker: calculate the confidence map and find the max position
void STCTracker::tracking(const Mat frame, Rect &in_out_trackBox) {
	Rect trackBox;
	trackBox.x = in_out_trackBox.x * resize_rto_;
	trackBox.y = in_out_trackBox.y * resize_rto_;
	trackBox.width = in_out_trackBox.width * resize_rto_;
	trackBox.height = in_out_trackBox.height * resize_rto_;
	Rect boxRegion;
	FrameNum++;
	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);

	// normalized by subtracting the average intensity of that region
	Scalar average = mean(gray(cxtRegion));
	Mat context;
	gray(cxtRegion).convertTo(context, CV_64FC1, 1.0, -average[0]);

	// multiplies a Hamming window to reduce the frequency effect of image boundary
	context = context.mul(hammingWin);

	// step 1: Get context prior probability
	//cout<<"context "<<context.rows<<" "<<context.cols<<endl;
	getCxtPriorPosteriorModel(context);

	// step 2-1: Execute 2D DFT for prior probability
	Mat priorFourier;
//	cout<<"box"<<cxtPriorPro.rows<<" "<<cxtPriorPro.cols<<endl;
	Mat planes1[] = { cxtPriorPro, Mat::zeros(cxtPriorPro.size(), CV_64F) };
	merge(planes1, 2, priorFourier);
	FDFT_inplace(priorFourier);

	// step 2-2: Execute 2D DFT for conditional probability
	Mat STCModelFourier;
	Mat planes2[] = { STCModel, Mat::zeros(STCModel.size(), CV_64F) };
	merge(planes2, 2, STCModelFourier);

//	dft(STCModelFourier,STCModelFourier);
	FDFT_inplace(STCModelFourier);

	// step 3: Calculate the multiplication
	Mat postFourier;
	complexOperation(STCModelFourier, priorFourier, postFourier, 0);

	// step 4: Execute 2D inverse DFT for posterior probability namely confidence map
	Mat confidenceMap;
	BDFT(postFourier, confidenceMap);

	// step 5: Find the max position
	Point point;
	double maxVal;
	minMaxLoc(confidenceMap, 0, &maxVal, 0, &point);
	maxValue.push_back(maxVal);

	/***********update scale by Eq.(15)**********/
	if (FrameNum % (num + 2) == 0) {
		double scale_curr = 0.0;

		for (int k = 0; k < num; k++) {
			scale_curr += sqrt(
					maxValue[FrameNum - k - 2] / maxValue[FrameNum - k - 3]);
		}

		scale = (1 - lambda) * scale + lambda * (scale_curr / num);
		sigma = sigma * scale;

	}
	// step 6-1: update center, trackBox and context region
	center.x = cxtRegion.x + point.x;
	center.y = cxtRegion.y + point.y;
	trackBox.x = center.x - 0.5 * trackBox.width;
	trackBox.y = center.y - 0.5 * trackBox.height;
	trackBox &= Rect(0, 0, frame.cols, frame.rows);
	//boundary
	cxtRegion.x = center.x - cxtRegion.width * 0.5;
	if (cxtRegion.x < 0) {
		cxtRegion.x = 0;
	}
	cxtRegion.y = center.y - cxtRegion.height * 0.5;
	if (cxtRegion.y < 0) {
		cxtRegion.y = 0;
	}
	if (cxtRegion.x + cxtRegion.width > frame.cols) {
		cxtRegion.x = frame.cols - cxtRegion.width;
	}
	if (cxtRegion.y + cxtRegion.height > frame.rows) {
		cxtRegion.y = frame.rows - cxtRegion.height;
	}

	//cout<<"cxtRegionXY"<<cxtRegion.x<<" "<<cxtRegion.y<<endl;
	//cout<<"cxtRegion"<<cxtRegion.height<<" "<<cxtRegion.width<<endl;
	//cout<<"frame"<<frame.rows<<" "<<frame.cols<<endl;

	//cxtRegion &= Rect(0, 0, frame.cols, frame.rows);
	//cout<<"cxtRegionXY"<<cxtRegion.x<<" "<<cxtRegion.y<<endl;
	//cout<<"cxtRegion"<<cxtRegion.height<<" "<<cxtRegion.width<<endl;

	boxRegion = cxtRegion;
	// step 7: learn Spatio-Temporal context model from this frame for tracking next frame
	average = mean(gray(cxtRegion));
	//cout<<"cxtRegion"<<cxtRegion.height<<" "<<cxtRegion.width<<endl;

	gray(cxtRegion).convertTo(context, CV_64FC1, 1.0, -average[0]);

	//cout<<"hamm"<<hammingWin.rows<<" "<<hammingWin.cols<<endl;

	context = context.mul(hammingWin);
	learnSTCModel(context);
	float new_box_w;
	float new_box_h;
	if (line_k != 0) {
		float box_rto = linefit(center.y) / linefit(box_init_y);
		new_box_w = box_w * box_rto;
		new_box_h = box_h * box_rto;
	} else {
		new_box_w = box_w;
		new_box_h = box_h;
	}
	in_out_trackBox.x = (center.x - 0.5 * new_box_w) / resize_rto_;
	in_out_trackBox.y = (center.y - 0.5 * new_box_h) / resize_rto_;
	in_out_trackBox.width = new_box_w / resize_rto_;
	in_out_trackBox.height = new_box_h / resize_rto_;
}
float STCTracker::linefit(const int &y) {
	float tmp = line_k * y + line_b;
	if (tmp < 2)
		return 2;
	else
		return tmp;
}
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------

