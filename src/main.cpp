#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "DL_detector/detection_ssd.h"
#include "tracker/stc_mot_tracker.h"
#include "fftw3.h"

using namespace std;
using namespace cv;



int main(int argn, char **arg) {
	DetectionSSD *detector;
	STC_MOT_Tracker *tracker;
	const unsigned char LabelColors[][3] = { { 251, 144, 17 }, { 2, 224, 17 }, {
			247, 13, 145 }, { 206, 36, 255 }, { 0, 78, 255 } };

	timing_profiler t_profiler_;
	string t_profiler_str_;
	if (argn != 3) {
		printf(
				"ERROR: Wrong input format.\nInput Format: .build/main car/face videofile.\n");
		exit(0);
	}
	// init detector
	if (strcmp(arg[1], "car") == 0)
		detector = new DetectionSSD(vehicle_detection);
	else if (strcmp(arg[1], "face") == 0)
		detector = new DetectionSSD(face_detection);
	else {
		printf("ERROR: The first parameter should be car or face.\n");
		exit(0);
	}
	tracker = new STC_MOT_Tracker();
	// read video
	VideoCapture capture(arg[2]);
	if (!capture.isOpened())
		printf("Error: can not open video.\n");
	unsigned long tot_frm_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
	unsigned long frame_cnt = 1;
	capture.set(CV_CAP_PROP_POS_FRAMES, frame_cnt);
	int fps = 15; //capture.get(CV_CAP_PROP_FPS) / 2;
	printf("The input video has %ld frames.\n", tot_frm_num);
	printf("The frame fps is %d fps.\n", fps);
	bool stop = false;
	Mat frame;
	namedWindow("Tracking Debug");
	int delay = 1;
	deque<Mat> all_imgs;
	bool display = true;
	ofstream timelog("time_time.log");
	while (!stop) {
		if (!capture.read(frame)) {
			printf("Error: can not read video frames.\n");
			return -1;
		}
		all_imgs.push_back(frame.clone());
		vector<Mat> imgs;
		imgs.push_back(frame);
		vector<vector<Rect> > pos;
		vector<vector<float> > conf;
		vector<vector<unsigned char> > type;

		TrackingResult result;
		if ((frame_cnt - 1) % fps == 0) {
			detector->Predict(imgs, pos, conf, type);
			tracker->Update(frame, frame_cnt, true, pos[0], type[0], result);
			if (display) {
				for (int i = 0; i < result.size(); i++) {
					char tmp_char[512];
					for (int j = 0; j < result[i].obj.size(); j++) {
						int color_idx = result[i].obj[j].obj_id % 5;
						rectangle(all_imgs[0], result[i].obj[j].loc,
								Scalar(LabelColors[color_idx][2],
										LabelColors[color_idx][1],
										LabelColors[color_idx][0]), 3, 8, 0);
						sprintf(tmp_char,
								"%lu type-%u score-%.0f sl-%d dir-%d-%d",
								result[i].obj[j].obj_id, result[i].obj[j].type,
								result[i].obj[j].score, result[i].obj[j].sl,
								result[i].obj[j].dir.up_down_dir,
								result[i].obj[j].dir.left_right_dir);
						cv::putText(all_imgs[0], tmp_char,
								cv::Point(result[i].obj[j].loc.x,
										result[i].obj[j].loc.y - 12),
								CV_FONT_HERSHEY_COMPLEX, 0.7,
								Scalar(LabelColors[color_idx][2],
										LabelColors[color_idx][1],
										LabelColors[color_idx][0]), 2);
					}
					sprintf(tmp_char, "output/%06lu.jpg", result[i].frm_id);
					imwrite(tmp_char, all_imgs[0]);
					all_imgs.pop_front();
				}
			}

		} else {

			t_profiler_.reset();
			vector<Rect> pos_empty;
			vector<unsigned char> type_empty;
			tracker->Update(frame, frame_cnt, false, pos_empty, type_empty,
					result);
			t_profiler_str_ = "track all";
			t_profiler_.update(t_profiler_str_);
//			timelog.write("sda");
			timelog<< frame_cnt<<":"<< t_profiler_.getTimeProfileString()<<endl;
			LOG(INFO)<<"frame["<<frame_cnt<<"]"<<t_profiler_.getTimeProfileString()<<endl;
		}
		if (display) {
//			for (int i = 0; i < type[0].size(); i++)
//				if (type[0][i] == car)
//					rectangle(frame, pos[0][i], Scalar(0, 0, 255), 3, 8, 0);
//				else if (type[0][i] == person)
//					rectangle(frame, pos[0][i], Scalar(0, 255, 0), 3, 8, 0);
//				else if (type[0][i] == bicycle)
//					rectangle(frame, pos[0][i], Scalar(255, 0, 0), 3, 8, 0);
//				else if (type[0][i] == tricycle)
//					rectangle(frame, pos[0][i], Scalar(125, 125, 0), 3, 8, 0);
//			imshow("Tracking Debug", frame);
			int c = waitKey(delay);
			if ((char) c == 27 || frame_cnt >= tot_frm_num) {
				stop = true;
			}
		}
		frame_cnt++;
	}
	capture.release();
	delete detector;
	delete tracker;
	return 0;
}
