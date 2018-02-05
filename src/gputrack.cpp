#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <map>
#include "gputracker/gpustc_mot_tracker.h"

using namespace std;
using namespace cv;

typedef struct detect_object_t {
	float confidence;
	int typeID;
	Rect position;
	void readFrom(istream& in) {
		in >> confidence >> typeID >> position.x >> position.y >> position.width
				>> position.height;
	}
	void writeTo(ostream& out) {
		out << " " << confidence << " " << typeID << " " << position.x << " "
				<< position.y << " " << position.width << " "
				<< position.height;

	}
} DetectObject;

void read_detection_result(string filename,
		map<int, vector<DetectObject> >& detect_result) {
	ifstream in(filename.c_str());
	while (!in.eof()) {
		int frameId, objectNum;
		in >> frameId >> objectNum;
		for (int i = 0; i < objectNum; i++) {
			DetectObject detectobj;
			detectobj.readFrom(in);
			detect_result[frameId].push_back(detectobj);
		}
	}
}

int main(int argn, char **arg) {
	unsigned long SKIPED_FRAME = 1000;
	unsigned long frame_cnt = 1000;

	STC_MOT_Tracker *tracker;
	const unsigned char LabelColors[][3] = { { 251, 144, 17 }, { 2, 224, 17 }, {
			247, 13, 145 }, { 206, 36, 255 }, { 0, 78, 255 } };
	timing_profiler t_profiler_;
	string t_profiler_str_;
	map<int, vector<DetectObject> > detect_result;
	read_detection_result("result_detect", detect_result);

	tracker = new STC_MOT_Tracker();
	// read video
	VideoCapture capture("C001.mp4");
	if (!capture.isOpened())
		printf("Error: can not open video.\n");
	unsigned long tot_frm_num = capture.get(CV_CAP_PROP_FRAME_COUNT);

	capture.set(CV_CAP_PROP_POS_FRAMES, SKIPED_FRAME);
	int fps = 15; //capture.get(CV_CAP_PROP_FPS) / 2;
	printf("The input video has %ld frames.\n", tot_frm_num);
	printf("The frame fps is %d fps.\n", fps);
	bool stop = false;
	Mat frame;
	namedWindow("Tracking Debug");
	int delay = 1;
	deque<Mat> all_imgs;
	bool display = false;
	ofstream timelog("time_time.log");
	t_profiler_.reset();
	while (!stop) {
		if (!capture.read(frame)) {
			printf("Error: can not read video frames.\n");
			return -1;
		}
		if (display) {
			all_imgs.push_back(frame.clone());
		}

		bool be_tracked;
		TrackingResult result;
		vector<Rect> pos;
		vector<unsigned char> type;

		if ((frame_cnt - 1) % fps == 0) {
			vector<DetectObject> & det_vec = detect_result[int(frame_cnt)];
			t_profiler_str_ = "track all";
			t_profiler_.update(t_profiler_str_);
			cout << frame_cnt << ":" << t_profiler_.getTimeProfileString()
					<< endl;
			cout << frame_cnt << " " << det_vec.size() << endl;

			for (int i = 0; i < det_vec.size(); i++) {
				pos.push_back(det_vec[i].position);
				type.push_back(det_vec[i].typeID);
			}
			t_profiler_.reset();

			tracker->Update(frame, frame_cnt, true, pos, type, result);
		} else {

			vector<Rect> pos_empty;
			vector<unsigned char> type_empty;
			tracker->Update(frame, frame_cnt, false, pos_empty, type_empty,
					result);

//			timelog.write("sda");
//			timelog<< frame_cnt<<":"<< t_profiler_.getTimeProfileString()<<endl;
//			cout<<"frame["<<frame_cnt<<"]"<<t_profiler_.getTimeProfileString()<<endl;
		}
		if (display) {
//			cout << "result size:" << result.size() << "  all_imgs:"
//					<< all_imgs.size() << endl;
			for (int i = 0; i < result.size(); i++) {
				char tmp_char[512];
				for (int j = 0; j < result[i].obj.size(); j++) {
					int color_idx = result[i].obj[j].obj_id % 5;

					rectangle(all_imgs[0], result[i].obj[j].loc,
							Scalar(LabelColors[color_idx][2],
									LabelColors[color_idx][1],
									LabelColors[color_idx][0]), 3, 8, 0);

					sprintf(tmp_char, "%lu type-%u score-%.0f sl-%d dir-%d-%d",
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
				if ((result[i].frm_id - 1) % fps == 0) {

					vector<DetectObject> & det_vec = detect_result[int(
							result[i].frm_id)];
					cout << "frm_id:" << det_vec.size() << endl;
					for (int j = 0; j < det_vec.size(); j++) {
						rectangle(all_imgs[0], det_vec[j].position,
								Scalar(0, 0, 255), 4, 8, 0);
					}
				}

				imshow("Tracking Debug", all_imgs[0]);
//				sprintf(tmp_char, "output_origin/%06lu.jpg", result[i].frm_id);
//				imwrite(tmp_char,all_imgs[0]);
//				cout<<frame_cnt<<"    "<<result[i].frm_id<<endl;
//					imwrite(tmp_char, all_imgs[0]);
				all_imgs.pop_front();

				int c = waitKey(delay);
				if ((char) c == 27) {
					stop = true;
				}
			}
		}
		if (frame_cnt >= tot_frm_num / 10)
			stop = true;
		frame_cnt++;
	}
	capture.release();
	delete tracker;
	return 0;
}
