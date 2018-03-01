#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "DL_detector/detection_ssd.h"
#include <iostream>

using namespace std;
using namespace cv;

class DataReader {
public:
	DataReader() {

	}
	virtual ~DataReader() {

	}
	virtual int readNextImage(Mat& mat) {
		return 0;
	}
	virtual bool isOpened() {
		cout << "Father" << endl;
		return false;
	}
};

class VideoDataReader: public DataReader {
public:
	VideoDataReader(string videofile, int SKIPED_FRAME = 1) {
		cout << videofile << endl;
		capture.open(videofile);
		if (!capture.isOpened()) {
			printf("Error: can not open video.\n");
			return;
		}
		tot_frm_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
		capture.set(CV_CAP_PROP_POS_FRAMES, SKIPED_FRAME);
		index_frame += SKIPED_FRAME;
		openned = true;
	}
	virtual ~VideoDataReader() {
		capture.release();
	}
	virtual int readNextImage(Mat& mat) {
		if (index_frame > tot_frm_num - 1)
			return -2;
		if (!capture.read(mat))
			return -1;
		index_frame += 1;
		return index_frame - 1;
	}
	virtual bool isOpened() {
		return openned;
	}
	VideoCapture capture;
	int index_frame = 0;
	unsigned long tot_frm_num;
	bool openned = false;
};

class ImagelistDataReader: public DataReader {
public:
	ImagelistDataReader(string listfile, int SKIPED_FRAME = 0) {
		read_labeled_imagelist(listfile.c_str(), imagenames);
		tot_frm_num = imagenames.size();
		index_frame += SKIPED_FRAME;
		openned = true;
	}
	void read_labeled_imagelist(const char* filename,
			vector<string> &imagenames) {
		ifstream fcin(filename);
		if (!fcin.is_open()) {
			cout << "file not opened";
			exit(0);
		}
		string imagename;
		while (!fcin.eof()) {
			fcin >> imagename;
			imagenames.push_back(imagename);
			cout << "h::::" << imagename << " " << endl;
		}
	}

	virtual ~ImagelistDataReader() {

	}
	virtual int readNextImage(Mat& mat) {
		if (index_frame >= tot_frm_num)
			return -2;
		mat = cv::imread(imagenames[index_frame]);
		if (mat.cols <= 0)
			return -1;
		index_frame += 1;
		return index_frame - 1;
	}
	virtual bool isOpened() {
		return openned;
	}
	vector<string> imagenames;
	int index_frame;
	unsigned long tot_frm_num;
	bool openned = false;
};

int main(int argn, char **arg) {

	DetectionSSD *detector;

	const unsigned char LabelColors[][3] = { { 251, 144, 17 }, { 2, 224, 17 }, {
			247, 13, 145 }, { 206, 36, 255 }, { 0, 78, 255 } };

	timing_profiler t_profiler_;
	string t_profiler_str_;

//	if(argn <2){
//		cout<<"Need dataset num."<<endl;
//		exit(0);
//	}
	string test_name(arg[1]);

	string datasetname = "testn"+test_name;
	// read video
	DataReader* data_reader = new VideoDataReader("/home/liuhao/workspace/1_dgvehicle/LHTracking/data/"+test_name + ".mp4");
//	DataReader* data_reader = new ImagelistDataReader("images_all/n"+test_name+".list");
	if (!data_reader->isOpened()) {
		printf("Error: data reader can not open.\n");
		return -1;
	}

	// init detector
	detector = new DetectionSSD(vehicle_detection);


	unsigned long SKIPED_FRAME = 1;
	unsigned long frame_cnt = -1;

	int fps = 15; //capture.get(CV_CAP_PROP_FPS) / 2;

	printf("The frame fps is %d fps.\n", fps);
	bool stop = false;
	Mat frame;
	namedWindow("Tracking Debug");
	int delay = 1;
//	deque<Mat> all_imgs;
	bool display = true;
	ofstream timelog("time_time.log");

	ofstream detect_result_file("detect" + datasetname);
	while (!stop) {
		frame_cnt = data_reader->readNextImage(frame);
		if (frame_cnt == -1) {
			printf("Error: can not read frames.\n");
			return -1;
		} else if (frame_cnt == -2) {
			printf("Image done.\n");
			break;
		}
		char tmp_char[512];
//		all_imgs.push_back(frame.clone());
		vector<Mat> img_vec;
		vector<vector<Rect> > pos_vec;
		vector<vector<float> > conf_vec;
		vector<vector<unsigned char> > type_vec;

		if ((frame_cnt - 1) % fps == 0) {

			img_vec.clear();
			pos_vec.clear();
			conf_vec.clear();
			type_vec.clear();
			img_vec.push_back(frame);
			detector->Predict(img_vec, pos_vec, conf_vec, type_vec);
			if (!(pos_vec.size() == 1 && conf_vec.size() == 1
					&& type_vec.size() == 1)) {
				cout
						<< "The detection result size is not 1, which is not consistent with the input image size"; // @suppress("Function cannot be resolved")
				exit(0);
			}
			Mat& img = img_vec[0];
			vector<Rect> & pos = pos_vec[0];
			vector<float> & conf = conf_vec[0];
			vector<unsigned char> type = type_vec[0];

			cout << frame_cnt << " object num:" << pos.size() << endl;

			//write down
			detect_result_file << frame_cnt << " " << pos.size();
			for (int i = 0; i < pos.size(); i++) {
				detect_result_file << " " << conf[i] << " " << int(type[i])
						<< " " << pos[i].x << " " << pos[i].y << " "
						<< pos[i].width << " " << pos[i].height;
			}
			detect_result_file << endl;

			if (display) {

				for (int i = 0; i < pos.size(); i++) {
					int color_idx = type[i] % 5;

					rectangle(frame, pos[i],
							Scalar(LabelColors[color_idx][2],
									LabelColors[color_idx][1],
									LabelColors[color_idx][0]), 3, 8, 0);
					sprintf(tmp_char, "%lu type-%u score-%.0f", type[i],
							type[i], conf[i]);
					cv::putText(frame, tmp_char,
							cv::Point(pos[i].x, pos[i].y - 12),
							CV_FONT_HERSHEY_COMPLEX, 0.7,
							Scalar(LabelColors[color_idx][2],
									LabelColors[color_idx][1],
									LabelColors[color_idx][0]), 2);

//					sprintf(tmp_char, "output/%06lu.jpg", result[0].frm_id);
//					imwrite(tmp_char, frame);

				}
				imshow("Tracking Debug", frame);
				int c = waitKey(1);
				if ((char) c == 27) {
					stop = true;
				}

			}

		}

		sprintf(tmp_char, (datasetname + "/%lu.jpg").c_str(), frame_cnt);
		imwrite(tmp_char, frame);
		frame_cnt++;
	}
//	capture.release();
	delete detector;
	return 0;
}
