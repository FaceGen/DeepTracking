#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "tracker/stc_mot_tracker.h"
#include <map>

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

typedef struct detect_object_t {
	float confidence;
	int typeID;
	Rect position;
	void readFrom(istream& in) {
		in >> confidence >> typeID >> position.x >> position.y >> position.width
				>> position.height;
//		cout<<" "<<confidence<<" "<<typeID<<" "<<position.x<<" "<<position.y<<" "<<position.width<<" "<<position.height<<endl;
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
			cout << frameId << endl;
		}
	}
}

int main(int argn, char **arg) {
	if(argn <2){
		cout<<"Need dataset num."<<endl;
		exit(0);
	}
	string test_name(arg[1]);

	string datasetname = "testn"+test_name;
	ofstream track_out_file("origin_track_n"+test_name+".log");
	DataReader* data_reader = new ImagelistDataReader("images_all/n"+test_name+".list");

	//	string mp4name = "C001";
	//	DataReader* data_reader = new VideoDataReader(datasetname + ".mp4");

	STC_MOT_Tracker *tracker;
	const unsigned char LabelColors[][3] = { { 251, 144, 17 }, { 2, 224, 17 }, {
			247, 13, 145 }, { 206, 36, 255 }, { 0, 78, 255 } };

	timing_profiler t_profiler_;
	string t_profiler_str_;
	map<int, vector<DetectObject> > detect_result;

	read_detection_result("detect_" + test_name, detect_result);


	if (!data_reader->isOpened()) {
		printf("Error: data reader can not open.\n");
		return -1;
	}

	tracker = new STC_MOT_Tracker();
	// read video
//	VideoCapture capture(datasetname + ".mp4");
//	if (!capture.isOpened())
//		printf("Error: can not open video.\n");
//	unsigned long tot_frm_num = capture.get(CV_CAP_PROP_FRAME_COUNT);
	unsigned long SKIPED_FRAME = 1;
	unsigned long frame_cnt = 1;
//	capture.set(CV_CAP_PROP_POS_FRAMES, SKIPED_FRAME);
	int fps = 15; //capture.get(CV_CAP_PROP_FPS) / 2;
//	printf("The input video has %ld frames.\n", tot_frm_num);
	printf("The frame fps is %d fps.\n", fps);
	bool stop = false;
	Mat frame;
	namedWindow("Tracking Debug");
	int delay = 100;
	deque<Mat> all_imgs;
	bool display = true;
	ofstream timelog("time_time.log");

	while (!stop) {
		frame_cnt = data_reader->readNextImage(frame);
		if (frame_cnt == -1) {
			printf("Error: can not read frames.\n");
			return -1;
		} else if (frame_cnt == -2) {
			printf("Image done.\n");
			break;
		}
//		if(frame_cnt>32)
//			break;
//		if (!capture.read(frame)) {
//			printf("Error: can not read video frames.\n");
//			return -1;
//		}
		all_imgs.push_back(frame.clone());
//		vector<Mat> imgs;
//		imgs.push_back(frame);
//		vector<vector<Rect> > pos;
//		vector<vector<float> > conf;
//		vector<vector<unsigned char> > type;
		bool be_tracked;
		TrackingResult result;
		if ((frame_cnt - 1) % fps == 0) {
			//read it
//			detector->Predict(imgs, pos, conf, type);
			vector<Rect> pos;
			vector<unsigned char> type;

			vector<DetectObject> & det_vec = detect_result[int(frame_cnt)];
			cout << frame_cnt << " " << det_vec.size() << endl;
			for (int i = 0; i < det_vec.size(); i++) {
				pos.push_back(det_vec[i].position);
				type.push_back(det_vec[i].typeID);
			}

			tracker->Update(frame, frame_cnt, true, pos, type, result);
		} else {

			t_profiler_.reset();
			vector<Rect> pos_empty;
			vector<unsigned char> type_empty;
			tracker->Update(frame, frame_cnt, false, pos_empty, type_empty,
					result);
			t_profiler_str_ = "track all";
			t_profiler_.update(t_profiler_str_);
//			timelog.write("sda");
//			timelog<< frame_cnt<<":"<< t_profiler_.getTimeProfileString()<<endl;
//			cout<<"frame["<<frame_cnt<<"]"<<t_profiler_.getTimeProfileString()<<endl;
		}
		if (display) {
			cout << "result size:" << result.size() << "  all_imgs:"
					<< all_imgs.size() << endl;
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
					track_out_file<<result[i].frm_id<<" "<<result[i].obj[j].obj_id<<" "<<int(result[i].obj[j].type)<<" "\
							<<result[i].obj[j].loc.x<<" "<<result[i].obj[j].loc.y<<" "<<result[i].obj[j].loc.width<<\
							" "<<result[i].obj[j].loc.height<<"\n";
				}

////				if (detect_result.find(result[i].frm_id)>0) {
//					vector<DetectObject> & det_vec = detect_result[int(result[i].frm_id)];
//					for (int i = 0; i < det_vec.size(); i++) {
//						int color_idx = det_vec[i].typeID % 5;
//
//						rectangle(frame, det_vec[i].position,
//								Scalar(255,255,255), 3, 8, 0);
//						sprintf(tmp_char, "%lu type-%u score-%.0f", det_vec[i].typeID,
//								det_vec[i].typeID, det_vec[i].confidence);
//						cv::putText(frame, tmp_char,
//								cv::Point(det_vec[i].position.x, det_vec[i].position.y - 12),
//								CV_FONT_HERSHEY_COMPLEX, 0.7,
//								Scalar(LabelColors[color_idx][2],
//										LabelColors[color_idx][1],
//										LabelColors[color_idx][0]), 2);
//					}
//				}


				sprintf(tmp_char, (datasetname + "_track/%06lu.jpg").c_str(),
						result[i].frm_id);
				imshow("Tracking Debug", all_imgs[0]);
//				if ((result[i].frm_id - 1) % fps == 0)
//					imwrite(tmp_char, all_imgs[0]);
				all_imgs.pop_front();
				int c = waitKey(1);
				if ((char) c == 27) {
					stop = true;
				}
			}

//			for (int i = 0; i < type[0].size(); i++)
//				if (type[0][i] == car)
//					rectangle(frame, pos[0][i], Scalar(0, 0, 255), 3, 8, 0);
//				else if (type[0][i] == person)
//					rectangle(frame, pos[0][i], Scalar(0, 255, 0), 3, 8, 0);
//				else if (type[0][i] == bicycle)
//					rectangle(frame, pos[0][i], Scalar(255, 0, 0), 3, 8, 0);
//				else if (type[0][i] == tricycle)
//					rectangle(frame, pos[0][i], Scalar(125, 125, 0), 3, 8, 0);
			cout << "frame_cnt:" << frame_cnt << endl;

		}
		frame_cnt++;
	}
//	capture.release();
	delete tracker;
	return 0;
}
