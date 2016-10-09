#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void output_squares(Mat src, int im_count = 1) {
  int width = src.cols, height = src.rows, dim = min(width, height);

  Mat gray(src.rows, src.cols, CV_8UC3);
  Mat bw(src.rows, src.cols, CV_8UC3);
  Mat mask(src.rows, src.cols, CV_8UC3);
  Mat dst(src.rows, src.cols, CV_8UC3);

  cvtColor(src, gray, COLOR_BGR2GRAY);
  threshold(gray, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);

  int morph_size = 30;
  Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1 ), Point(morph_size, morph_size));
  morphologyEx(bw, mask, MORPH_BLACKHAT
               , element);
  bitwise_not(mask, mask);

  int th = dim / 80, lo_bnd = dim * .8, hi_bnd = dim * 1.41;
  float sq_scale = 1.3;
  vector<Point2f> corners;
  goodFeaturesToTrack(gray, corners, 128, .1, 20, mask, 5);

  float max_count = 0;
  vector<Point2f> max_pts;
  vector<Point2f> pts;

  for (Point2f c1 : corners) {
    for (Point2f c2 : corners) {
      Point2f n = c2 - c1;
      if (norm(n) <= lo_bnd || norm(n) > hi_bnd) {
        continue;
      }

      Point2f s1((n.x + n.y) / 2, (n.y - n.x) / 2);
      Point2f s2((n.x - n.y) / 2, (n.y + n.x) / 2);

      float count = 0;
      pts.clear();

      for (float k1 = 0; k1 < 9; ++k1) {
        Point2f r1 = s1 * (k1 / 8);
        for (float k2 = 0; k2 < 9; ++k2) {
          Point2f r2 = s2 * (k2 / 8);
          Point2f pt = c1 + r1 + r2;

          bool miss = true;
          for (Point2f c3 : corners) {
            if (norm(pt - c3) <= th) {
              ++count;
              pts.push_back(c3);
              miss = false;
              break;
            }
          }
          if (miss) {
            pts.push_back(pt);
          }
        }
      }

      if (count > max_count) {
        max_count = count;
        max_pts = pts;
      }

    }
  }

  Mat cpy = src.clone();
  for (Point2f p : max_pts) {
    circle(cpy, p, 5, Scalar(255.), -1);
  }

  imshow("", cpy);
  int key = waitKey(0) & 0xff;

  if (key == ' ') {
    for (int i = 0; i < 8; ++i) { 
      for (int j = 0; j < 8; ++j) {
        int x = 9 * i + j;
        vector<Point2f> q = {max_pts[x], max_pts[x + 1], max_pts[x + 9], max_pts[x + 10]};
        RotatedRect rect = minAreaRect(q);
        rect.size = Size(rect.size.width * sq_scale, rect.size.height * sq_scale);
        Mat M, rotated, cropped;
        float angle = rect.angle;
        Size rect_size = rect.size;
        if (rect.angle < -45.) {
          angle += 90.0;
          swap(rect_size.width, rect_size.height);
        }
        M = getRotationMatrix2D(rect.center, angle, 1.0);
        warpAffine(src, rotated, M, src.size(), INTER_CUBIC);
        getRectSubPix(rotated, rect_size, rect.center, cropped);
        
        imwrite("out/squares/sq" + to_string(8 * i + j) + "_" + to_string(im_count) + ".jpg", cropped);
        /*imshow("", cropped);
        waitKey(0);*/
      }
    }
  }

  // imwrite("out/out.jpg", cpy);


}


int main(int argc, char *argv[]) {

	if (argc != 2) {
		printf("usage: duchess <image_path>\n");
		return 1;
	}

	const char *filename = argv[1];

	/*Mat src = imread(filename, 1);
	if (src.empty()) {
		cout << "Cannot open " << filename << "\n";
		return 1;
	}

  output_squares(src);*/

	VideoCapture cap;
  cap.open(string(filename));
  if (!cap.isOpened()) {
    cout << "Cannot open " << filename << "\n";
    return 1;
  }
  for (int i = 0; i < 100; ++i) {
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
      break;
    }
    output_squares(frame, i + 1);
  }

	return 0;
}

