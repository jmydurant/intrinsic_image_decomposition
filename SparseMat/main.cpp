/************************************************************************/
/* this program is used to calculate the intrinsic image                 
   jmydurant@hotmail.com
   image name is hehe.png
*/
/************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include <Eigen/UmfPackSupport>

#pragma warning(disable:4996)

using namespace std;
using namespace cv;

int nRows, nCols, nChannels;
double eps = 0.1;

inline int ind(int i, int j) {
	return i * nCols + j;
}

inline double pp(double x) {
	return x * x;
}

void doMath() {
	Mat fuckImage = imread("hehe.png");


	Mat originalImage = fuckImage.clone(); //Mat(fuckImage, Rect(0, 0, 401, 400));
	//imshow("original", originalImage);
	//waitKey();
	//return 0;
	Mat image;
	originalImage.convertTo(image, CV_64FC3);
	image /= 255.0;


	nRows = originalImage.rows, nCols = originalImage.cols, nChannels = originalImage.channels();
	printf("row is %d col is %d\n", nRows, nCols);
	int cnt = nRows * nCols;
	Mat image_vec = Mat(nRows, nCols, CV_64FC3);
	Mat image_val = Mat(nRows, nCols, CV_64FC1);
	Mat log_I = Mat(nRows, nCols, CV_64FC1);

	int maxInd = -1;
	double maxval = -10000000.0;

	for (int i = 0; i < nRows; i++) {
		double * u = image.ptr<double>(i);
		double * vec = image_vec.ptr<double>(i);
		double * val = image_val.ptr<double>(i);
		double * log_i = log_I.ptr<double>(i);
		for (int j = 0; j < nCols; j++) {
			double len = sqrt(u[j * nChannels] * u[j * nChannels] + u[j * nChannels + 1] * u[j * nChannels + 1] + u[j * nChannels + 2] * u[j * nChannels + 2]);
			val[j] = len;
			vec[j * nChannels] = u[j * nChannels] / len;
			vec[j * nChannels + 1] = u[j * nChannels + 1] / len;
			vec[j * nChannels + 2] = u[j * nChannels + 2] / len;
			log_i[j] = log(len);
			if (maxval < log_i[j]) {
				maxval = log_i[j];
				maxInd = ind(i, j);
			}
		}
	}

	std::vector<Eigen::Triplet<double>> pool;
	Eigen::SparseMatrix<double> A = Eigen::SparseMatrix<double>(cnt, cnt);
	Eigen::VectorXd b = Eigen::VectorXd(cnt);
	Eigen::VectorXd u = Eigen::VectorXd(cnt);
	for (int i = 0; i < cnt; i++) b(i) = 0;
	int bigCnt = 0, smallCnt = 0;

	for (int i = 0; i < nRows; i++) {
		double * vec_u = image_vec.ptr<double>(i);
		double * vec_v = vec_u;
		double * log_u = log_I.ptr<double>(i);
		double * log_v = log_u;
		bool flag;
		if (i + 1 < nRows) {
			flag = true;
			vec_v = image_vec.ptr<double>(i + 1);
			log_v = log_I.ptr<double>(i + 1);
		}
		else flag = false;
		for (int j = 0; j < nCols; j++) {

			if (flag) {
				int index_i = ind(i, j), index_j = ind(i + 1, j);
				double diff = sqrt(pp(vec_u[j * nChannels] - vec_v[j * nChannels]) + pp(vec_u[j * nChannels + 1] - vec_v[j * nChannels + 1]) + pp(vec_u[j * nChannels + 2] - vec_v[j * nChannels + 2]));
				if (diff > eps) {
					bigCnt++;
					pool.push_back(Eigen::Triplet<double>(index_i, index_i, 2.0));
					pool.push_back(Eigen::Triplet<double>(index_j, index_j, 2.0));
					pool.push_back(Eigen::Triplet<double>(index_i, index_j, -2.0));
					pool.push_back(Eigen::Triplet<double>(index_j, index_i, -2.0));
				}
				else {
					smallCnt++;
					double omega = 100.0;
					double delta_I = log_u[j] - log_v[j];
					b(index_i) += 2.0 * omega * delta_I;
					b(index_j) -= 2.0 * omega * delta_I;
					pool.push_back(Eigen::Triplet<double>(index_i, index_i, 2.0 * (1 + omega)));
					pool.push_back(Eigen::Triplet<double>(index_j, index_j, 2.0 * (1 + omega)));
					pool.push_back(Eigen::Triplet<double>(index_i, index_j, -2.0 * (1 + omega)));
					pool.push_back(Eigen::Triplet<double>(index_j, index_i, -2.0 * (1 + omega)));
				}
			}
			if (j + 1 < nCols) {
				int index_i = ind(i, j), index_j = ind(i, j + 1);
				double diff = sqrt(pp(vec_u[j * nChannels] - vec_u[(j + 1) * nChannels]) + pp(vec_u[j * nChannels + 1] - vec_u[(j + 1) * nChannels + nChannels + 1]) + pp(vec_u[j * nChannels + 2] - vec_u[(j + 1) * nChannels + 2]));
				if (diff > eps) {
					pool.push_back(Eigen::Triplet<double>(index_i, index_i, 2.0));
					pool.push_back(Eigen::Triplet<double>(index_j, index_j, 2.0));
					pool.push_back(Eigen::Triplet<double>(index_i, index_j, -2.0));
					pool.push_back(Eigen::Triplet<double>(index_j, index_i, -2.0));
				}
				else {
					double omega = 100.0;
					double delta_I = log_u[j] - log_u[j + 1];
					b(index_i) += 2.0 * omega * delta_I;
					b(index_j) -= 2.0 * omega * delta_I;
					pool.push_back(Eigen::Triplet<double>(index_i, index_i, 2.0 * (1 + omega)));
					pool.push_back(Eigen::Triplet<double>(index_j, index_j, 2.0 * (1 + omega)));
					pool.push_back(Eigen::Triplet<double>(index_i, index_j, -2.0 * (1 + omega)));
					pool.push_back(Eigen::Triplet<double>(index_j, index_i, -2.0 * (1 + omega)));
				}
			}
		}
	}

	// left down brightest...

	pool.push_back(Eigen::Triplet<double>(maxInd, maxInd, 200.0));

	A.setFromTriplets(pool.begin(), pool.end());
	printf("begin to solve");
	Eigen::UmfPackLU<const Eigen::SparseMatrix<double, Eigen::ColMajor>> lu_of_A;
	lu_of_A.compute(A);
	u = lu_of_A.solve(b);

	Mat result(nRows, nCols, CV_64FC1);
	Mat colorImage(nRows, nCols, CV_64FC3);
	int nowInd = 0;
	for (int i = 0; i < nRows; i++) {
		double * res = result.ptr<double>(i);
		double * hehe = colorImage.ptr<double>(i);
		double * cao = image.ptr<double>(i);
		for (int j = 0; j < nCols; j++) {

			double val = exp(u(nowInd));
			hehe[j * nChannels] = cao[j * nChannels] / val;
			hehe[j * nChannels + 1] = cao[j * nChannels + 1] / val;
			hehe[j * nChannels + 2] = cao[j * nChannels + 2] / val;
			res[j] = val;
			//printf("i %d j %d val %.4lf\n", i, j, val);
			nowInd++;
		}
	}
	printf("now Ind is %d\n", nowInd);
	imshow("origin", originalImage);
	imshow("result", result);
	imshow("color", colorImage);
	waitKey();

	printf("finished big is %d  small is %d\n", bigCnt, smallCnt);
}

double dx[1010][1010], dy[1010][1010], dz[1010][1010];
int dh[1010][1010];
double dr[1010][1010];
double dc[1010][1010];
double Pi = 3.1415926535897932384;
void myDebug(int fuck[][1010], int _row, int _col) {
	for (int i = 0; i < _row; i++) {
		for (int j = 0; j < _col; j++) {
			cout << fuck[i][j] << " ";
		}
		cout << endl;
	}
}

void calnorm() {
	int nRows = 480, nCols = 640;
	freopen("snapshot_val.in", "r", stdin);

	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			scanf("%d", dh[i] + j);
		}
	}

	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			if (dh[i][j] == -1) {
				dr[i][j] = dc[i][j] = -1;
				continue;
			}
			double temp_r = 2.0 * dh[i][j] * tan(28.5 * Pi / 180.0);
			dr[i][j] = temp_r / 640.0;
			double temp_c = 2.0 * dh[i][j] * tan(21.5 * Pi / 180.0);
			dc[i][j] = temp_c / 480.0;
		}
	}

	Mat ans = Mat::zeros(nRows, nCols, CV_64FC3);
	memset(dx, 0, sizeof(dx));
	memset(dy, 0, sizeof(dy));
	memset(dz, 0, sizeof(dz));


	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			if (dh[i][j] == -1) continue;
			int r_state = 3, c_state = 3;
			if (j == 0 || dh[i][j - 1] == -1) r_state -= 1;
			if (j == nCols || dh[i][j + 1] == -1) r_state -= 2;
			if (r_state == 0) continue;
			if (i == 0 || dh[i - 1][j] == -1) c_state -= 1;
			if (i == nRows || dh[i + 1][j] == -1) c_state -= 2;
			if (c_state == 0) continue;
			double a1, a2, a3, b1, b2, b3;
			a2 = 0.0, b1 = 0.0;
			if (r_state == 1) {
				a1 = (dr[i][j] + dr[i][j - 1]) / 2.0;
				a3 = (dh[i][j] - dh[i][j - 1]);
			}
			else if (r_state == 2) {
				a1 = (dr[i][j] + dr[i][j + 1]) / 2.0;
				a3 = (dh[i][j + 1] - dh[i][j]);
			}
			else {
				a1 = (dr[i][j - 1] + 2.0 * dr[i][j] + dr[i][j + 1]) / 4.0;
				a3 = (dh[i][j + 1] - dh[i][j - 1]) / 2.0;
			}
			if (c_state == 1) {
				b2 = (dc[i][j] + dc[i - 1][j]) / 2.0;
				b3 = (dh[i][j] - dh[i - 1][j]);
			}
			else if (c_state == 2) {
				b2 = (dc[i][j] + dc[i + 1][j]) / 2.0;
				b3 = (dh[i + 1][j] - dh[i][j]);
			}
			else {
				b2 = (dc[i - 1][j] + 2.0 * dc[i][j] + dc[i + 1][j]) / 4.0;
				b3 = (dh[i + 1][j] - dh[i - 1][j]) / 2.0;
			}

			double len;
			double tx = a2 * b3 - a3 * b2;
			double ty = a3 * b1 - a1 * b3;
			double tz = a1 * b2 - a2 * b1;
			if (tz < 0.0){
				tx = -tx;
				ty = -ty;
				tz = -tz;
			}
			len = tx * tx + ty * ty + tz * tz;
			len = sqrt(len);
			tx /= len; ty /= len; tz /= len;
			dx[i][j] = tx, dy[i][j] = ty, dz[i][j] = tz;
			ans.at<Vec3d>(i, j)[0] = abs(tx * 255);
			ans.at<Vec3d>(i, j)[1] = abs(ty * 255);
			ans.at<Vec3d>(i, j)[2] = abs(tz * 255);
		}
	}
	Mat hehe(nRows, nCols, CV_8UC3);
	ans.convertTo(hehe, CV_8UC3);

	imwrite("snapshot-norm.jpg", ans);
}

int main(void) {
	
	//calnorm();
	doMath();

	return 0;

	//std::vector<Eigen::Triplet<double>> hehe;
	//Eigen::SparseMatrix<double> A = Eigen::SparseMatrix<double>(3, 3);
	//Eigen::VectorXd b = Eigen::VectorXd(3);
	//Eigen::VectorXd u = Eigen::VectorXd(3);
	////A.reserve(100);
	//hehe.push_back(Eigen::Triplet<double>(0, 0, 1));
	//hehe.push_back(Eigen::Triplet<double>(0, 1, 2));
	//hehe.push_back(Eigen::Triplet<double>(0, 2, 3));
	//hehe.push_back(Eigen::Triplet<double>(1, 0, 2));
	//hehe.push_back(Eigen::Triplet<double>(1, 1, 3));
	//hehe.push_back(Eigen::Triplet<double>(1, 2, 4));
	//hehe.push_back(Eigen::Triplet<double>(2, 0, 3));
	//hehe.push_back(Eigen::Triplet<double>(2, 1, 4));
	//hehe.push_back(Eigen::Triplet<double>(2, 2, 7));

	//hehe.push_back(Eigen::Triplet<double>(1, 1, 3));

	//A.setFromTriplets(hehe.begin(), hehe.end());
	//
	//b(0) = 6;
	//b(1) = 12;
	//b(2) = 14;

	//Eigen::UmfPackLU<const Eigen::SparseMatrix<double, Eigen::ColMajor>> lu_of_A;
	//lu_of_A.compute(A);

	//u = lu_of_A.solve(b);

	//cout << u << endl;

	//return 0;

}