
#include <cstring>
#include "interp.h"

#define	MAX(a, b) (((a) > (b)) ? (a) : (b))	 
#define	MIN(a, b) (((a) < (b)) ? (a) : (b))	 

#define TRIANGLE_KERNEL 4
void _bicubic_byte(
	const unsigned char * src,
	int width, int height, int nColor,
	int neww, int newh,
	unsigned char *dst
)
{
	//this subroutines refer to Matlab imresize: contributions 
	//shrinking image and anti-alias effect
	//kernel ---triangle kernle, kernel_width = 2;  //translated from MATLAB
	double  kernel_width;
	double  scale[2]; //scaleX, scaleY;
	int     inlength[2], outlength[2]; //newW, newH;
	int     P;
	double  *weights;
	int     *indices;

	double    *pmid;
	double    *src2 = new double[nColor * width * height];
	unsigned char    *tmp = new unsigned char[nColor * neww * newh];

	scale[0] = double(newh) / double(height);
	scale[1] = double(neww) / double(width);

	bool reverse;
	if (scale[0] > scale[1])
		reverse = true;
	else
		reverse = false;

	if (reverse)
	{
		scale[0] = double(neww) / double(width);
		scale[1] = double(newh) / double(height);
		inlength[0] = width;
		inlength[1] = height;
		outlength[0] = neww;
		outlength[1] = newh;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				for (int c = 0; c < nColor; c++)
					src2[nColor*(j*height + i) + c] = double(src[nColor*(i*width + j) + c]);
		height = inlength[0];
		width = inlength[1];
		newh = outlength[0];
		neww = outlength[1];
	}
	else
	{
		inlength[0] = height;
		inlength[1] = width;
		outlength[0] = newh;
		outlength[1] = neww;
		//memcpy(src2, src, sizeof(BYTE) * nColor * height * width);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				for (int c = 0; c < nColor; c++)
					src2[nColor*(i*width + j) + c] = double(src[nColor*(i*width + j) + c]);
	}

	if (nColor == 3) {
		pmid = new double[3 * width*newh];
		memset(pmid, 0, sizeof(double) * 3 * newh*width);
		memset(dst, 0, sizeof(unsigned char) * 3 * newh*neww);
	}
	else if (nColor == 1) {
		pmid = new double[width*newh];
		memset(pmid, 0, sizeof(double)*newh*width);
		memset(dst, 0, sizeof(unsigned char)*newh*neww);
	}
	else {
		pmid = NULL; //error
		delete[] src2;
		delete[] tmp;
		return;
	}

	int left;
	double center, sum, dist;
	//first Y dimension
	{
		if (scale[0] < 1)
			kernel_width = TRIANGLE_KERNEL / scale[0];
		else
			kernel_width = TRIANGLE_KERNEL;

		P = ceil(kernel_width) + 2;

		weights = new double[P];
		indices = new int[P];
		for (int y = 0; y < newh; y++) {
			//center, and left 
			center = (y + 0.5) / scale[0] - 0.5;  //+ 0.5
			left = floor(center - kernel_width / 2);

			//triangle kernel weights, and normalize
			sum = 0.0;
			for (int i = 0; i < P; i++) {
				indices[i] = left + i;

				if (scale[0] < 1)
					dist = scale[0] * fabs(center - indices[i]);
				else
					dist = fabs(center - indices[i]);

				if (dist <= 1)
					weights[i] = 1.5*dist*dist*dist - 2.5*dist*dist + 1;
				else if (dist <= 2)
					weights[i] = -0.5*dist*dist*dist + 2.5*dist*dist - 4 * dist + 2;
				else
					weights[i] = 0.0;

				if (scale[0] < 1)
					weights[i] *= scale[0];

				sum += weights[i];

				//����Խ���жϣ����þ���padding
				if (indices[i] < 0)
					indices[i] = -indices[i] - 1;
				else if (indices[i] > height - 1)
					indices[i] = 2 * height - 1 - indices[i];

			}
			//normalize weights;
			for (int i = 0; i < P; i++)
				weights[i] = weights[i] / sum;

			//��ֵ����
			//����matlab�汾��pmid�������������ͼ����ͬ�ľ���
			for (int x = 0; x < width; x++) {
				if (nColor == 3) {
					double sr, sg, sb;
					sb = 0.0; sg = 0.0; sr = 0.0;
					for (int i = 0; i < P; i++) {
						sb += weights[i] * src2[3 * indices[i] * width + 3 * x + 0];
						sg += weights[i] * src2[3 * indices[i] * width + 3 * x + 1];
						sr += weights[i] * src2[3 * indices[i] * width + 3 * x + 2];
					}
					sb = MIN(MAX(sb, 0), 255);
					pmid[3 * y*width + 3 * x + 0] = int(sb + 0.5);
					sg = MIN(MAX(sg, 0), 255);
					pmid[3 * y*width + 3 * x + 1] = int(sg + 0.5);
					sr = MIN(MAX(sr, 0), 255);
					pmid[3 * y*width + 3 * x + 2] = int(sr + 0.5);
				}
				else { //nColor == 1				
					double sp;
					sp = 0.0;
					for (int i = 0; i < P; i++) {
						sp += weights[i] * src2[indices[i] * width + x];
					}
					sp = MIN(MAX(sp, 0), 255);
					pmid[y*width + x] = int(sp + 0.5);
				}
			}
		}
		delete[] weights;
		delete[] indices;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//X dimension
	{
		if (scale[1] < 1)
			kernel_width = TRIANGLE_KERNEL / scale[1];
		else
			kernel_width = TRIANGLE_KERNEL;
		P = ceil(kernel_width) + 2;
		weights = new double[P];
		indices = new int[P];

		for (int x = 0; x < neww; x++) {

			center = (x + 0.5) / scale[1] - 0.5;  //+0.5
			left = floor(center - kernel_width / 2);
			//get weights & ptindex for this dimension
			//mapping from output space to input space
			sum = 0.0;
			for (int i = 0; i < P; i++) {
				indices[i] = left + i;

				if (scale[1] < 1)
					dist = scale[1] * fabs(center - indices[i]);
				else
					dist = fabs(center - indices[i]);

				if (dist <= 1)
					weights[i] = 1.5*dist*dist*dist - 2.5*dist*dist + 1;
				else if (dist <= 2)
					weights[i] = -0.5*dist*dist*dist + 2.5*dist*dist - 4 * dist + 2;
				else
					weights[i] = 0.0;

				if (scale[1] < 1)
					weights[i] *= scale[1];

				sum += weights[i];

				//����Խ���жϣ����þ���padding
				if (indices[i] < 0)
					indices[i] = -indices[i] - 1;
				else if (indices[i] > width - 1)
					indices[i] = 2 * width - 1 - indices[i];

			}
			//normalize weights;
			for (int i = 0; i < P; i++)
				weights[i] = weights[i] / sum;

			for (int y = 0; y < newh; y++) {
				if (nColor == 3) {
					double sr, sg, sb;
					sb = 0.0;	sg = 0.0;	sr = 0.0;
					for (int i = 0; i < P; i++) {
						sb += weights[i] * pmid[3 * y*width + 3 * indices[i] + 0];
						sg += weights[i] * pmid[3 * y*width + 3 * indices[i] + 1];
						sr += weights[i] * pmid[3 * y*width + 3 * indices[i] + 2];
					}
					sb = MIN(255, MAX(sb, 0));
					sg = MIN(255, MAX(sg, 0));
					sr = MIN(255, MAX(sr, 0));
					//sb = std::round(sb);
					//sg = std::round(sg);
					//sr = std::round(sr);
					tmp[3 * y*neww + 3 * x + 0] = (unsigned char)(sb + 0.5);
					tmp[3 * y*neww + 3 * x + 1] = (unsigned char)(sg + 0.5);
					tmp[3 * y*neww + 3 * x + 2] = (unsigned char)(sr + 0.5);
				}
				else if (nColor == 1) {
					double sp;
					sp = 0.0;
					for (int i = 0; i < P; i++) {
						sp += weights[i] * pmid[y*width + indices[i]];
					}
					//sp = MIN(255, MAX(sp, 0));
					//sp = std::round(sp);
					tmp[y*neww + x] = (unsigned char)(sp + 0.5);
				}
			}
		}
		delete[] weights;
		delete[] indices;
	}

	if (reverse)
		for (int i = 0; i < newh; i++)
			for (int j = 0; j < neww; j++)
				for (int c = 0; c < nColor; c++)
					dst[nColor*(j*newh + i) + c] = tmp[nColor*(i*neww + j) + c];
	else
		memcpy(dst, tmp, sizeof(unsigned char) * nColor * newh * neww);

	delete[] tmp;
	delete[] src2;
	delete[] pmid;
}

void _bicubic_float(
	const float * src,
	int width, int height, int nColor,
	int neww, int newh,
	float *dst
)
{
	//this subroutines refer to Matlab imresize: contributions 
	//shrinking image and anti-alias effect
	//kernel ---triangle kernle, kernel_width = 2;  //translated from MATLAB
	double  kernel_width;
	double  scale[2]; //scaleX, scaleY;
	int     inlength[2], outlength[2]; //newW, newH;
	int     P;
	double  *weights;
	int     *indices;

	double    *pmid;
	double    *src2 = new double[nColor * width * height];
	unsigned char    *tmp = new unsigned char[nColor * neww * newh];

	scale[0] = double(newh) / double(height);
	scale[1] = double(neww) / double(width);

	bool reverse;
	if (scale[0] > scale[1])
		reverse = true;
	else
		reverse = false;

	if (reverse)
	{
		scale[0] = double(neww) / double(width);
		scale[1] = double(newh) / double(height);
		inlength[0] = width;
		inlength[1] = height;
		outlength[0] = neww;
		outlength[1] = newh;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				for (int c = 0; c < nColor; c++)
					src2[nColor*(j*height + i) + c] = double(MIN(MAX((src[nColor*(i*width + j) + c] + 1) / 2 * 255, 0), 255));
		height = inlength[0];
		width = inlength[1];
		newh = outlength[0];
		neww = outlength[1];
	}
	else
	{
		inlength[0] = height;
		inlength[1] = width;
		outlength[0] = newh;
		outlength[1] = neww;
		//memcpy(src2, src, sizeof(BYTE) * nColor * height * width);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				for (int c = 0; c < nColor; c++)
					src2[nColor*(i*width + j) + c] = double(MIN(MAX((src[nColor*(i*width + j) + c] + 1) / 2 * 255, 0), 255));
	}

	if (nColor == 3) {
		pmid = new double[3 * width*newh];
		memset(pmid, 0, sizeof(double) * 3 * newh*width);
		memset(dst, 0, sizeof(unsigned char) * 3 * newh*neww);
	}
	else if (nColor == 1) {
		pmid = new double[width*newh];
		memset(pmid, 0, sizeof(double)*newh*width);
		memset(dst, 0, sizeof(unsigned char)*newh*neww);
	}
	else {
		pmid = NULL; //error
		delete[] src2;
		delete[] tmp;
		return;
	}

	int left;
	double center, sum, dist;
	//first Y dimension
	{
		if (scale[0] < 1)
			kernel_width = TRIANGLE_KERNEL / scale[0];
		else
			kernel_width = TRIANGLE_KERNEL;

		P = ceil(kernel_width) + 2;

		weights = new double[P];
		indices = new int[P];
		for (int y = 0; y < newh; y++) {
			//center, and left 
			center = (y + 0.5) / scale[0] - 0.5;  //+ 0.5
			left = floor(center - kernel_width / 2);

			//triangle kernel weights, and normalize
			sum = 0.0;
			for (int i = 0; i < P; i++) {
				indices[i] = left + i;

				if (scale[0] < 1)
					dist = scale[0] * fabs(center - indices[i]);
				else
					dist = fabs(center - indices[i]);

				if (dist <= 1)
					weights[i] = 1.5*dist*dist*dist - 2.5*dist*dist + 1;
				else if (dist <= 2)
					weights[i] = -0.5*dist*dist*dist + 2.5*dist*dist - 4 * dist + 2;
				else
					weights[i] = 0.0;

				if (scale[0] < 1)
					weights[i] *= scale[0];

				sum += weights[i];

				//����Խ���жϣ����þ���padding
				if (indices[i] < 0)
					indices[i] = -indices[i] - 1;
				else if (indices[i] > height - 1)
					indices[i] = 2 * height - 1 - indices[i];

			}
			//normalize weights;
			for (int i = 0; i < P; i++)
				weights[i] = weights[i] / sum;

			//��ֵ����
			//����matlab�汾��pmid�������������ͼ����ͬ�ľ���
			for (int x = 0; x < width; x++) {
				if (nColor == 3) {
					double sr, sg, sb;
					sb = 0.0; sg = 0.0; sr = 0.0;
					for (int i = 0; i < P; i++) {
						sb += weights[i] * src2[3 * indices[i] * width + 3 * x + 0];
						sg += weights[i] * src2[3 * indices[i] * width + 3 * x + 1];
						sr += weights[i] * src2[3 * indices[i] * width + 3 * x + 2];
					}
					sb = MIN(MAX(sb, 0), 255);
					pmid[3 * y*width + 3 * x + 0] = int(sb + 0.5);
					sg = MIN(MAX(sg, 0), 255);
					pmid[3 * y*width + 3 * x + 1] = int(sg + 0.5);
					sr = MIN(MAX(sr, 0), 255);
					pmid[3 * y*width + 3 * x + 2] = int(sr + 0.5);
				}
				else { //nColor == 1				
					double sp;
					sp = 0.0;
					for (int i = 0; i < P; i++) {
						sp += weights[i] * src2[indices[i] * width + x];
					}
					sp = MIN(MAX(sp, 0), 255);
					pmid[y*width + x] = int(sp + 0.5);
				}
			}
		}
		delete[] weights;
		delete[] indices;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	//X dimension
	{
		if (scale[1] < 1)
			kernel_width = TRIANGLE_KERNEL / scale[1];
		else
			kernel_width = TRIANGLE_KERNEL;
		P = ceil(kernel_width) + 2;
		weights = new double[P];
		indices = new int[P];

		for (int x = 0; x < neww; x++) {

			center = (x + 0.5) / scale[1] - 0.5;  //+0.5
			left = floor(center - kernel_width / 2);
			//get weights & ptindex for this dimension
			//mapping from output space to input space
			sum = 0.0;
			for (int i = 0; i < P; i++) {
				indices[i] = left + i;

				if (scale[1] < 1)
					dist = scale[1] * fabs(center - indices[i]);
				else
					dist = fabs(center - indices[i]);

				if (dist <= 1)
					weights[i] = 1.5*dist*dist*dist - 2.5*dist*dist + 1;
				else if (dist <= 2)
					weights[i] = -0.5*dist*dist*dist + 2.5*dist*dist - 4 * dist + 2;
				else
					weights[i] = 0.0;

				if (scale[1] < 1)
					weights[i] *= scale[1];

				sum += weights[i];

				//����Խ���жϣ����þ���padding
				if (indices[i] < 0)
					indices[i] = -indices[i] - 1;
				else if (indices[i] > width - 1)
					indices[i] = 2 * width - 1 - indices[i];

			}
			//normalize weights;
			for (int i = 0; i < P; i++)
				weights[i] = weights[i] / sum;

			for (int y = 0; y < newh; y++) {
				if (nColor == 3) {
					double sr, sg, sb;
					sb = 0.0;	sg = 0.0;	sr = 0.0;
					for (int i = 0; i < P; i++) {
						sb += weights[i] * pmid[3 * y*width + 3 * indices[i] + 0];
						sg += weights[i] * pmid[3 * y*width + 3 * indices[i] + 1];
						sr += weights[i] * pmid[3 * y*width + 3 * indices[i] + 2];
					}
					sb = MIN(255, MAX(sb, 0));
					sg = MIN(255, MAX(sg, 0));
					sr = MIN(255, MAX(sr, 0));
					//sb = std::round(sb);
					//sg = std::round(sg);
					//sr = std::round(sr);
					tmp[3 * y*neww + 3 * x + 0] = (unsigned char)(sb + 0.5);
					tmp[3 * y*neww + 3 * x + 1] = (unsigned char)(sg + 0.5);
					tmp[3 * y*neww + 3 * x + 2] = (unsigned char)(sr + 0.5);
				}
				else if (nColor == 1) {
					double sp;
					sp = 0.0;
					for (int i = 0; i < P; i++) {
						sp += weights[i] * pmid[y*width + indices[i]];
					}
					//sp = MIN(255, MAX(sp, 0));
					//sp = std::round(sp);
					tmp[y*neww + x] = (unsigned char)(sp + 0.5);
				}
			}
		}
		delete[] weights;
		delete[] indices;
	}

	if (reverse)
		for (int i = 0; i < newh; i++)
			for (int j = 0; j < neww; j++)
				for (int c = 0; c < nColor; c++)
					dst[nColor*(j*newh + i) + c] = MIN(MAX(float(tmp[nColor*(i*neww + j) + c]) / 255 * 2 - 1, -1), 1);
	else
		for (int i = 0; i < neww*newh*nColor; i++) {
			dst[i] = MIN(MAX(float(tmp[i]) / 255 * 2 - 1, -1), 1);
		}

	delete[] tmp;
	delete[] src2;
	delete[] pmid;
}