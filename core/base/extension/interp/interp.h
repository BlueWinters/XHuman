
#ifndef __INTERP__
#define __INTERP__

#include <cmath>


void  _bicubic_byte(
	const unsigned char * src,
	int width, int height, int nColor,
	int neww, int newh,
	unsigned char* dst
);

void  _bicubic_float(
	const float *src,
	int width, int height, int nColor,
	int neww, int newh,
	float* dst
);

#endif // !INTERP_H
