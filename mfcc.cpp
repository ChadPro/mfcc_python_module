#include <stdio.h>
#include <stdlib.h>
#include "/home/ly/anaconda3/envs/learn/include/python3.5m/Python.h"

int filterNum = 20;
int sampleRate = 16000;

#define Win_Time 0.025
#define Hop_Time 0.01
#define Pi 3.1415927

struct WavHead{
    char RIFF[4];
    int size0;
    char WAVE[4];
    char FMT[4];
    int size1;
    short fmttag;
    short channel;
    int samplespersec;
    int bytepersec;
    short blockalign;
    short bitpersamples;
    char SUBID[4];
    int size2;
};

double* pre_emphasizing(double *sample, int len, double factor){
	double *Sample = new double[len];
	Sample[0] = sample[0];
	for(int i = 1; i < len; i++)
	{
		//预加重过程
		Sample[i] = sample[i] - factor * sample[i - 1]; 
	}
	return Sample;
}

void Hamming( double *hamWin, int hamWinSize ){
	for (int i = 0; i < hamWinSize; i++)
	{
		hamWin[i] = (double)(0.54 - 0.46 * cos(2 * Pi * (double)i / ((double)hamWinSize - 1) ));
	}
}

void mfccFFT(double *frameSample, double *FFTSample, int frameSize, int pos){
    //对分帧加窗后的各帧信号进行FFT变换得到各帧的频谱
	//并对语音信号的频谱取模平方得到语音信号的功率谱
	double dataR[frameSize];
	double dataI[frameSize];
	for(int i = 0; i < frameSize; i++)
	{
		dataR[i] = frameSample[i + pos];
		dataI[i] = 0.0f;
	}
	
	int x0, x1, x2, x3, x4, x5, x6, xx, x7, x8;
	int i, j, k, b, p, L;
	float TR, TI, temp;
	/********** following code invert sequence ************/
	for(i = 0; i < frameSize; i++)
	{
		x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = 0;
		x0 = i & 0x01; x1 = (i / 2) & 0x01; x2 = (i / 4) & 0x01; x3 = (i / 8) & 0x01; x4 = (i / 16) & 0x01; 
		x5 = (i / 32) & 0x01; x6 = (i / 64) & 0x01; x7 = (i / 128) & 0x01; x8 = (i / 256) & 0x01;
		xx = x0 * 256 + x1 * 128 + x2 * 64 + x3 * 32 + x4 * 16 + x5 * 8 + x6 * 4 + x7 * 2 + x8;
		dataI[xx] = dataR[i];
	}
	for(i = 0; i < frameSize; i++)
	{
		dataR[i] = dataI[i]; dataI[i] = 0; 
	}
 
	/************** following code FFT *******************/
	for(L = 1; L <= 9; L++)
	{ /* for(1) */
		b = 1; i = L - 1;
		while(i > 0) 
		{
			b = b * 2; i--;
		} /* b= 2^(L-1) */
		for(j = 0; j <= b-1; j++) /* for (2) */
		{
			p = 1; i = 9 - L;
			while(i > 0) /* p=pow(2,7-L)*j; */
			{
				p = p * 2; i--;
			}
			p = p * j;
			for(k = j; k < 512; k = k + 2*b) /* for (3) */
			{
				TR = dataR[k]; TI = dataI[k]; temp = dataR[k + b];
				dataR[k] = dataR[k] + dataR[k + b] * cos(2 * Pi * p / frameSize) + dataI[k + b] * sin(2 * Pi * p / frameSize);
				dataI[k] = dataI[k] - dataR[k + b] * sin(2 * Pi * p / frameSize) + dataI[k + b] * cos(2 * Pi * p / frameSize);
				dataR[k + b] = TR - dataR[k + b] * cos(2 * Pi * p / frameSize) - dataI[k + b] * sin(2 * Pi * p / frameSize);
				dataI[k + b] = TI + temp * sin(2 * Pi * p / frameSize) - dataI[k + b] * cos(2 * Pi * p / frameSize);
			} /* END for (3) */
		} /* END for (2) */
	} /* END for (1) */
	for(i = 0; i < frameSize / 2; i++)
	{ 
	    FFTSample[i + pos] = (dataR[i] * dataR[i] + dataI[i] * dataI[i]);
        // FFTSample[i + pos] = dataR[i];
	}	
}
 
double* mfccFrame(const int pcm_len, const int frameNum, const int hopStep, const int frameSampleLen, const int frameSize, double *frameSample, double *Sample){
	double *hamWin;
	int hamWinSize = sampleRate * Win_Time;
	hamWin = new double[hamWinSize];
	Hamming(hamWin, hamWinSize);
	frameSample = new double[frameSampleLen];
	for(int i = 0; i < frameSampleLen; i++)
	    frameSample[i] = 0;
	
	double *FFTSample = new double[frameSampleLen];
	for(int i = 0; i < frameSampleLen; i++)
	    FFTSample[i] = 0;
	

	for(int i = 0; i * hopStep < pcm_len; i++)
	{
		for(int j = 0; j < frameSize; j++)
		{
			if(j < hamWinSize && i * hopStep + j < pcm_len)
			    frameSample[i * frameSize + j] = Sample[i * hopStep + j] * hamWin[j];
		    else
		        frameSample[i * frameSize + j] = 0;
		}
		mfccFFT(frameSample, FFTSample, frameSize, i * frameSize);
	}

	delete []frameSample;
	delete []hamWin;
	return FFTSample; 
}

void DCT(double* mel, double* c, int frameNum){  
    for(int k = 0; k < frameNum; k++)
    {
        for(int i = 0; i < 13; i++)  
        {  
        	for(int j = 0; j < filterNum; j++)  
        	{    
				c[k * filterNum + i] += mel[k * filterNum + j] * cos(Pi * i / (2 * filterNum) *  (2 * j + 1));
        	}	  
    	}
    }
} 

void computeMel(double* mel, int sampleRate, double *FFTSample, int frameNum, int frameSize){
	double freMax = sampleRate / 2;//实际最大频率 
	double freMin = 0;//实际最小频率 
	double melFremax = 1125 * log(1 + freMax / 700);//将实际频率转换成梅尔频率 
	double melFremin = 1125 * log(1 + freMin / 700);
	double k = (melFremax - melFremin) / (filterNum + 1);
	
	double *m = new double[filterNum + 2];
	double *h = new double[filterNum + 2];
	double *f = new double[filterNum + 2];
	
	for(int i = 0; i < filterNum + 2; i++)
	{
		m[i] = melFremin + k * i;
		h[i] = 700 * (exp(m[i] / 1125) - 1);//将梅尔频率转换成实际频率 
		f[i] = floor((frameSize + 1) * h[i] / sampleRate);
	}		
 
    delete[] m;  
    delete[] h;
	
	//计算出每个三角滤波器的输出: 对每一帧进行处理 	
	for(int i = 0; i < frameNum; i++)
	{
		for(int j = 1; j <= filterNum; j++)
		{
			double temp = 0;
			for(int z = 0; z < frameSize; z++)
			{
				if(z < f[j - 1])
				    temp = 0;
				else if(z >= f[j - 1] && z <= f[j])
				    temp = (z - f[j - 1]) / (f[j] - f[j - 1]);
				else if(z >= f[j] && z <= f[j + 1])
				    temp = (f[j + 1] - z) / (f[j + 1] - f[j]);
				else if(z > f[j + 1])
				    temp = 0;
				// mel[i][j - 1] += FFTSample[i * frameSize + z] * temp;
				mel[i * filterNum + j - 1] += FFTSample[i * frameSize + z] * temp;
			}
		}
    }
	
	//取对数 
	for(int i = 0; i < frameNum * filterNum; i++)
	{
		if(mel[i] <= 0.00000000001 || mel[i] >= 0.00000000001) {
			mel[i] = log(mel[i]);
		}
	}

	delete[] f;
}

bool str_is_data(const char* str) {
	if (str[0] == 'd' & str[1] == 'a' & str[2] == 't' & str[3] == 'a') return true;
	return false;
}


int get_wav_length(FILE* fp, int offset) {
	char* skip_buf = (char*)malloc(offset);
	size_t num = 0;
	num = fread(skip_buf, sizeof(char), offset, fp);

	char* idstr = (char*)malloc(4);
	memset(idstr, 0x0, 4);
	num = fread(idstr, sizeof(char), 4, fp);
	
	while(!str_is_data(idstr)) {
		int* skipsize = (int*)malloc(4);
		num = fread(skipsize, sizeof(char), 4, fp);

		num = fread(skip_buf, sizeof(char), skipsize[0], fp);
		num = fread(idstr, sizeof(char), 4, fp);

		free(skipsize);
		skipsize = NULL;
	}

	int* realsize = (int*)malloc(4);
	memset(realsize, 0x0, 4);
	num = fread(realsize, sizeof(char), 4, fp);
	int back = realsize[0];

	free(realsize);
	realsize = NULL;
	free(idstr);
	idstr = NULL;
	free(skip_buf);
	skip_buf = NULL;
	return back;
}

PyObject* mfcc(const char* filename, const int mfcc_nums) {
    // read wav
	size_t num = 0;
	int raw_data_len = 0;
	int sampel_bit = 0;
	WavHead* head = (WavHead*)malloc(sizeof(WavHead));
	memset(head, 0x0, sizeof(WavHead));

	FILE *fp = NULL;
	fp = fopen(filename, "r");
	num = fread(head, sizeof(char), sizeof(WavHead), fp);
	sampel_bit = int(head->bitpersamples / 8);

	if (str_is_data(head->SUBID)) {
		raw_data_len = head->size2 / sampel_bit;
	} else {
		raw_data_len = get_wav_length(fp, head->size2) / sampel_bit;
	}

	free(head);
	head = NULL;

    char* pcm_buf = (char*)malloc(raw_data_len * sampel_bit);
    int16_t* pcm_data = (int16_t*)malloc(raw_data_len * sampel_bit);
    memset(pcm_buf, 0x0, raw_data_len * sampel_bit);
    memset(pcm_data, 0x0, raw_data_len * sampel_bit);
    
    num = fread(pcm_buf, sizeof(char), raw_data_len * sampel_bit, fp);
	fclose(fp);
	fp = NULL;
    memcpy(pcm_data, pcm_buf, raw_data_len * sampel_bit);

    double* pcm_double = (double*)malloc(raw_data_len * 8);
    for (int i = 0; i < raw_data_len; ++i) {
        pcm_double[i] = (double)pcm_data[i]; 
    }

    free(pcm_buf);
    pcm_buf = NULL;
    free(pcm_data);
    pcm_data = NULL;

    /****** 预加重 ******/
    double factor = 0.95;
    double *pre_sample = pre_emphasizing(pcm_double, raw_data_len, factor);
	free(pcm_double);
	pcm_double = NULL;

    /****** 分帧、加窗、fft *******/
	int frameSize = (int)pow(2, ceil( log(Win_Time * sampleRate) / log(2.0))); // 分桢长度512, 400到512部分补0
    double* frameSample = NULL;
    double* FFTSample = NULL; 
	int hopStep = Hop_Time * sampleRate;
	int frameNum = ceil(double(raw_data_len) / hopStep);
	int frameSampleLen = frameNum * frameSize;


	FFTSample = mfccFrame(raw_data_len, frameNum, hopStep, frameSampleLen, frameSize, frameSample, pre_sample);
	delete []pre_sample;
	pre_sample = NULL;

	double* mel = (double*)malloc(frameNum * mfcc_nums * sizeof(double));
	memset(mel, 0x0, frameNum * mfcc_nums * sizeof(double));
	computeMel(mel, sampleRate, FFTSample, frameNum, frameSize);

	delete []FFTSample;
	FFTSample = NULL;

	double* c = (double*)malloc(frameNum * mfcc_nums * sizeof(double));
	memset(c, 0x0, frameNum * mfcc_nums * sizeof(double));
	DCT(mel, c, frameNum);

    PyObject* pyData = PyList_New(frameNum * mfcc_nums);
    for (int i = 0; i < frameNum * mfcc_nums; ++i) {
        PyObject* op = PyFloat_FromDouble(c[i]);
        PyList_SetItem(pyData, i, op);
    }
	free(c);
	c = NULL;
	free(mel);
	mel = NULL;

    return pyData;
}

static PyObject* QSAudio_mfcc(PyObject* self, PyObject* args) {
    char* filename;
	int mfcc_nums;
    if (!PyArg_ParseTuple(args, "si", &filename, &mfcc_nums)) return NULL;
    return (PyObject*)Py_BuildValue("N", mfcc(filename, mfcc_nums));
}

static PyMethodDef QSAudioMethods[] = {
    {"mfcc", QSAudio_mfcc, METH_VARARGS}
};

static struct PyModuleDef QSAudiomodule = {
    PyModuleDef_HEAD_INIT,
    "QSAudio",
    NULL,
    -1,
    QSAudioMethods
};

PyMODINIT_FUNC PyInit_QSAudio(void) {
    return PyModule_Create(&QSAudiomodule);
}
