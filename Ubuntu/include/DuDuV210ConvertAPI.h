#ifndef DUDU_V210CONVERT_API_H
#define DUDU_V210CONVERT_API_H

class IDuDuV210Converter {
public:
    virtual void Initialize() = 0;
    virtual void SetSrcSize(int w, int h) = 0;
    virtual void SetDstSize(int w, int h) = 0;
    virtual void AllocateMem() = 0;
    virtual void ConvertAndResize(unsigned short *src, unsigned char *p208Dst, int *nJPEGSize) = 0;
    virtual void FreeMemory() = 0;
};

#ifdef __cplusplus
extern "C"
{
#endif
    IDuDuV210Converter* DuDuV210ConverterAPICreate();
#ifdef __cplusplus
}
#endif

#endif // DUDU_V210CONVERT_API_H