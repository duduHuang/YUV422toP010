#ifndef DUDU_CONVERT_API_H
#define DUDU_CONVERT_API_H

class IDuDuConverter {
public:
    virtual void Initialize() = 0;
    virtual bool IsGPUSupport() = 0;
    virtual void ConvertAndResize(unsigned short *src, int nSrcW, int nSrcH,
        unsigned char *p208Dst, int nDstW, int nDstH, int *nJPEGSize) = 0;
    virtual void Destroy() = 0;
};

#ifdef __cplusplus
extern "C"
{
#endif
    IDuDuConverter* DuDuConverterAPICreate();
#ifdef __cplusplus
}
#endif

#endif // DUDU_CONVERT_API_H