#ifndef DUDU_CONVERT_API_H
#define DUDU_CONVERT_API_H

class IDuDuConverter {
public:
    virtual void Initialize() = 0;
    virtual bool IsGPUSupport() = 0;
    virtual void SetSrcSize(int w, int h) = 0;
    virtual void SetDstSize(int w, int h) = 0;
    virtual void AllocateMem() = 0;
    virtual void ConvertAndResize(unsigned short *src, unsigned char *p208Dst, int *nJPEGSize) = 0;
    virtual void FreeMemory() = 0;
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