#ifndef DUDU_RGBCONVERT_API_H
#define DUDU_RGBCONVERT_API_H

class IDuDuRGBConverter {
public:
    virtual void Initialize() = 0;
    virtual void SetSrcSize(int w, int h) = 0;
    virtual void SetDstSize(int w, int h) = 0;

    virtual void AllocateSrcAndTableMem() = 0;
    virtual void AllocateV210DstMem() = 0;
    virtual void AllocatNVJPEGRGBMem() = 0;

    virtual void SetCudaDevSrc(unsigned short *src) = 0;
    virtual void RGB10ConvertAndResizeToNVJPEG(unsigned char *pDst, int *nJPEGSize) = 0;
    virtual void RGB10ConvertToV210(unsigned short *pDst) = 0;

    virtual void FreeMemory() = 0;
};

#ifdef __cplusplus
extern "C"
{
#endif
    IDuDuRGBConverter* DuDuRGBConverterAPICreate();
#ifdef __cplusplus
}
#endif

#endif // DUDU_RGBCONVERT_API_H