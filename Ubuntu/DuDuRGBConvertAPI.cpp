#include "convertTool.h"
#include "DuDuRGBConvertAPI.h"

class RGBConverterToolWrapper : public IDuDuRGBConverter {
    ConverterTool* m_convertTool;
public:
    RGBConverterToolWrapper() : m_convertTool(NULL)
    {
        m_convertTool = new ConverterTool();
    }

    ~RGBConverterToolWrapper() {
        if (m_convertTool)
            delete m_convertTool;
        m_convertTool = NULL;
    }

    virtual void Initialize()
    {
        if (!m_convertTool)
            return;

        m_convertTool->initialCuda();
    }

    virtual bool IsGPUSupport()
    {
        if (!m_convertTool)
            return false;

        return m_convertTool->isGPUEnable();
    }

    virtual void SetSrcSize(int w, int h)
    {
        if (!m_convertTool)
            return;

        m_convertTool->setSrcSize(w, h);
    }

    virtual void SetDstSize(int w, int h)
    {
        if (!m_convertTool)
            return;

        m_convertTool->setDstSize(w, h);
    }

    virtual void AllocateSrcAndTableMem()
    {
        if (!m_convertTool)
            return;

        m_convertTool->lookupTableF();
        m_convertTool->allocatSrcMem();
    }

    virtual void SetCudaDevSrc(unsigned short *src)
    {
        if (!m_convertTool)
            return;

        m_convertTool->setCudaDevSrc(src);
    }

    virtual void AllocateV210DstMem()
    {
        if (!m_convertTool)
            return;

        m_convertTool->allocatV210DstMem();
    }

    virtual void AllocatNVJPEGRGBMem()
    {
        if (!m_convertTool)
            return;

        m_convertTool->allocatNVJPEGRGBMem();
    }

    virtual void RGB10ConvertAndResizeToNVJPEG(unsigned char *pDst, int *nJPEGSize)
    {
        if (!m_convertTool)
            return;

        m_convertTool->RGB10ConvertToRGB8NVJPEG(pDst, nJPEGSize);
    }

    virtual void RGB10ConvertToV210(unsigned short *pDst)
    {
        if (!m_convertTool)
            return;

        m_convertTool->RGB10ConvertToV210(pDst);
    }

    virtual void FreeMemory()
    {
        if (!m_convertTool)
            return;

        m_convertTool->freeMemory();
    }

    virtual void Destroy()
    {
        if (!m_convertTool)
            return;

        m_convertTool->destroyCudaEvent();
    }
};


extern "C" IDuDuRGBConverter* DuDuRGBConverterAPICreate()
{
    return new RGBConverterToolWrapper;
}
