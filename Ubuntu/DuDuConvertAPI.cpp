#include "convertTool.h"
#include "DuDuConvertAPI.h"

class ConverterToolWrapper : public IDuDuConverter {
	ConverterTool* m_convertTool;
public:
    ConverterToolWrapper() : m_convertTool(NULL) 
    {
        m_convertTool = new ConverterTool();
    }
    ~ConverterToolWrapper() {
        if (m_convertTool)
            delete m_convertTool;
        m_convertTool = NULL;
    }

    virtual void Initialize()
    {
        if (!m_convertTool)
            return;

        m_convertTool->initialCuda();
        m_convertTool->preprocess();
        m_convertTool->lookupTableF();
    }

    virtual bool IsGPUSupport() 
    {
        if (!m_convertTool)
            return false;

        return m_convertTool->isGPUEnable();
    }

    virtual void ConvertAndResize(unsigned short *src, int nSrcW, int nSrcH,
        unsigned char *p208Dst, int nDstW, int nDstH, int *nJPEGSize) 
    {
        if (!m_convertTool)
            return;
        
        m_convertTool->convertToP208ThenResize(src, nSrcW, nSrcH, 
            p208Dst, nDstW, nDstH, nJPEGSize);
    }

    virtual void Destroy()
    {
        if (!m_convertTool)
            return;

        m_convertTool->freeMemory();
        m_convertTool->destroyCudaEvent();
    }
};


extern "C" IDuDuConverter* DuDuConverterAPICreate()
{
	return new ConverterToolWrapper;
}
