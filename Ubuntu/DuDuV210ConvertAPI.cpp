#include "convertTool.h"
#include "DuDuV210ConvertAPI.h"

class V210ConverterToolWrapper : public IDuDuV210Converter {
    ConverterTool* m_convertTool;
public:
    V210ConverterToolWrapper() : m_convertTool(NULL) {
        m_convertTool = new ConverterTool();
    }

    ~V210ConverterToolWrapper() {
        if (m_convertTool)
            delete m_convertTool;
        m_convertTool = NULL;
    }

    virtual void Initialize() {
        if (!m_convertTool)
            return;

        m_convertTool->initialCuda();
    }

    virtual void SetSrcSize(int w, int h) {
        if (!m_convertTool)
            return;

        m_convertTool->setSrcSize(w, h);
    }

    virtual void SetDstSize(int w, int h) {
        if (!m_convertTool)
            return;

        m_convertTool->setDstSize(w, h);
    }

    virtual void AllocateMem() {
        if (!m_convertTool)
            return;

        m_convertTool->lookupTableF();
        m_convertTool->allocateMem();
    }

    virtual void ConvertAndResize(unsigned short *src, unsigned char *p208Dst, int *nJPEGSize) {
        if (!m_convertTool)
            return;

        m_convertTool->convertToP208ThenResize(src, p208Dst, nJPEGSize);
    }

    virtual void FreeMemory() {
        if (!m_convertTool)
            return;

        m_convertTool->freeMemory();
    }
};

extern "C" IDuDuV210Converter* DuDuV210ConverterAPICreate() {
    return new V210ConverterToolWrapper;
}
