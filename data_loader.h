#pragma once

#include <cstdio>
#include <cstdint>

/* Used to change the byte order. E.g., if the first two uint32_t values are 0x01080000 or 0x03080000.
The Inline specifier improves program performance by reducing the overhead of function calls and increasing execution speed. */
inline uint32_t EndianSwap (uint32_t a)
{
    return (a<<24) | ((a<<8) & 0x00ff0000) |
           ((a>>8) & 0x0000ff00) | (a>>24);
}
 
class MNISTData
{
public:
    MNISTData ()
    {
        m_labelData = nullptr;
        m_imageData = nullptr;
        m_imageCount = 0;
        m_labels = nullptr;
        m_pixels = nullptr;
    }
 
    bool Load (bool training)
    {
        // Set the expected image count
        m_imageCount = training ? 60000 : 10000;
 
        // Read labels
        const char* labelsFileName = training ? "train-labels.idx1-ubyte" : "t10k-labels.idx1-ubyte";
        FILE* file = fopen(labelsFileName,"rb");
        if (!file)
        {
            printf("Could not open %s for reading.\n", labelsFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_labelData = new uint8_t[fileSize];
        fread(m_labelData, fileSize, 1, file);
        fclose(file);
 
        // Read images
        const char* imagesFileName = training ? "train-images.idx3-ubyte" : "t10k-images.idx3-ubyte";
        file = fopen(imagesFileName, "rb");
        if (!file)
        {
            printf("Could not open %s for reading.\n", imagesFileName);
            return false;
        }
        fseek(file, 0, SEEK_END);
        fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        m_imageData = new uint8_t[fileSize];
        fread(m_imageData, fileSize, 1, file);
        fclose(file);
 
        /* Perform endian swapping on the label file's first two uint32 values if required,
        while the rest of the file contains uint8 values. */ 
        uint32_t* data = (uint32_t*)m_labelData;
        if (data[0] == 0x01080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
        }
 
        // Verifying if the label file has the right header
        if (data[0] != 2049 || data[1] != m_imageCount)
        {
            printf("The label data contains unexpected header values.\n");
            return false;
        }
        m_labels = (uint8_t*)&(data[2]);
 
        /* If necessary, perform endian swapping on the image file's first 
        four uint32 values. The remaining data consists of uint8 values. */
        data = (uint32_t*)m_imageData;
        if (data[0] == 0x03080000)
        {
            data[0] = EndianSwap(data[0]);
            data[1] = EndianSwap(data[1]);
            data[2] = EndianSwap(data[2]);
            data[3] = EndianSwap(data[3]);
        }
 
        // Verifying if the image file has the right header
        if (data[0] != 2051 || data[1] != m_imageCount || data[2] != 28 || data[3] != 28)
        {
            printf("The label data contains unexpected header values.\n");
            return false;
        }
        m_pixels = (uint8_t*)&(data[4]);
 
        // Convert the pixels from uint8 to float
        m_pixelsFloat.resize(m_imageCount * 28 * 28);
        for (size_t i = 0; i < 28 * 28 * m_imageCount; ++i)
            m_pixelsFloat[i] = float(m_pixels[i]) / 255.0f;
 
        return true;
    }
 
    ~MNISTData ()
    {
        delete[] m_labelData;
        delete[] m_imageData;
    }
 
    size_t NumImages () const { return m_imageCount; }
 
    const float* GetImage (size_t index, uint8_t& label) const
    {
        label = m_labels[index];
        return &m_pixelsFloat[index * 28 * 28];
    }
 
private:

    uint8_t* m_labelData;
    uint8_t* m_imageData;
    size_t m_imageCount;
    uint8_t* m_labels;
    uint8_t* m_pixels;
    std::vector<float> m_pixelsFloat;
};