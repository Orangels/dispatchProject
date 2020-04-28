/*https://github.com/ReneNyffenegger/cpp-base64/blob/master/base64.cpp
   base64.cpp and base64.h

   base64 encoding and decoding with C++.

   Version: 1.01.00

   Copyright (C) 2004-2017 René Nyffenegger

   This source code is provided 'as-is', without any express or implied
   warranty. In no event will the author be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   René Nyffenegger rene.nyffenegger@adp-gmbh.ch

*/

#include "base64.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#define min(A, B)  ( (A) <= (B) ?  (A) : (B) )

static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";


static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

void base64_encode(const unsigned char *Data, int DataByte, char *strEncode) {
    //编码表
    const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    //返回值
    unsigned char Tmp[4] = {0};
    int LineLength = 0, idx = 0;
    for (int i = 0; i < (int) (DataByte / 3); i++) {
        Tmp[1] = *Data++;
        Tmp[2] = *Data++;
        Tmp[3] = *Data++;
        strEncode[idx++] = EncodeTable[Tmp[1] >> 2];
        strEncode[idx++] = EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
        strEncode[idx++] = EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
        strEncode[idx++] = EncodeTable[Tmp[3] & 0x3F];
        if (LineLength += 4, LineLength == 76) {
            strEncode[idx++] = '\r';
            strEncode[idx++] = '\n';
            LineLength = 0;
        }
    }
    //对剩余数据进行编码
    int Mod = DataByte % 3;
    if (Mod == 1) {
        Tmp[1] = *Data++;
        strEncode[idx++] = EncodeTable[(Tmp[1] & 0xFC) >> 2];
        strEncode[idx++] = EncodeTable[((Tmp[1] & 0x03) << 4)];
        strEncode[idx++] = '=';
        strEncode[idx++] = '=';
    } else if (Mod == 2) {
        Tmp[1] = *Data++;
        Tmp[2] = *Data++;
        strEncode[idx++] = EncodeTable[(Tmp[1] & 0xFC) >> 2];
        strEncode[idx++] = EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
        strEncode[idx++] = EncodeTable[((Tmp[2] & 0x0F) << 2)];
        strEncode[idx++] = '=';
    }
}

int mainv() {
    cv::Mat ResImg = cv::imread("base_demo1.jpg");
    std::vector <uchar> buffer;
    buffer.resize(static_cast<size_t>(ResImg.rows) * static_cast<size_t>(ResImg.cols));
    std::cout << ResImg.rows << 'c' << ResImg.cols << ';' << buffer.size() << ';';
    char *encoded = new char[buffer.size() + 1000];
    std::cout << "buffer.size:" << buffer.size() << std::endl;

    cv::imencode(".jpg", ResImg, buffer);
    auto *enc_msg = reinterpret_cast<unsigned char *>(buffer.data());
    base64_encode(enc_msg, buffer.size(), encoded);
    return 0;
}