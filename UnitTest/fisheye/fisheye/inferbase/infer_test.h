//
// Created by 李大冲 on 2019-09-23.
//

#ifndef RETINANET_INFER_INFER_H
#define RETINANET_INFER_INFER_H

#endif //RETINANET_INFER_INFER_H

class Infer_RT {
public:
    Infer_RT();

    void initia(const char *engine_file, const char *name, int info[]);

    void process_(float *img_input, float *scores, float *boxes, float *classes,
                  int batch, const char *name);

    void run();

    bool has_inita;
private:
    retinanet::Engine *engine;
    void *data_d, *scores_d, *boxes_d, *classes_d;
    vector<float> data;
    int channels, num_det, inputSize_0, inputSize_1;
};