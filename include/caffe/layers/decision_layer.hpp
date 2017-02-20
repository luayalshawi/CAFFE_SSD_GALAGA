/*
 * decision_layer.hpp
 *
 *  Created on: Jan 29, 2017
 *      Author: luay
 */

#ifndef CAFFE_DECISION_LAYER_HPP_
#define CAFFE_DECISION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class DecisionLayer : public Layer<Dtype> {
 public:
  explicit DecisionLayer(const LayerParameter& param): Layer<Dtype>(param) {}
//  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Decision"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
      NOT_IMPLEMENTED;
  }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	    NOT_IMPLEMENTED;
  }
};

}// namespace caffe



#endif // CAFFE_DECISION_LAYER_HPP_
