/*
 * decision_layer.cpp
 *
 *  Created on: Jan 29, 2017
 *      Author: luay
 */


#include <vector>
#include <string>
#include "caffe/layers/decision_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void DecisionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) {
		LOG(WARNING) << "DecisionLayer: Reshape";
		vector<int> top_shape;
		// Since the number of bboxes to be kept is unknown before nms, we manually
		// set it to (fake) 1.
		//top_shape.push_back(1);
		// Each row is a 7 dimension vector, which stores
		// [image_id, label, confidence, xmin, ymin, xmax, ymax]
		top_shape.push_back(1);
		top_shape.push_back(1);
		top[0]->Reshape(top_shape);
	}
	template <typename Dtype>
	void DecisionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
	{
		LOG(WARNING) << "Forward_cpu";
		const Dtype* bottom_data = bottom[0]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		const int count = bottom[0]->count();

//		for (int i = 0; i < count; ++i) {
//			LOG(WARNING) << "Forward_cpu";
//			top_data[i%7] = i+10.0;
//		}

//		LOG(WARNING) << "Count:"<<count;
//		for (int i = 0; i < count; ++i) {
//			// [image_id, label, confidence, xmin, ymin, xmax, ymax]
//			LOG(WARNING) << "Bottom Data: "<<bottom_data[i];
//		}

	}

//#ifdef CPU_ONLY
//STUB_GPU_FORWARD(DecisionLayer);
//#endif
#ifdef CPU_ONLY
STUB_GPU_FORWARD(DecisionLayer, Forward);
#endif
INSTANTIATE_CLASS(DecisionLayer);
REGISTER_LAYER_CLASS(Decision);

}  // namespace caffe
