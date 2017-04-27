#include <vector>
#include <string>
#include "caffe/layers/decision_layer.hpp"
struct coordinate{
	float xmin;
	float ymin;
	float xmax;
	float ymax;
};
namespace caffe {

	template <typename Dtype>
	void DecisionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	    const vector<Blob<Dtype>*>& top) {
	    	LOG(WARNING) << "Forward_gpu";
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* top_data = top[0]->mutable_cpu_data();
			const int count = bottom[0]->count();
			LOG(WARNING) << "gpu Count:"<<count;
			std::map < float, struct coordinate > detections;//key:ymin
			std::map <float,struct coordinate> playerCoord;//key:xmin
//
//			int width = 1280;
//			int height = 720;
			for (int i = 0; i < count; i=i+7) {

//				LOG(WARNING) << "------START--------------";

				//[image_id, label, confidence, xmin, ymin, xmax, ymax]
				float conf = bottom_data[i+2];
				int label = bottom_data[i+1];
				float xmin = bottom_data[i+3];//;*width;
				float ymin =bottom_data[i+4];//;*height;
				float xmax =bottom_data[i+5];//;*width;
				float ymax = bottom_data[i+6];//;*height;
				struct coordinate coord;
				coord.xmin = xmin;
				coord.ymin = ymin;
				coord.xmax = xmax;
				coord.ymax = ymax;

//				LOG(WARNING) << "image_id:"<<bottom_data[i];
//				LOG(WARNING) << "label:"<<label;
//				LOG(WARNING) << "confidence:"<<conf;
//				LOG(WARNING) << "xmin:"<<xmin;
//				LOG(WARNING) << "ymin:"<<ymin;
//				LOG(WARNING) << "xmax:"<<xmax;
//				LOG(WARNING) << "ymax:"<<ymax;
//				LOG(WARNING) << "--------END--------------";
				if (label == 1)
				{
					playerCoord[xmin] = coord;
					continue;
				}else if (label==16)
				{
					continue;
				}

				detections[ymin] = coord;

				//top_data[i%7] = i+10.0;
			}
//			LOG(WARNING) << "gpu Count:"<<count;

			LOG(WARNING) <<playerCoord.begin()->first;
			//logic for decision
			float finalDec = 0;
			struct coordinate curr_player = playerCoord.begin()->second;

			std::map < float, struct coordinate > ::reverse_iterator it;

			for ( it = detections.rbegin(); it != detections.rend(); ++it )
			{

				LOG(WARNING) << it->first << ":" << it->second.xmin<<","<< it->second.xmax;   // string's value
				//LOG(WARNING) << "gpu Count:"<<count;

				//check how close is it to the player y-coordinate
				//xmin player, xmax enemy = right
				float delXRight = curr_player.xmin - it->second.xmax;
				float delXLeft = curr_player.xmax - it->second.xmin;
				// test
				top_data[0] = -1;

				if (it->second.ymax >= 0.40)
				{
					//either left or right
					if (delXRight >=0.0 && delXRight <=0.01)
					{
						top_data[0] = 0;
						// return;
					}
					if (delXLeft >=0.0 && delXLeft <=0.01)
					{
						top_data[0] = 1;
						// return;
					}
				}
				//shoot playerY ~= enemyY
				if((delXRight>=0.0 && delXRight <=0.01)  || (delXLeft >=0.0 && delXLeft <=0.01)){
					// test 
					if (top_data[0] == 0)
					{
						top_data[0] = 3;
					}
					else if (top_data[0] == 1)
					{
						top_data[0] = 4;
					}
					else
					{
						top_data[0] = 2;
						// return;
					}
				}
				// test
				if (top_data[0] != -1)
					return;
			}
			top_data[0] = -1;
	}
INSTANTIATE_LAYER_GPU_FUNCS(DecisionLayer);

}  // namespace caffe
