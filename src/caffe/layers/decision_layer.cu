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

			for (int i = 0; i < count; i=i+7) {


				//[image_id, label, confidence, xmin, ymin, xmax, ymax]
				int label = bottom_data[i+1];
				float conf = bottom_data[i+2];
				float xmin = bottom_data[i+3];
				float ymin =bottom_data[i+4];
				float xmax =bottom_data[i+5];
				float ymax = bottom_data[i+6];
				struct coordinate coord;
				coord.xmin = xmin;
				coord.ymin = ymin;
				coord.xmax = xmax;
				coord.ymax = ymax;

				LOG(WARNING) << "------START--------------";
				LOG(WARNING) << "image_id:"<<bottom_data[i];
				LOG(WARNING) << "label:"<<label;
				LOG(WARNING) << "confidence:"<<conf;
				LOG(WARNING) << "xmin:"<<xmin;
				LOG(WARNING) << "ymin:"<<ymin;
				LOG(WARNING) << "xmax:"<<xmax;
				LOG(WARNING) << "ymax:"<<ymax;
				LOG(WARNING) << "--------END--------------";
				if (label == 1)
				{
					playerCoord[xmin] = coord;
					continue;

				}
				else if (label==16 || label==13)//ignore player's bullets and explosion
				{
					continue;
				}

				detections[ymin] = coord;

			}

			//logic for decision


			LOG(WARNING) <<playerCoord.begin()->first;

			// we get the player since player has the smallest x-value
			// we don't want to get life located on the far right of the game
			struct coordinate curr_player = playerCoord.begin()->second;
			LOG(WARNING) << "player.x:" << curr_player.xmin<<","<<  curr_player.xmax;

			struct coordinate enemy;

			std::map < float, struct coordinate > ::reverse_iterator it;// to start with largest y coordinates

			//defense
			for ( it = detections.rbegin(); it != detections.rend(); ++it )
			{

				LOG(WARNING) << it->first << ":" << it->second.xmin<<","<< it->second.xmax;

				//check how close is it to the player y-coordinate
				//xmin player, xmax enemy = right
				float delXRight = curr_player.xmin - it->second.xmax;
				float delXLeft = curr_player.xmax - it->second.xmin;

				if (it->second.ymax >= 0.55)
				{

					if (delXRight >=0.0 && delXRight <=0.05)
					{
						top_data[0] = 0.0;//right
						return;
					}
					if (delXLeft >=0.0 && delXLeft <=0.05)
					{
						top_data[0] = 1.0;//left
						return;
					}
				}
				//shoot playerX ~= enemyX
				if((delXRight>=0.0 && delXRight <=0.01)  || (delXLeft >=0.0 && delXLeft <=0.01)){
					top_data[0] = 2.0;//shoot
					return;
				}

				//enemy below the player
//				if(it->second.ymin > curr_player.ymin)
//				{
//					top_data[0] = -1;
//					return;
//				}

				//keep record of the last enemy after the loop
				if(it->second.xmin>0)
				{
					enemy.xmin = it->second.xmin;
					enemy.xmax = it->second.xmax;
					enemy.ymin = 0.0;
					enemy.ymax = 0.0;
				}


			}

			//if you don't see player -- do nothing
			if(curr_player.xmin==0 && curr_player.xmax==0)
			{
				top_data[0] = -1;
				return;
			}

			//prepare to attack
			if ( (enemy.xmin > curr_player.xmin) || (curr_player.xmin <= 0.16))
			{
				top_data[0] = 10.0;//right
				return;
			}
			LOG(WARNING) <<"enemy.xmin:"<<enemy.xmin<<", enemy.xmax:"<<enemy.xmax;

			if(enemy.xmin < curr_player.xmin)
			{
				top_data[0] = 100.0;//left
				return;
			}
			top_data[0] = -1;
	}
INSTANTIATE_LAYER_GPU_FUNCS(DecisionLayer);

}  // namespace caffe
