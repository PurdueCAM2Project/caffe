#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YoloLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  for(int j = 0; j < bottom.size(); j++){
    int max = bottom[j]->num_axes();
    for(int i = 0; i < max; i++){
      fprintf(stdout,"bottom[%d]->shape(%d) = %d\n",j,i,bottom[j]->shape(i));
    }
  }

  //TODO: grab these from protoxt
  num_ = bottom[0]->shape(0); // batch size
  k_ = 5; // number of anchors: could be derived from provided anchor boxes in prototxt
  image_input_size_ = 416; // image input size
  num_classes_ = 200; // 
  result_side_size_ = image_input_size_ / 32; // downsample by 32: for 416 is 13x13
  num_bboxes_ = k_ * pow(result_side_size_,2); // total number of bounding boxes
  loc_size_ = (num_classes_ + 5)*k_;// # of float-point values at each location on the output map
  input_size_ = loc_size_ * std::pow(result_side_size_,2);
  fprintf(stdout,"num_bboxes: %d\n",num_bboxes_);

  // read in from prototxt
  // normalization_ = this->layer_param_.loss_param().normalize() ?
  //   LossParameter_NormalizationMode_VALID :
  //   LossParameter_NormalizationMode_BATCH_SIZE;
  normalization_ = LossParameter_NormalizationMode_VALID;

  CHECK_EQ(input_size_, bottom[0]->shape(1)*bottom[0]->shape(2)*bottom[0]->shape(3))
    << "The output size is incorrect.";
  
  // use the anchors for appropriat dataset
  // TODO: allow definition to be read in from prototxt 
  float anchor_init[] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
  for(int i = 0; i < k_/2; i++){
    anchors_.push_back(std::make_pair(anchor_init[2*i],anchor_init[2*i+1]));
  }
  
  // -=-=-=-=-=-=-=-=-=-=-
  // initialize loc_layer
  // -=-=-=-=-=-=-=-=-=-=-

  vector<int> loss_shape(1, 1);
  // Set up localization loss layer.
  loc_weight_ = 5.0; //multibox_loss_param.loc_weight();
  // loc_loss_type_ = multibox_loss_param.loc_loss_type();
  conf_weight_ = .5; //multibox_loss_param.conf_weight();

  // fake shape
  vector<int> loc_shape(1, 1); 
  loc_shape.push_back(4);
  loc_pred_.Reshape(loc_shape);
  loc_gt_.Reshape(loc_shape);
  loc_bottom_vec_.push_back(&loc_pred_);
  loc_bottom_vec_.push_back(&loc_gt_);
  loc_loss_.Reshape(loss_shape);
  loc_top_vec_.push_back(&loc_loss_);

  if(true){
    // always use euclidean loss layer
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_l2_loc");
    layer_param.set_type("EuclideanLoss");
    layer_param.add_loss_weight(loc_weight_);
    loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
  }else{
    LOG(FATAL) << "Unknown localization loss type.";
  }

  // -=-=-=-=-=-=-=-=-=-=-
  // initialize conf_layer
  // -=-=-=-=-=-=-=-=-=-=-

  // Set up confidence loss layer.
  // conf_loss_type_ = multibox_loss_param.conf_loss_type();
  //if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
  if (true) {
    conf_bottom_vec_.push_back(&conf_pred_);
    conf_bottom_vec_.push_back(&conf_gt_);
    conf_loss_.Reshape(loss_shape);
    conf_top_vec_.push_back(&conf_loss_);

    // always use softmax
    LayerParameter layer_param;
    layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
    layer_param.set_type("SoftmaxWithLoss");
    layer_param.add_loss_weight(Dtype(1.));
    layer_param.mutable_loss_param()->set_normalization(
							LossParameter_NormalizationMode_NONE);
    SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
    softmax_param->set_axis(1);
    // Fake reshape.
    vector<int> conf_shape(1, 1);
    conf_gt_.Reshape(conf_shape);
    conf_shape.push_back(num_classes_);
    conf_pred_.Reshape(conf_shape);
    conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
  }else{
    LOG(FATAL) << "Unknown confidence loss type.";
  }

}

template <typename Dtype>
void YoloLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top); //not run b/c 1st axis size is not equal for data and labels
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  num_gt_ = bottom[1]->height();
  top[0]->Reshape(loss_shape);

}

template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  // for(int j = 0; j < bottom.size(); j++){
  //   int max = bottom[j]->num_axes();
  //   for(int i = 0; i < max; i++){
  //     fprintf(stdout,"bottom[%d]->shape(%d) = %d\n",j,i,bottom[j]->shape(i));
  //   }
  // }

  //create boundingbox
  // Retrieve all predictions.
  vector<vector<NormalizedBBox> > all_loc_preds;
  vector<ConfMap> all_conf_preds;
  // num_bboxes,num_classes

  // Retrieve all ground truth.
  map<int, vector<NormalizedBBox> > all_gt_bboxes;
  GetGroundTruth(gt_data, num_gt_, 0, false, &all_gt_bboxes);

  // get locations and predictions of bounding boxes
  GetLocConfPredictions_yolo(loc_data,num_bboxes_,loc_size_,
	num_,k_,num_classes_,result_side_size_,&all_loc_preds,&all_conf_preds);
  
  // // Find matches between source bboxes and ground truth bboxes.
  // vector<map<int, vector<float> > > all_match_overlaps;
  // FindMatches(all_loc_preds, all_gt_bboxes, prior_bboxes, prior_variances,
  //             multibox_loss_param_, &all_match_overlaps, &all_match_indices_);

  num_matches_ = 0;
  // LOSS for BOUNDING BOXES
  if (num_matches_ >= 1) { //num matches is number of errors... i think
    // Form data to pass on to loc_loss_layer_.
    vector<int> loc_shape(2);
    loc_shape[0] = 1;
    loc_shape[1] = num_matches_ * 4;
    loc_pred_.Reshape(loc_shape);
    loc_gt_.Reshape(loc_shape);
    Dtype* loc_pred_data = loc_pred_.mutable_cpu_data();
    Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();
    // EncodeLocPrediction(all_loc_preds, all_gt_bboxes, all_match_indices_,
    //                     prior_bboxes, prior_variances, multibox_loss_param_,
    //                     loc_pred_data, loc_gt_data);
    loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
    loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
  } else {
    // skip the forward pass and sent result to 0
    loc_loss_.mutable_cpu_data()[0] = 0;
  }

  //LOSS for CONFIDENCE
  



  // set the loss for this layer
  top[0]->mutable_cpu_data()[0] = 0;
  // tempararily leave num_mathes_ = 0 until the num_matches_ outpset is properly set
  if(true){
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(normalization_, num_, 1, num_matches_);
    // fprintf(stdout,"\n\nnormalizer: %.3f loc_weight_: %.3f loc_loss_.cpu_data()[0]: %.3f\n",normalizer,
    // 	    loc_weight_,loc_loss_.cpu_data()[0]);
    top[0]->mutable_cpu_data()[0] += loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
  }
  if(true){
    Dtype normalizer = LossLayer<Dtype>::GetNormalizer(normalization_, num_, 1, num_matches_);
    // fprintf(stdout,"normalizer: %.3f loc_weight_: %.3f conf_loss_.cpu_data()[0]: %.3f\n",normalizer,
    // 	    loc_weight_,conf_loss_.cpu_data()[0]);
    top[0]->mutable_cpu_data()[0] += conf_weight_ * conf_loss_.cpu_data()[0] / normalizer;
  }
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Scale gradient.
  Dtype loss_weight = top[0]->cpu_diff()[0];

}

INSTANTIATE_CLASS(YoloLossLayer);
REGISTER_LAYER_CLASS(YoloLoss);

}  // namespace caffe
