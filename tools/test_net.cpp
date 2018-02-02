#include <cuda_runtime.h>
#include <Python.h>
#include <cstring>
#include <cstdlib>
#include <vector>

#include <fcntl.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace std;

const char* pipe_in = "/dev/shm/yolo_in";
const char* pipe_out = "/dev/shm/yolo_out";

inline void load_input_blob(Blob<float>* blob,int fd){
  int num,tot,offset;
  float* b0 = blob->mutable_cpu_data();
  tot = 0;
  while(tot < 540000){ //270000*sizeof(float)
    offset = tot/2; //tot/sizeof(float)
    num = read(fd, (float*)(b0 + offset),540000 - tot);
    if (num < 0){
      cerr << "Error: " << strerror(errno);
      break;
    }
    tot += num;
  }
}

inline void model_forward(Net<float>& caffe_net, \
			vector<Blob<float>*> bottom,int fd){

  int curr_size,tot,num,offset;
  //int write_size[] = {1797342,35768,71536};
  //int write_size[] = {4096,153600,115200,16384,9216,921600};
  const vector<Blob<float>*>& result = caffe_net.Forward(bottom, NULL);
  //cout << "result[idx].size(): " << result.size() << endl;
  for(int idx = 0; idx < result.size(); idx++){
    tot = 0; // reset total counter
    curr_size = result[idx]->count() * sizeof(float);
    while(tot < curr_size){
      offset = tot/sizeof(float);
      num = write(fd, (float*)(result[idx]->cpu_data() + offset),curr_size - tot);
      if (num < 0){
	cerr << "Error: " << strerror(errno);
	break;
      }
      tot += num;
    }
  }
}



int main(int argc, char** argv) {

  // CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  // CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  //-----------------------------
  //  INIT & LOAD IN PARAMETERS
  //-----------------------------
  if (argc < 4 || argc > 6) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
	       << "[CPU/GPU] [Device ID]";
    return 1;
  }

  //-----------------------------
  //       SET GPU/CPU
  //-----------------------------
  if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 6) {
      device_id = atoi(argv[5]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  //-----------------------------
  //      LOAD CAFFE NET
  //-----------------------------
  Net<float> caffe_net(argv[1], caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(argv[2]);

  //-----------------------------
  //  CREATE BLOB(PROTO)(VECTOR)
  //-----------------------------
  Blob<float>* blob = new Blob<float>(1, 3,448,448);
  vector<Blob<float>*> bottom;
  bottom.push_back(blob);

  //-----------------------------
  //   PERPARE READ FIFO PIPE
  //-----------------------------
  int status = mkfifo(pipe_in,0666);
  if (status < 0){
    printf("\n%s \n",strerror(errno));
  }
  int fd_r = open(pipe_in,O_RDONLY);
  if (fd_r < 0){
    printf("\n%s \n",strerror(errno));
  }

  //-----------------------------
  //   PERPARE WRITE FIFO PIPE
  //-----------------------------
  status = mkfifo(pipe_out,0666);
  if (status < 0){
    printf("\n%s \n",strerror(errno));
  }
  int fd_w = open(pipe_out,O_WRONLY);
  if (fd_w < 0){
    printf("\n%s \n",strerror(errno));
  }
  


  while(true){

    //----------------------------
    //     LOAD IMG IN BLOB
    //----------------------------
    load_input_blob(bottom[0],fd_r);

    //-------------------------------
    //    FORWARD AND WRITE OUTPUT
    //-------------------------------
    model_forward(caffe_net, bottom, fd_w);
  }
  
  close(fd_r);
  close(fd_w);

  return 0;
}
